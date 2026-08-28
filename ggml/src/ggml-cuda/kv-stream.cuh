#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <cstdint>

// Experimental block-granular KV cache streaming (CUDA/HIP/MUSA execution
// backend - this file builds unmodified for all three, same as the rest of
// ggml-cuda).
//
// The authoritative KV cache lives in the pinned host buffer type exposed
// here. A fixed-size device pool backs a uniform-per-layer resident-page
// cache plus a small async transfer ring (see llama_kv_stream_pool_layout_make
// in src/llama-kv-stream-config.h for the same pool-layout math, computed
// host-side before this runtime is created).
//
// Note on scope: this is a simplified design relative to the CUDA reference
// this was adapted from, which lets individual model layers give up resident
// pages (in round-robin fashion) so decode can keep its full active-page
// count even when an even per-layer split would fall short, plus precise
// per-row dirty tracking. Here every layer always gets the same
// resident_pages_per_layer, and a page is invalidated (not precisely
// tracked) whenever its host tensor is written. Simpler and easier to
// verify without a CUDA toolchain at hand; the tradeoff is streaming
// somewhat more eagerly under a tight pool instead of borrowing pages from
// other layers.
//
// The FlashAttention dispatch that actually reads/writes resident pages and
// drives the transfer ring is a later stage - this file is self-contained
// and unreferenced by the rest of the codebase until then. See
// src/llama-kv-stream-{config,plan,softmax}.h for the backend-agnostic
// planning layer this is the first execution backend for.

struct ggml_backend_cuda_kv_stream_runtime;
typedef ggml_backend_cuda_kv_stream_runtime * ggml_backend_cuda_kv_stream_runtime_t;

struct ggml_backend_cuda_kv_stream_params {
    int      device      = 0;
    size_t   pool_bytes  = 0; // total fixed-size device pool (resident pages + transfer ring)
    size_t   page_bytes  = 0; // bytes per resident page / transfer ring slot
    uint32_t stage_slots = 0; // transfer ring capacity, in pages
    uint32_t layer_count = 0; // logical KV layers sharing the resident cache
};

// Allocates the pinned host buffer type and the fixed-size device pool for
// `params.device`, carved into a uniform-per-layer resident-page cache and a
// `params.stage_slots`-deep transfer ring. Returns nullptr on any allocation
// or layout failure - the caller should fall back to the ordinary
// non-streaming KV cache path.
ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
        const ggml_backend_cuda_kv_stream_params & params);

// Same as above, but takes plain scalar arguments instead of the CUDA-specific
// params struct, and returns/takes an opaque void* handle - this is the shape
// resolved by llama-kv-cache.cpp through ggml_backend_reg_get_proc_address
// (registered under "ggml_backend_kv_stream_runtime_new_for_device" in
// ggml-cuda.cu), so that generic code never needs to include this header.
void * ggml_backend_cuda_kv_stream_runtime_new_for_device(
        int device, size_t pool_bytes, size_t page_bytes, uint32_t stage_slots, uint32_t layer_count);

void ggml_backend_cuda_kv_stream_runtime_free(ggml_backend_cuda_kv_stream_runtime_t runtime);

// The pinned host buffer type backing the authoritative KV cache tensors.
// The KV cache requests this instead of the ordinary CUDA device buffer
// type for a layer when streaming is enabled.
ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
        ggml_backend_cuda_kv_stream_runtime_t runtime);

size_t   ggml_backend_cuda_kv_stream_pool_bytes(ggml_backend_cuda_kv_stream_runtime_t runtime);
uint32_t ggml_backend_cuda_kv_stream_resident_pages_per_layer(ggml_backend_cuda_kv_stream_runtime_t runtime);

// Adjusts the resident/ring split by changing the ring to `stage_slots`
// pages, following llama_kv_stream_partition_adapt's decision (see
// src/llama-kv-stream-plan.h). Invalidates all resident pages - callers must
// expect a reload burst right after a successful repartition.
bool ggml_backend_cuda_kv_stream_repartition(
        ggml_backend_cuda_kv_stream_runtime_t runtime, uint32_t stage_slots);
