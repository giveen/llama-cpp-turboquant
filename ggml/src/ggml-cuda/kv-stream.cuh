#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>

// Experimental block-granular KV cache streaming (CUDA/HIP/MUSA execution
// backend - this file builds unmodified for all three, same as the rest of
// ggml-cuda).
//
// The authoritative KV cache lives in the pinned host buffer type exposed
// here; a fixed-size device pool is also allocated for the resident-page
// cache and async transfer ring that the FlashAttention dispatch consumes,
// added in a later stage once that dispatch exists. Until then this file is
// self-contained and unreferenced by the rest of the codebase - see
// src/llama-kv-stream-{config,plan,softmax}.h for the backend-agnostic
// planning layer this is the first execution backend for.

struct ggml_backend_cuda_kv_stream_runtime;
typedef ggml_backend_cuda_kv_stream_runtime * ggml_backend_cuda_kv_stream_runtime_t;

struct ggml_backend_cuda_kv_stream_params {
    int    device     = 0;
    size_t pool_bytes = 0; // fixed-size device pool (resident pages + transfer ring)
};

// Allocates the pinned host buffer type and the fixed-size device pool for
// `params.device`. Returns nullptr on any allocation failure - the caller
// should fall back to the ordinary non-streaming KV cache path.
ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
        const ggml_backend_cuda_kv_stream_params & params);

void ggml_backend_cuda_kv_stream_runtime_free(ggml_backend_cuda_kv_stream_runtime_t runtime);

// The pinned host buffer type backing the authoritative KV cache tensors.
// The KV cache requests this instead of the ordinary CUDA device buffer
// type for a layer when streaming is enabled.
ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(
        ggml_backend_cuda_kv_stream_runtime_t runtime);

size_t ggml_backend_cuda_kv_stream_pool_bytes(ggml_backend_cuda_kv_stream_runtime_t runtime);
