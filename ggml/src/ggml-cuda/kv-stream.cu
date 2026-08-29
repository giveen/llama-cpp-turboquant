#include "kv-stream.cuh"

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "common.cuh"

#include <atomic>
#include <cstring>

// TODO(kv-stream): the active mechanism today is just the pinned-host buffer
// type below: FlashAttention reads the whole cached context directly via
// CUDA UVA on every decode step, which already works and measurably reduces
// VRAM (see docs/adaptive-kv-stream-session-summary.md), but re-reads
// everything over PCIe each step with no host<->device overlap.
//
// An earlier version of this file also built a resident-page cache and an
// async transfer ring (plus two device-side online-softmax fixup kernels)
// meant to chunk that read into page-sized host->device copies overlapped
// with compute - but nothing in fattn.cu ever called into them, so they
// were pure dead weight: a full-size, always-allocated cudaMalloc device
// pool that nothing read or wrote. Removed rather than kept as unused
// scaffolding (see the session history for the "reduce LOC" discussion);
// ggml_backend_cuda_kv_stream_params still accepts page_bytes/stage_slots/
// layer_count for API stability across the generic
// ggml_backend_kv_stream_runtime_new_for_device_t plugin signature, but the
// CUDA implementation ignores them for now.
//
// Finishing this would mean a chunked dispatch in fattn.cu: loop over
// pages, copy each one host->device (prefetching the next page while the
// kernel still computes on the current one), and merge partial per-page
// results via running online-softmax (same math as
// llama_kv_stream_softmax_merge used to implement, before it was removed
// for the same reason - it was never callable from the actual device
// kernel path). This is a new dispatch path across every existing FA
// kernel variant (VEC/TILE/MMA, every K/V type pair, every head dim), not
// a small tweak - realistically 300-500+ lines of new host+device code,
// unverifiable without a CUDA compiler and GPU. Expected payoff: less PCIe
// overhead via prefetch/compute overlap (uncertain how much - modern UVA
// page migration is already decent), better scaling at very large
// contexts, and it's the only design that would ever generalize to
// Vulkan/Metal (neither has anything like CUDA's UVA). Deliberately not
// started - see the "ring work" discussion in the session history for the
// full tradeoff.

// -----------------------------------------------------------------------
// Runtime: owns the pinned host buffer type used for the authoritative KV
// cache storage of streaming-eligible layers.
// -----------------------------------------------------------------------

struct ggml_backend_cuda_kv_stream_runtime {
    int device = 0;

    std::atomic<uint32_t> references{1};
    ggml_backend_buffer_type buffer_type{};
};

namespace {

struct ggml_backend_cuda_kv_stream_buffer_context {
    ggml_backend_cuda_kv_stream_runtime_t runtime = nullptr;
    void * host_data = nullptr;
};

void ggml_backend_cuda_kv_stream_runtime_release(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    if (runtime == nullptr || runtime->references.fetch_sub(1, std::memory_order_acq_rel) != 1) {
        return;
    }
    delete runtime;
}

const char * ggml_backend_cuda_kv_stream_buffer_type_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_CUDA_NAME "_KV_Stream_Host";
}

void ggml_backend_cuda_kv_stream_buffer_free(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    CUDA_CHECK(cudaFreeHost(context->host_data));
    ggml_backend_cuda_kv_stream_runtime_release(context->runtime);
    delete context;
}

void * ggml_backend_cuda_kv_stream_buffer_base(ggml_backend_buffer_t buffer) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    return context->host_data;
}

void ggml_backend_cuda_kv_stream_buffer_memset(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memset(static_cast<char *>(tensor->data) + offset, value, size);
}

void ggml_backend_cuda_kv_stream_buffer_set(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void ggml_backend_cuda_kv_stream_buffer_get(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

void ggml_backend_cuda_kv_stream_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    memset(context->host_data, value, buffer->size);
}

const ggml_backend_buffer_i ggml_backend_cuda_kv_stream_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_kv_stream_buffer_free,
    /* .get_base        = */ ggml_backend_cuda_kv_stream_buffer_base,
    /* .init_tensor     = */ nullptr,
    /* .memset_tensor   = */ ggml_backend_cuda_kv_stream_buffer_memset,
    /* .set_tensor      = */ ggml_backend_cuda_kv_stream_buffer_set,
    /* .get_tensor      = */ ggml_backend_cuda_kv_stream_buffer_get,
    /* .set_tensor_2d   = */ nullptr,
    /* .get_tensor_2d   = */ nullptr,
    /* .cpy_tensor      = */ nullptr,
    /* .clear           = */ ggml_backend_cuda_kv_stream_buffer_clear,
    /* .reset           = */ nullptr,
};

ggml_backend_buffer_t ggml_backend_cuda_kv_stream_buffer_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * runtime = static_cast<ggml_backend_cuda_kv_stream_runtime_t>(buft->context);
    ggml_cuda_set_device(runtime->device);

    void * host_data = nullptr;
    const cudaError_t error = cudaHostAlloc(
        &host_data, size, cudaHostAllocMapped | cudaHostAllocWriteCombined);
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB pinned KV storage failed: %s\n",
                __func__, size/1024.0/1024.0, cudaGetErrorString(error));
        return nullptr;
    }

    runtime->references.fetch_add(1, std::memory_order_relaxed);
    auto * context = new ggml_backend_cuda_kv_stream_buffer_context{runtime, host_data};
    return ggml_backend_buffer_init(buft, ggml_backend_cuda_kv_stream_buffer_interface, context, size);
}

size_t ggml_backend_cuda_kv_stream_buffer_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
}

size_t ggml_backend_cuda_kv_stream_buffer_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    return ggml_nbytes(tensor);
}

bool ggml_backend_cuda_kv_stream_buffer_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

const ggml_backend_buffer_type_i ggml_backend_cuda_kv_stream_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_kv_stream_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_kv_stream_buffer_alloc,
    /* .get_alignment    = */ ggml_backend_cuda_kv_stream_buffer_alignment,
    /* .get_max_size     = */ nullptr,
    /* .get_alloc_size   = */ ggml_backend_cuda_kv_stream_buffer_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_kv_stream_buffer_is_host,
};

} // namespace

ggml_backend_cuda_kv_stream_runtime_t ggml_backend_cuda_kv_stream_runtime_new(
        const ggml_backend_cuda_kv_stream_params & params) {
    if (params.device < 0 || params.device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    auto * runtime = new ggml_backend_cuda_kv_stream_runtime;
    runtime->device = params.device;
    runtime->buffer_type = {
        /* .iface   = */ ggml_backend_cuda_kv_stream_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), params.device),
        /* .context = */ runtime,
    };
    return runtime;
}

void ggml_backend_cuda_kv_stream_runtime_free(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    ggml_backend_cuda_kv_stream_runtime_release(runtime);
}

void * ggml_backend_cuda_kv_stream_runtime_new_for_device(
        int device, size_t pool_bytes, size_t page_bytes, uint32_t stage_slots, uint32_t layer_count) {
    GGML_UNUSED(pool_bytes);
    GGML_UNUSED(page_bytes);
    GGML_UNUSED(stage_slots);
    GGML_UNUSED(layer_count);

    ggml_backend_cuda_kv_stream_params params;
    params.device = device;
    return ggml_backend_cuda_kv_stream_runtime_new(params);
}

ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    return runtime == nullptr ? nullptr : &runtime->buffer_type;
}
