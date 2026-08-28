#include "kv-stream.cuh"

#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "common.cuh"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

// -----------------------------------------------------------------------
// Resident-page cache: a uniform-per-layer slice of the device pool. Each
// (layer, slot) pair is either empty or mirrors one page-worth of tokens
// from the authoritative host KV cache for that layer. Which host page (if
// any) currently occupies a slot, and dispatching pages in and out, is the
// job of the FlashAttention dispatch added in a later stage - this cache
// only owns the geometry and the coarse "has anything changed under us"
// invalidation.
// -----------------------------------------------------------------------

struct ggml_cuda_kv_stream_resident_cache {
    size_t   scratch_bytes            = 0;
    size_t   page_bytes               = 0;
    uint32_t layer_count              = 0;
    uint32_t resident_pages_per_layer = 0;

    // [layer_count * resident_pages_per_layer], 1 if that device slot
    // currently mirrors a valid host page.
    std::vector<uint8_t> loaded;
};

static bool kv_stream_resident_cache_layout(
        size_t pool_bytes, size_t scratch_bytes, size_t page_bytes, uint32_t layer_count,
        uint32_t & resident_pages_per_layer) {
    if (page_bytes == 0 || layer_count == 0 || scratch_bytes >= pool_bytes ||
            scratch_bytes % page_bytes != 0) {
        return false;
    }

    const size_t resident_pages_total = (pool_bytes - scratch_bytes) / page_bytes;
    const size_t pages_per_layer      = resident_pages_total / layer_count;
    if (pages_per_layer == 0 || pages_per_layer > UINT32_MAX) {
        return false;
    }

    resident_pages_per_layer = uint32_t(pages_per_layer);
    return true;
}

static ggml_cuda_kv_stream_resident_cache * ggml_cuda_kv_stream_resident_cache_new(
        size_t pool_bytes, size_t scratch_bytes, size_t page_bytes, uint32_t layer_count) {
    uint32_t resident_pages_per_layer = 0;
    if (!kv_stream_resident_cache_layout(
            pool_bytes, scratch_bytes, page_bytes, layer_count, resident_pages_per_layer)) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_kv_stream_resident_cache;
    cache->scratch_bytes            = scratch_bytes;
    cache->page_bytes               = page_bytes;
    cache->layer_count               = layer_count;
    cache->resident_pages_per_layer = resident_pages_per_layer;
    cache->loaded.assign(size_t(layer_count) * resident_pages_per_layer, 0);
    return cache;
}

static void ggml_cuda_kv_stream_resident_cache_free(ggml_cuda_kv_stream_resident_cache * cache) {
    delete cache;
}

static void ggml_cuda_kv_stream_resident_cache_reset(ggml_cuda_kv_stream_resident_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    std::fill(cache->loaded.begin(), cache->loaded.end(), 0);
}

static uint32_t ggml_cuda_kv_stream_resident_cache_pages_per_layer(
        const ggml_cuda_kv_stream_resident_cache * cache) {
    return cache == nullptr ? 0 : cache->resident_pages_per_layer;
}

// Recomputes the resident/ring split for a device pool that keeps its total
// size fixed (`pool_bytes` unchanged) while the ring grows or shrinks to
// `scratch_bytes`. Always invalidates every resident page: with a uniform
// per-layer split, changing resident_pages_per_layer moves every layer's
// base offset, so page contents cannot be migrated in place.
static bool ggml_cuda_kv_stream_resident_cache_repartition(
        ggml_cuda_kv_stream_resident_cache * cache, size_t pool_bytes, size_t scratch_bytes) {
    if (cache == nullptr) {
        return false;
    }

    uint32_t resident_pages_per_layer = 0;
    if (!kv_stream_resident_cache_layout(
            pool_bytes, scratch_bytes, cache->page_bytes, cache->layer_count, resident_pages_per_layer)) {
        return false;
    }

    cache->scratch_bytes            = scratch_bytes;
    cache->resident_pages_per_layer = resident_pages_per_layer;
    cache->loaded.assign(size_t(cache->layer_count) * resident_pages_per_layer, 0);
    return true;
}

// -----------------------------------------------------------------------
// Transfer ring: a small set of device-memory slots, each page_bytes wide,
// used to stage host->device copies ahead of the FlashAttention dispatch
// consuming them. The copy stream is kept separate from the compute stream
// so a page transfer can overlap with attention over already-resident
// pages; the ready/consumed event pairs are what make that overlap safe:
//
//   copy stream:    wait(consumed[slot]) -> memcpyAsync -> record(ready[slot])
//   compute stream: wait(ready[slot])    -> read slot    -> record(consumed[slot])
//
// so a slot is never overwritten while the previous occupant is still being
// read, and never read before its transfer has actually landed.
// -----------------------------------------------------------------------

struct ggml_cuda_kv_stream_transfer_ring {
    char * pool_data      = nullptr; // first capacity_slots*page_bytes bytes of the device pool
    size_t page_bytes     = 0;
    uint32_t capacity_slots = 0;
    uint32_t active_slots   = 0;

    cudaStream_t copy_stream = nullptr;
    std::vector<cudaEvent_t> ready;
    std::vector<cudaEvent_t> consumed;
};

static void ggml_cuda_kv_stream_transfer_ring_free(ggml_cuda_kv_stream_transfer_ring * ring) {
    if (ring == nullptr) {
        return;
    }
    for (cudaEvent_t event : ring->ready) {
        if (event != nullptr) {
            (void) cudaEventDestroy(event);
        }
    }
    for (cudaEvent_t event : ring->consumed) {
        if (event != nullptr) {
            (void) cudaEventDestroy(event);
        }
    }
    if (ring->copy_stream != nullptr) {
        (void) cudaStreamDestroy(ring->copy_stream);
    }
    delete ring;
}

static ggml_cuda_kv_stream_transfer_ring * ggml_cuda_kv_stream_transfer_ring_new(
        void * pool_data, size_t page_bytes, uint32_t capacity_slots) {
    if (pool_data == nullptr || page_bytes == 0 || capacity_slots == 0) {
        return nullptr;
    }

    auto * ring = new ggml_cuda_kv_stream_transfer_ring;
    ring->pool_data       = static_cast<char *>(pool_data);
    ring->page_bytes      = page_bytes;
    ring->capacity_slots  = capacity_slots;
    ring->active_slots    = capacity_slots;
    ring->ready.resize(capacity_slots, nullptr);
    ring->consumed.resize(capacity_slots, nullptr);

    if (cudaStreamCreateWithFlags(&ring->copy_stream, cudaStreamNonBlocking) != cudaSuccess) {
        (void) cudaGetLastError();
        ggml_cuda_kv_stream_transfer_ring_free(ring);
        return nullptr;
    }
    for (uint32_t slot = 0; slot < capacity_slots; ++slot) {
        if (cudaEventCreateWithFlags(&ring->ready[slot], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventCreateWithFlags(&ring->consumed[slot], cudaEventDisableTiming) != cudaSuccess) {
            (void) cudaGetLastError();
            ggml_cuda_kv_stream_transfer_ring_free(ring);
            return nullptr;
        }
        // A slot that has never been used yet must not block its first copy.
        CUDA_CHECK(cudaEventRecord(ring->consumed[slot], ring->copy_stream));
    }

    return ring;
}

static bool ggml_cuda_kv_stream_transfer_ring_set_active_slots(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t active_slots) {
    if (ring == nullptr || active_slots == 0 || active_slots > ring->capacity_slots) {
        return false;
    }
    ring->active_slots = active_slots;
    return true;
}

// Queues an async host->device copy of one page into `slot`, after waiting
// for the slot's previous occupant to be fully consumed by the compute
// stream. Records ready[slot] once the copy is queued.
static bool ggml_cuda_kv_stream_transfer_ring_copy_page_async(
        ggml_cuda_kv_stream_transfer_ring * ring, uint32_t slot, const void * host_src, size_t bytes) {
    if (ring == nullptr || slot >= ring->active_slots || host_src == nullptr || bytes > ring->page_bytes) {
        return false;
    }

    CUDA_CHECK(cudaStreamWaitEvent(ring->copy_stream, ring->consumed[slot], 0));
    CUDA_CHECK(cudaMemcpyAsync(
            ring->pool_data + size_t(slot)*ring->page_bytes, host_src, bytes,
            cudaMemcpyHostToDevice, ring->copy_stream));
    CUDA_CHECK(cudaEventRecord(ring->ready[slot], ring->copy_stream));
    return true;
}

// Makes `compute_stream` wait for slot's pending copy (if any) to land
// before reading it.
static bool ggml_cuda_kv_stream_transfer_ring_wait_ready(
        const ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream, uint32_t slot) {
    if (ring == nullptr || slot >= ring->active_slots) {
        return false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(compute_stream, ring->ready[slot], 0));
    return true;
}

// Releases `slot` back to the copy stream once `compute_stream` is done
// reading it.
static bool ggml_cuda_kv_stream_transfer_ring_mark_consumed(
        ggml_cuda_kv_stream_transfer_ring * ring, cudaStream_t compute_stream, uint32_t slot) {
    if (ring == nullptr || slot >= ring->active_slots) {
        return false;
    }
    CUDA_CHECK(cudaEventRecord(ring->consumed[slot], compute_stream));
    return true;
}

static void * ggml_cuda_kv_stream_transfer_ring_slot_data(
        const ggml_cuda_kv_stream_transfer_ring * ring, uint32_t slot) {
    if (ring == nullptr || slot >= ring->active_slots) {
        return nullptr;
    }
    return ring->pool_data + size_t(slot)*ring->page_bytes;
}

// -----------------------------------------------------------------------
// Online-softmax fixup kernels: combine one new partial FlashAttention
// result (computed against a just-streamed page) into a running
// accumulator for one (token, head) output row, using the same
// max-rescaling as llama_kv_stream_softmax_merge (src/llama-kv-stream-softmax.h)
// and the existing flash_attn_combine_results (fattn-common.cuh) - applied
// incrementally, one part at a time, instead of over a fixed list gathered
// up front, since streamed pages arrive one at a time across separate
// kernel launches.
//
// Callers must zero-initialize the accumulator before the first partial for
// a row is combined into it: acc_meta = {-INFINITY, 0.0f}, acc_numerator
// all zero. exp(-INFINITY - anything finite) evaluates to 0, so the first
// combine correctly reduces to acc := part with no special-cased first step.
// -----------------------------------------------------------------------

template <int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void kv_stream_accumulate_partial(
        const float  * __restrict__ part_numerator, // [D]
        const float2 * __restrict__ part_meta,       // {max_logit, denominator}
        float  * __restrict__ acc_numerator,          // [D], read-modify-write
        float2 * __restrict__ acc_meta) {              // {max_logit, denominator}, read-modify-write
    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    __shared__ float2 old_meta;
    __shared__ float2 new_meta;
    __shared__ float  combined_max;
    if (tid == 0) {
        old_meta     = *acc_meta;
        new_meta     = *part_meta;
        combined_max = fmaxf(old_meta.x, new_meta.x);
    }
    __syncthreads();

    const float old_scale = expf(old_meta.x - combined_max);
    const float new_scale = expf(new_meta.x - combined_max);

    acc_numerator[tid] = old_scale*acc_numerator[tid] + new_scale*part_numerator[tid];
    if (tid == 0) {
        acc_meta->x = combined_max;
        acc_meta->y = old_scale*old_meta.y + new_scale*new_meta.y;
    }
}

template <int D> // D == head size
__launch_bounds__(D, 1)
static __global__ void kv_stream_normalize_result(
        const float  * __restrict__ acc_numerator, // [D]
        const float2 * __restrict__ acc_meta,        // {max_logit, denominator}
        float  * __restrict__ dst) {                  // [D]
    const int tid = threadIdx.x;
    __builtin_assume(tid < D);
    dst[tid] = acc_numerator[tid] / acc_meta->y;
}

// -----------------------------------------------------------------------
// Runtime: owns the pinned host buffer type plus the fixed-size device pool
// carved into the resident cache and transfer ring above.
// -----------------------------------------------------------------------

struct ggml_backend_cuda_kv_stream_runtime {
    int    device     = 0;
    size_t pool_bytes = 0;
    void * pool_data  = nullptr; // fixed-size device pool: [0, scratch_bytes) is the transfer
                                  // ring, [scratch_bytes, pool_bytes) is the resident cache

    ggml_cuda_kv_stream_resident_cache * resident_cache = nullptr;
    ggml_cuda_kv_stream_transfer_ring  * transfer_ring  = nullptr;

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

    ggml_cuda_set_device(runtime->device);
    ggml_cuda_kv_stream_transfer_ring_free(runtime->transfer_ring);
    ggml_cuda_kv_stream_resident_cache_free(runtime->resident_cache);
    if (runtime->pool_data != nullptr) {
        CUDA_CHECK(cudaFree(runtime->pool_data));
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

// A write to the authoritative host storage may change what any resident
// device page mirrors; the cheap, always-correct response is to invalidate
// everything and let the (later-stage) dispatch reload lazily, rather than
// track precisely which pages changed.

void ggml_backend_cuda_kv_stream_buffer_memset(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    ggml_cuda_kv_stream_resident_cache_reset(context->runtime->resident_cache);
    memset(static_cast<char *>(tensor->data) + offset, value, size);
}

void ggml_backend_cuda_kv_stream_buffer_set(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    ggml_cuda_kv_stream_resident_cache_reset(context->runtime->resident_cache);
    memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void ggml_backend_cuda_kv_stream_buffer_get(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

void ggml_backend_cuda_kv_stream_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = static_cast<ggml_backend_cuda_kv_stream_buffer_context *>(buffer->context);
    ggml_cuda_kv_stream_resident_cache_reset(context->runtime->resident_cache);
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
    if (params.device < 0 || params.device >= ggml_backend_cuda_get_device_count() ||
            params.pool_bytes == 0 || params.page_bytes == 0 || params.stage_slots == 0 ||
            params.layer_count == 0 ||
            params.page_bytes > std::numeric_limits<size_t>::max()/params.stage_slots) {
        return nullptr;
    }

    const size_t scratch_bytes = params.page_bytes*size_t(params.stage_slots);
    if (scratch_bytes >= params.pool_bytes) {
        return nullptr;
    }

    auto * runtime = new ggml_backend_cuda_kv_stream_runtime;
    runtime->device     = params.device;
    runtime->pool_bytes = params.pool_bytes;

    ggml_cuda_set_device(params.device);
    const cudaError_t error = cudaMalloc(&runtime->pool_data, params.pool_bytes);
    if (error != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB KV streaming pool on device %d failed: %s\n",
                __func__, params.pool_bytes/1024.0/1024.0, params.device, cudaGetErrorString(error));
        delete runtime;
        return nullptr;
    }

    runtime->resident_cache = ggml_cuda_kv_stream_resident_cache_new(
            params.pool_bytes, scratch_bytes, params.page_bytes, params.layer_count);

    // The ring's own slot/event arrays are sized to the largest the ring
    // could ever grow to (the whole pool), so a later repartition can widen
    // the *active* slot count up to that bound without reallocating.
    const size_t max_ring_slots = params.pool_bytes/params.page_bytes;
    runtime->transfer_ring = (runtime->resident_cache == nullptr || max_ring_slots == 0 ||
            max_ring_slots > UINT32_MAX) ? nullptr :
        ggml_cuda_kv_stream_transfer_ring_new(runtime->pool_data, params.page_bytes, uint32_t(max_ring_slots));
    if (runtime->transfer_ring != nullptr) {
        if (!ggml_cuda_kv_stream_transfer_ring_set_active_slots(runtime->transfer_ring, params.stage_slots)) {
            ggml_cuda_kv_stream_transfer_ring_free(runtime->transfer_ring);
            runtime->transfer_ring = nullptr;
        }
    }
    if (runtime->resident_cache == nullptr || runtime->transfer_ring == nullptr) {
        ggml_cuda_kv_stream_resident_cache_free(runtime->resident_cache);
        ggml_cuda_kv_stream_transfer_ring_free(runtime->transfer_ring);
        CUDA_CHECK(cudaFree(runtime->pool_data));
        delete runtime;
        return nullptr;
    }

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
    ggml_backend_cuda_kv_stream_params params;
    params.device      = device;
    params.pool_bytes  = pool_bytes;
    params.page_bytes  = page_bytes;
    params.stage_slots = stage_slots;
    params.layer_count = layer_count;
    return ggml_backend_cuda_kv_stream_runtime_new(params);
}

ggml_backend_buffer_type_t ggml_backend_cuda_kv_stream_buffer_type(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    return runtime == nullptr ? nullptr : &runtime->buffer_type;
}

size_t ggml_backend_cuda_kv_stream_pool_bytes(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    return runtime == nullptr ? 0 : runtime->pool_bytes;
}

uint32_t ggml_backend_cuda_kv_stream_resident_pages_per_layer(ggml_backend_cuda_kv_stream_runtime_t runtime) {
    return runtime == nullptr ? 0 :
        ggml_cuda_kv_stream_resident_cache_pages_per_layer(runtime->resident_cache);
}

bool ggml_backend_cuda_kv_stream_repartition(
        ggml_backend_cuda_kv_stream_runtime_t runtime, uint32_t stage_slots) {
    if (runtime == nullptr || runtime->resident_cache == nullptr || runtime->transfer_ring == nullptr ||
            stage_slots == 0 || runtime->transfer_ring->page_bytes > std::numeric_limits<size_t>::max()/stage_slots) {
        return false;
    }

    const size_t scratch_bytes = runtime->transfer_ring->page_bytes*size_t(stage_slots);
    if (scratch_bytes >= runtime->pool_bytes) {
        return false;
    }

    ggml_cuda_set_device(runtime->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!ggml_cuda_kv_stream_transfer_ring_set_active_slots(runtime->transfer_ring, stage_slots) ||
            !ggml_cuda_kv_stream_resident_cache_repartition(runtime->resident_cache, runtime->pool_bytes, scratch_bytes)) {
        return false;
    }
    return true;
}
