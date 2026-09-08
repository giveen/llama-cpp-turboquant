#include "kvarn.cuh"
#include "turbo-quant.cuh"
#include "ggml-kvarn-quant.h"

#include <cuda_fp16.h>

// KVarN CUDA kernels. Mirrors the CPU reference in ggml-kvarn-quant.c
// exactly (same clamp constants, same iteration order) - see that file for
// the algorithm's derivation and origin.
//
// SEAL: one block per side (K, V), 128 threads/block, one 128x128 tile
// (64KB) held in dynamic shared memory - needs the extended shared-memory
// opt-in (CUDA_SET_SHARED_MEMORY_LIMIT) since 64KB exceeds the default
// 48KB static limit. Each thread owns one row/column index for the
// Sinkhorn balancing passes (independent per-row/per-column reductions,
// no cross-thread reduction needed) and one output row for quantization.
//
// MATERIALIZE: fully elementwise given precomputed scales - one thread per
// output row, no shared memory or synchronization needed at all.

#define KVARN_N 128

__device__ __forceinline__ float kvarn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Sample std (n-1 denominator) of column c of the ORIGINAL tile, rescaled by
// the current log_s_col/log_s_row (i.e. of "cur", without materializing it).
__device__ __forceinline__ float kvarn_std_col_cur(const float * tile, const float * inv_s_col, const float * inv_s_row, int c) {
    const float inv_sc = inv_s_col[c];
    float sum = 0.0f, sumsq = 0.0f;
    for (int r = 0; r < KVARN_N; r++) {
        float v = tile[r * (KVARN_N + 1) + c] * inv_sc * inv_s_row[r];
        sum += v; sumsq += v * v;
    }
    const float mean = sum / KVARN_N;
    const float var  = (sumsq - KVARN_N * mean * mean) / (KVARN_N - 1);
    return sqrtf(var > 0.0f ? var : 0.0f);
}

__device__ __forceinline__ float kvarn_std_row_cur(const float * tile, const float * inv_s_col, const float * inv_s_row, int r) {
    const float inv_sr = inv_s_row[r];
    const float * row = tile + r * (KVARN_N + 1);
    float sum = 0.0f, sumsq = 0.0f;
    for (int c = 0; c < KVARN_N; c++) {
        float v = row[c] * inv_sr * inv_s_col[c];
        sum += v; sumsq += v * v;
    }
    const float mean = sum / KVARN_N;
    const float var  = (sumsq - KVARN_N * mean * mean) / (KVARN_N - 1);
    return sqrtf(var > 0.0f ? var : 0.0f);
}

__device__ __forceinline__ float kvarn_cur_at(const float * tile, const float * log_s_col, const float * log_s_row, int r, int c) {
    return tile[r * KVARN_N + c] / (expf(log_s_col[c]) * expf(log_s_row[r]));
}

// Runs Sinkhorn balancing + quantize-and-store on an already-loaded 128x(128+1)
// -strided tile (thread t owns row t of the tile, and writes record row t on
// output). Shared by k_kvarn_seal (single flat/strided source) and
// k_kvarn_cpy_seal (mixed persistent-tail/freshly-rotated source) - the tile
// load differs between them, everything after loading is identical.
__device__ __forceinline__ void kvarn_seal_balance_and_store(
        float * tile, int t, int bits, int sinkhorn_iters, uint8_t * record,
        size_t payload_off, size_t s_row_off, size_t zp_off, size_t s_col_off, size_t row_amp_off) {
    __shared__ float log_s_col[KVARN_N];
    __shared__ float log_s_row[KVARN_N];
    __shared__ float inv_s_col[KVARN_N];
    __shared__ float inv_s_row[KVARN_N];
    __shared__ float s_col_best[KVARN_N];
    __shared__ float s_row_best[KVARN_N];
    __shared__ float stat[KVARN_N];
    __shared__ float imbalance_best;

    log_s_col[t]  = 0.0f;
    log_s_row[t]  = 0.0f;
    inv_s_col[t]  = 1.0f;
    inv_s_row[t]  = 1.0f;
    s_col_best[t] = 1.0f;
    s_row_best[t] = 1.0f;
    __syncthreads();

    // Initial imbalance (log_s all zero, so cur == tile).
    {
        const float col_std = kvarn_std_col_cur(tile, inv_s_col, inv_s_row, t);
        stat[t] = col_std;
        __syncthreads();
        if (t == 0) {
            float col_min = stat[0], col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
            imbalance_best = col_max / fmaxf(col_min, 1e-8f);
        }
        __syncthreads();

        const float row_std = kvarn_std_row_cur(tile, inv_s_col, inv_s_row, t);
        stat[t] = row_std;
        __syncthreads();
        if (t == 0) {
            float row_min = stat[0], row_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { row_min = fminf(row_min, stat[i]); row_max = fmaxf(row_max, stat[i]); }
            imbalance_best += row_max / fmaxf(row_min, 1e-8f);
        }
        __syncthreads();
    }

    for (int iter = 0; iter < sinkhorn_iters; iter++) {
        // column pass: update log_s_col[t] using the CURRENT log_s_row
        {
            const float std = kvarn_clampf(kvarn_std_col_cur(tile, inv_s_col, inv_s_row, t), 1e-3f, 1e3f);
            log_s_col[t] = kvarn_clampf(log_s_col[t] + logf(std), -0.3f, 10.0f);
            inv_s_col[t] = expf(-log_s_col[t]);
        }
        __syncthreads();

        // row pass: update log_s_row[t] using the UPDATED log_s_col
        {
            const float std = kvarn_clampf(kvarn_std_row_cur(tile, inv_s_col, inv_s_row, t), 1e-3f, 1e3f);
            log_s_row[t] = kvarn_clampf(log_s_row[t] + logf(std), -0.3f, 10.0f);
            inv_s_row[t] = expf(-log_s_row[t]);
        }
        __syncthreads();

        // imbalance with both updated; keep best
        const float col_std2 = kvarn_std_col_cur(tile, inv_s_col, inv_s_row, t);
        stat[t] = col_std2;
        __syncthreads();
        float col_min, col_max;
        if (t == 0) {
            col_min = stat[0]; col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
        }
        __syncthreads();

        const float row_std2 = kvarn_std_row_cur(tile, inv_s_col, inv_s_row, t);
        stat[t] = row_std2;
        __syncthreads();
        if (t == 0) {
            float row_min = stat[0], row_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { row_min = fminf(row_min, stat[i]); row_max = fmaxf(row_max, stat[i]); }
            const float imbalance = col_max / fmaxf(col_min, 1e-8f) + row_max / fmaxf(row_min, 1e-8f);
            if (imbalance <= imbalance_best) {
                imbalance_best = imbalance;
                for (int i = 0; i < KVARN_N; i++) {
                    s_col_best[i] = expf(log_s_col[i]);
                    s_row_best[i] = expf(log_s_row[i]);
                }
            }
        }
        __syncthreads();
    }

    // Quantize row t of the best-balanced tile.
    {
        float balanced[KVARN_N];
        float lo = 1e30f, hi = -1e30f;
        for (int c = 0; c < KVARN_N; c++) {
            const float v = tile[t * (KVARN_N + 1) + c] / (s_col_best[c] * s_row_best[t]);
            balanced[c] = v;
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }

        const int   qmax  = (1 << bits) - 1;
        const float range = (hi - lo) / qmax;
        const float scale = fmaxf(range, 1e-10f);

        uint8_t q[KVARN_N];
        for (int c = 0; c < KVARN_N; c++) {
            const float v = roundf((balanced[c] - lo) / scale);
            q[c] = (uint8_t) kvarn_clampf(v, 0.0f, (float) qmax);
        }

        // Pack this row's `bits`-wide values into registers, then write to global mem
        uint32_t local_packed[24]; // max 6 bits * 128 / 32 = 24 words
        const int row_words_n = (bits * KVARN_N) / 32;
        for (int i = 0; i < row_words_n; i++) local_packed[i] = 0;

        const uint32_t mask = (uint32_t) ((1u << bits) - 1u);
        for (int c = 0; c < KVARN_N; c++) {
            const uint32_t value = q[c] & mask;
            const size_t bit_offset = (size_t) c * bits;

            const int word_idx = bit_offset >> 5;
            const int bit_shift = bit_offset & 31;

            local_packed[word_idx] |= (value << bit_shift);
            if (bit_shift + bits > 32) {
                local_packed[word_idx + 1] |= (value >> (32 - bit_shift));
            }
        }
        uint32_t * row_words = (uint32_t *)(record + payload_off + (size_t) t * (bits * KVARN_N / 8));
        if ((row_words_n % 4) == 0 && (((uintptr_t) row_words) % 16) == 0) {
            uint4 * row_u4 = (uint4 *) row_words;
            const uint4 * local_u4 = (const uint4 *) local_packed;
#pragma unroll
            for (int i = 0; i < row_words_n / 4; i++) {
                row_u4[i] = local_u4[i];
            }
        } else {
            for (int i = 0; i < row_words_n; i++) {
                row_words[i] = local_packed[i];
            }
        }

        const half scale_h = __float2half(scale);
        const half zp_h    = __float2half(lo);
        const half ramp_h  = __float2half(s_row_best[t]);
        ((half *)(record + s_row_off))[t]   = scale_h;
        ((half *)(record + zp_off))[t]      = zp_h;
        ((half *)(record + row_amp_off))[t] = ramp_h;

        const half scol_h = __float2half(s_col_best[t]);
        ((half *)(record + s_col_off))[t]   = scol_h;
    }
}

// One block per chunk and side. blockIdx.x selects chunk, blockIdx.y selects K (0) or V (1).
__global__ void k_kvarn_seal(
        const float * __restrict__ k_tail,
        const float * __restrict__ v_tail,
        uint8_t * __restrict__ dst,
        int key_bits, int value_bits, int sinkhorn_iters,
        struct kvarn_tile_layout layout,
        int n_complete_per_head,  // groups per head; 0 = flat (existing behavior)
        int n_groups_max,         // sealed dimension stride; ignored when flat
        int n_sealed0) {          // first sealed slot; ignored when flat
    extern __shared__ float tile[];

    const int is_v      = blockIdx.y;
    const int chunk_idx = blockIdx.x;
    const int t         = threadIdx.x;

    // Flat mode (n_complete_per_head == 0): chunk_idx indexes linearly.
    // Strided mode: chunk_idx = head * n_complete_per_head + group_in_head.
    const float * src;
    uint8_t * record;
    if (n_complete_per_head == 0) {
        src    = (is_v ? v_tail : k_tail) + (size_t) chunk_idx * (KVARN_N * KVARN_N);
        record = dst + (size_t) chunk_idx * layout.tile_bytes;
    } else {
        const int head  = chunk_idx / n_complete_per_head;
        const int group = chunk_idx % n_complete_per_head;
        src    = (is_v ? v_tail : k_tail) + (size_t) head * (KVARN_N * KVARN_N) * n_complete_per_head
                                          + (size_t) group * (KVARN_N * KVARN_N);
        record = dst + (size_t) head * (size_t) n_groups_max * layout.tile_bytes
                     + (size_t) (n_sealed0 + group) * layout.tile_bytes;
    }
    const int bits = is_v ? value_bits : key_bits;

    size_t payload_off, s_row_off, zp_off, s_col_off, row_amp_off;
    if (is_v) {
        payload_off = layout.v_payload_off; s_row_off = layout.v_s_row_off;
        zp_off      = layout.v_zp_off;      s_col_off = layout.v_s_col_off;
        row_amp_off = layout.v_row_amp_off;
    } else {
        payload_off = layout.k_payload_off; s_row_off = layout.k_s_row_off;
        zp_off      = layout.k_zp_off;      s_col_off = layout.k_s_col_off;
        row_amp_off = layout.k_row_amp_off;
    }

    // Load: thread t owns row t (coalesced within the row, stride avoids bank conflicts).
    for (int c = 0; c < KVARN_N; c++) {
        tile[t * (KVARN_N + 1) + c] = src[t * KVARN_N + c];
    }
    __syncthreads();

    kvarn_seal_balance_and_store(tile, t, bits, sinkhorn_iters, record,
            payload_off, s_row_off, zp_off, s_col_off, row_amp_off);
}

// One thread per output row - fully elementwise, no shared memory needed.
__global__ void k_kvarn_materialize(
        const uint8_t * __restrict__ sealed,
        const float   * __restrict__ tail,
        float * __restrict__ out,
        int bits, int n_sealed, int n_total,
        struct kvarn_tile_layout layout, int is_v, int max_n_sealed) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int h   = blockIdx.y;
    if (row >= n_total) return;

    const int group = row / KVARN_N;
    const int r      = row % KVARN_N;

    const size_t head_sealed_offset = (size_t) h * max_n_sealed * layout.tile_bytes;
    const size_t head_tail_offset   = (size_t) h * 128 * 128;

    const uint8_t * sealed_h = sealed + head_sealed_offset;
    const float   * tail_h   = tail + head_tail_offset;

    const int n_head_kv = gridDim.y;
    float * out_row = out + (size_t) row * (128 * n_head_kv) + (size_t) h * 128;

    if (group < n_sealed) {
        const uint8_t * record = sealed_h + (size_t) group * layout.tile_bytes;

        size_t payload_off, s_row_off, zp_off, s_col_off, row_amp_off;
        if (is_v) {
            payload_off = layout.v_payload_off; s_row_off = layout.v_s_row_off;
            zp_off      = layout.v_zp_off;      s_col_off = layout.v_s_col_off;
            row_amp_off = layout.v_row_amp_off;
        } else {
            payload_off = layout.k_payload_off; s_row_off = layout.k_s_row_off;
            zp_off      = layout.k_zp_off;      s_col_off = layout.k_s_col_off;
            row_amp_off = layout.k_row_amp_off;
        }

        half scale_h, zp_h, ramp_h;
        memcpy(&scale_h, record + s_row_off   + (size_t) r * sizeof(half), sizeof(half));
        memcpy(&zp_h,    record + zp_off      + (size_t) r * sizeof(half), sizeof(half));
        memcpy(&ramp_h,  record + row_amp_off + (size_t) r * sizeof(half), sizeof(half));
        const float scale  = __half2float(scale_h);
        const float zp     = __half2float(zp_h);
        const float s_row  = __half2float(ramp_h);

        const uint8_t * payload = record + payload_off;
        const uint32_t * row_words = (const uint32_t *)(payload + (size_t) r * (bits * KVARN_N / 8));
        const int row_words_n = (bits * KVARN_N) / 32;
        uint32_t local_packed[24];
        if ((row_words_n % 4) == 0 && (((uintptr_t) row_words) % 16) == 0) {
            uint4 * local_u4 = (uint4 *) local_packed;
            const uint4 * row_u4 = (const uint4 *) row_words;
#pragma unroll
            for (int i = 0; i < row_words_n / 4; i++) {
                local_u4[i] = row_u4[i];
            }
        } else {
            for (int i = 0; i < row_words_n; i++) {
                local_packed[i] = row_words[i];
            }
        }

        const float s_scale = s_row * scale;
        const float s_zp    = s_row * zp;
        const half * s_col_ptr = (const half *)(record + s_col_off);

        for (int c = 0; c < KVARN_N; c++) {
            const int bit_offset = c * bits;
            const int word_idx = bit_offset >> 5;
            const int shift = bit_offset & 31;
            
            uint32_t word0 = local_packed[word_idx];
            uint32_t word1 = (word_idx + 1 < row_words_n) ? local_packed[word_idx + 1] : 0;
            
            uint32_t value = __funnelshift_r(word0, word1, shift) & ((1u << bits) - 1u);
            const float other = __half2float(s_col_ptr[c]);

            out_row[c] = (s_scale * (float) value + s_zp) * other;
        }
    } else {
        // still-exact tail row
        const int tail_row = row - n_sealed * KVARN_N;
        const float * tail_src = tail_h + (size_t) tail_row * KVARN_N;
        for (int c = 0; c < KVARN_N; c++) out_row[c] = tail_src[c];
    }
}

void ggml_cuda_op_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * k_tail = dst->src[0];
    const ggml_tensor * v_tail = dst->src[1];

    GGML_ASSERT(k_tail->type == GGML_TYPE_F32 && v_tail->type == GGML_TYPE_F32);

    int32_t key_bits, value_bits, sinkhorn_iters;
    memcpy(&key_bits,       dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits,     dst->op_params + 1, sizeof(int32_t));
    memcpy(&sinkhorn_iters, dst->op_params + 2, sizeof(int32_t));

    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);

    cudaStream_t stream = ctx.stream();

    const size_t shared_bytes = (size_t) KVARN_N * (KVARN_N + 1) * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_kvarn_seal, shared_bytes);

    const int64_t n_complete = k_tail->ne[1] / 128;
    const int64_t n_head_kv  = k_tail->ne[2];
    const int64_t n_chunks   = n_complete * n_head_kv;

    dim3 grid((unsigned int) n_chunks, 2); // x = chunk_idx, y=0 -> K, y=1 -> V
    dim3 block(KVARN_N);
    k_kvarn_seal<<<grid, block, shared_bytes, stream>>>(
            (const float *) k_tail->data, (const float *) v_tail->data, (uint8_t *) dst->data,
            key_bits, value_bits, sinkhorn_iters, layout,
            0, 0, 0); // flat mode: n_complete_per_head=0
}

void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sealed = dst->src[0];
    const ggml_tensor * tail   = dst->src[1];

    GGML_ASSERT(sealed->type == GGML_TYPE_I8);
    GGML_ASSERT(tail->type == GGML_TYPE_F32);

    int32_t key_bits, value_bits, is_v, n_total, tail_count;
    memcpy(&key_bits,   dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits, dst->op_params + 1, sizeof(int32_t));
    memcpy(&is_v,       dst->op_params + 2, sizeof(int32_t));
    memcpy(&n_total,    dst->op_params + 3, sizeof(int32_t));
    memcpy(&tail_count, dst->op_params + 4, sizeof(int32_t));

    if (n_total <= 0) {
        return;
    }

    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);
    const int bits     = is_v ? value_bits : key_bits;
    const int n_sealed = (n_total - tail_count) / KVARN_N;

    cudaStream_t stream = ctx.stream();

    const int block = 128;
    int n_head_kv = tail->ne[2];
    dim3 grid((n_total + block - 1) / block, n_head_kv, 1);
    
    // sealed is [tile_bytes, max_n_sealed, n_head_kv]
    int max_n_sealed = sealed->ne[1];

    k_kvarn_materialize<<<grid, block, 0, stream>>>(
            (const uint8_t *) sealed->data, (const float *) tail->data, (float *) dst->data,
            bits, n_sealed, n_total, layout, is_v, max_n_sealed);
}

// k_kvarn_cpy_write: WHT-rotate one incoming token and write it to `dst` at
// row (dst_row_offset + tok) for head h. `dst_head_stride` is the number of
// floats between consecutive heads in `dst` (128*128 when writing straight
// into the persistent tail; n_tokens*128 when writing into a scratch buffer
// sized exactly for this call's incoming tokens). One block per (token,
// head), 128 threads, warp-shuffle WHT (see k_turbo_wht_f32_fast).
__global__ void k_kvarn_cpy_write(
        const float * __restrict__ cur,
        float       * __restrict__ dst,
        size_t dst_head_stride, int dst_row_offset) {
    const int tok = blockIdx.x;
    const int h   = blockIdx.y;
    const int c   = threadIdx.x;
    const int n_head_kv = gridDim.y;

    __shared__ float x[KVARN_N];
    x[c] = cur[((size_t) tok * n_head_kv + h) * 128 + c];
    __syncthreads();

    // Forward WHT: SIGNS1 -> butterfly -> (1/sqrt128)*SIGNS2
    // Sign bit indexing: element c is bit (c & 31) of word SIGNBITS[c >> 5].
    {
        const unsigned w = TURBO_WHT_SIGNBITS1[c >> 5];
        if ((w >> (c & 31)) & 1u) x[c] = -x[c];
    }
    __syncthreads();

    const int lane = c & 31;
    float val = x[c];
#pragma unroll
    for (int h2 = 1; h2 < 32; h2 <<= 1) {
        float o = __shfl_xor_sync(0xFFFFFFFF, val, h2);
        val = (lane & h2) ? (o - val) : (val + o);
    }
    x[c] = val;
    __syncthreads();
    if (c % 64 < 32) { float a = x[c], b = x[c+32]; x[c] = a+b; x[c+32] = a-b; }
    __syncthreads();
    if (c % 128 < 64) { float a = x[c], b = x[c+64]; x[c] = a+b; x[c+64] = a-b; }
    __syncthreads();

    constexpr float inv_sqrt = 0.08838834764831845f;
    {
        const unsigned w = TURBO_WHT_SIGNBITS2[c >> 5];
        val = x[c] * inv_sqrt;
        if ((w >> (c & 31)) & 1u) val = -val;
    }

    const int row = dst_row_offset + tok;
    dst[(size_t) h * dst_head_stride + (size_t) row * 128 + c] = val;
}

// k_kvarn_cpy_seal: seals n_complete groups for ONE side (K or V - the caller
// picks the field offsets/bits) from a virtual stream formed by the
// persistent tail (rows [0, tail_count0), already rotated) followed by
// `rotated` (this call's incoming tokens, pre-rotated by k_kvarn_cpy_write
// into a [128, n_tokens, n_head_kv] scratch buffer). Only group 0 can mix
// both sources (tail_count0 < 128 always); later groups read purely from
// `rotated`. blockIdx.x = group index, blockIdx.y = head.
__global__ void k_kvarn_cpy_seal(
        const float * __restrict__ old_tail, // [128, tail_capacity, n_head_kv]
        const float * __restrict__ rotated,  // [128, n_tokens, n_head_kv]
        uint8_t * __restrict__ sealed,
        int bits, int sinkhorn_iters, size_t tile_bytes,
        size_t payload_off, size_t s_row_off, size_t zp_off, size_t s_col_off, size_t row_amp_off,
        int tail_count0, size_t tail_head_stride, size_t rotated_head_stride, int n_groups_max, int n_sealed0) {
    extern __shared__ float tile[];

    const int g = blockIdx.x;
    const int h = blockIdx.y;
    const int t = threadIdx.x;

    const int abs_row = g * KVARN_N + t;
    const float * src_row = (abs_row < tail_count0)
            ? (old_tail + (size_t) h * tail_head_stride + (size_t) abs_row * 128)
            : (rotated  + (size_t) h * rotated_head_stride + (size_t) (abs_row - tail_count0) * 128);

    for (int c = 0; c < KVARN_N; c++) {
        tile[t * (KVARN_N + 1) + c] = src_row[c];
    }
    __syncthreads();

    uint8_t * record = sealed + (size_t) h * (size_t) n_groups_max * tile_bytes
                               + (size_t) (n_sealed0 + g) * tile_bytes;

    kvarn_seal_balance_and_store(tile, t, bits, sinkhorn_iters, record,
            payload_off, s_row_off, zp_off, s_col_off, row_amp_off);
}

__global__ void k_kvarn_cpy_tail_copy(
        const float * old_tail,
        const float * __restrict__ rotated,
        float       * tail_base,
        size_t tail_head_stride, size_t rotated_head_stride,
        int tail_count0, int src_virtual_base) {
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int c = threadIdx.x;

    const int src_virtual = src_virtual_base + r;
    const float * src_row = (src_virtual < tail_count0)
            ? (old_tail + (size_t) h * tail_head_stride + (size_t) src_virtual * 128)
            : (rotated  + (size_t) h * rotated_head_stride + (size_t) (src_virtual - tail_count0) * 128);
    const float val = src_row[c];
    __syncthreads();
    tail_base[(size_t) h * tail_head_stride + (size_t) r * 128 + c] = val;
}

void ggml_cuda_op_kvarn_cpy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur         = dst->src[0];
    const ggml_tensor * tail_base   = dst->src[1];
    const ggml_tensor * sealed_base = dst->src[2];

    int32_t is_v, key_bits, value_bits, sinkhorn_iters, tail_count0, n_sealed0, n_tokens;
    memcpy(&is_v,           dst->op_params + 0, sizeof(int32_t));
    memcpy(&key_bits,       dst->op_params + 1, sizeof(int32_t));
    memcpy(&value_bits,     dst->op_params + 2, sizeof(int32_t));
    memcpy(&sinkhorn_iters, dst->op_params + 3, sizeof(int32_t));
    memcpy(&tail_count0,    dst->op_params + 4, sizeof(int32_t));
    memcpy(&n_sealed0,      dst->op_params + 5, sizeof(int32_t));
    memcpy(&n_tokens,       dst->op_params + 6, sizeof(int32_t));

    const int n_head_kv = (int) cur->ne[1];
    const int tail_capacity = (int) tail_base->ne[1];
    const size_t tail_head_stride = (size_t) tail_capacity * 128;
    cudaStream_t stream = ctx.stream();

    const int64_t total_rows     = (int64_t) tail_count0 + n_tokens;
    const int64_t n_complete     = total_rows > tail_capacity ? (total_rows - (tail_capacity - KVARN_N)) / KVARN_N : 0;
    const int     tail_count_end = (int) (total_rows - n_complete * KVARN_N);

    if (n_complete == 0) {
        // Case A: everything fits in the existing tail capacity - append directly.
        dim3 grid((unsigned) n_tokens, (unsigned) n_head_kv);
        k_kvarn_cpy_write<<<grid, KVARN_N, 0, stream>>>(
                (const float *) cur->data, (float *) tail_base->data,
                /*dst_head_stride=*/tail_head_stride, /*dst_row_offset=*/tail_count0);
        return;
    }

    // Case B: at least one group completes - rotate into scratch, seal, restart the tail.
    const size_t rotated_head_stride = (size_t) n_tokens * 128;
    ggml_cuda_pool_alloc<float> rotated(ctx.pool(), (size_t) n_tokens * n_head_kv * 128);
    {
        dim3 grid((unsigned) n_tokens, (unsigned) n_head_kv);
        k_kvarn_cpy_write<<<grid, KVARN_N, 0, stream>>>(
                (const float *) cur->data, rotated.get(),
                rotated_head_stride, /*dst_row_offset=*/0);
    }

    {
        const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);
        const size_t shared_bytes = (size_t) KVARN_N * (KVARN_N + 1) * sizeof(float);
        CUDA_SET_SHARED_MEMORY_LIMIT(k_kvarn_cpy_seal, shared_bytes);

        const int bits = is_v ? value_bits : key_bits;
        size_t payload_off, s_row_off, zp_off, s_col_off, row_amp_off;
        if (is_v) {
            payload_off = layout.v_payload_off; s_row_off = layout.v_s_row_off;
            zp_off      = layout.v_zp_off;      s_col_off = layout.v_s_col_off;
            row_amp_off = layout.v_row_amp_off;
        } else {
            payload_off = layout.k_payload_off; s_row_off = layout.k_s_row_off;
            zp_off      = layout.k_zp_off;      s_col_off = layout.k_s_col_off;
            row_amp_off = layout.k_row_amp_off;
        }

        const int n_groups_max = (int) sealed_base->ne[1];
        dim3 grid_seal((unsigned) n_complete, (unsigned) n_head_kv);
        k_kvarn_cpy_seal<<<grid_seal, KVARN_N, shared_bytes, stream>>>(
                (const float *) tail_base->data, rotated.get(), (uint8_t *) sealed_base->data,
                bits, sinkhorn_iters, layout.tile_bytes, payload_off, s_row_off, zp_off, s_col_off, row_amp_off,
                tail_count0, tail_head_stride, rotated_head_stride, n_groups_max, n_sealed0);
    }

    if (tail_count_end > 0) {
        dim3 grid_tail((unsigned) tail_count_end, (unsigned) n_head_kv);
        k_kvarn_cpy_tail_copy<<<grid_tail, KVARN_N, 0, stream>>>(
                (const float *) tail_base->data, rotated.get(), (float *) tail_base->data,
                tail_head_stride, rotated_head_stride,
                tail_count0, /*src_virtual_base=*/(int) (n_complete * KVARN_N));
    }
}
