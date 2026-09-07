#include "kvarn.cuh"
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
__device__ __forceinline__ float kvarn_std_col_cur(const float * tile, const float * log_s_col, const float * log_s_row, int c) {
    const float inv_sc = expf(-log_s_col[c]);
    double sum = 0.0, sumsq = 0.0;
    for (int r = 0; r < KVARN_N; r++) {
        double v = tile[r * KVARN_N + c] * inv_sc * expf(-log_s_row[r]);
        sum += v; sumsq += v * v;
    }
    const double mean = sum / KVARN_N;
    const double var  = (sumsq - KVARN_N * mean * mean) / (KVARN_N - 1);
    return sqrtf(var > 0.0 ? var : 0.0);
}

__device__ __forceinline__ float kvarn_std_row_cur(const float * tile, const float * log_s_col, const float * log_s_row, int r) {
    const float inv_sr = expf(-log_s_row[r]);
    const float * row = tile + r * KVARN_N;
    double sum = 0.0, sumsq = 0.0;
    for (int c = 0; c < KVARN_N; c++) {
        double v = row[c] * inv_sr * expf(-log_s_col[c]);
        sum += v; sumsq += v * v;
    }
    const double mean = sum / KVARN_N;
    const double var  = (sumsq - KVARN_N * mean * mean) / (KVARN_N - 1);
    return sqrtf(var > 0.0 ? var : 0.0);
}

__device__ __forceinline__ float kvarn_cur_at(const float * tile, const float * log_s_col, const float * log_s_row, int r, int c) {
    return tile[r * KVARN_N + c] / (expf(log_s_col[c]) * expf(log_s_row[r]));
}

// One block per side. blockIdx.y selects K (0) or V (1). Both write into the
// same output record (disjoint byte ranges: K's payload/metadata vs V's).
__global__ void k_kvarn_seal(
        const float * __restrict__ k_tail,
        const float * __restrict__ v_tail,
        uint8_t * __restrict__ record,
        int key_bits, int value_bits, int sinkhorn_iters,
        struct kvarn_tile_layout layout) {
    extern __shared__ float tile[]; // KVARN_N * KVARN_N, dynamic (64KB)

    __shared__ float log_s_col[KVARN_N];
    __shared__ float log_s_row[KVARN_N];
    __shared__ float s_col_best[KVARN_N];
    __shared__ float s_row_best[KVARN_N];
    __shared__ float stat[KVARN_N];
    __shared__ float imbalance_best;

    const int is_v = blockIdx.y;
    const int t    = threadIdx.x; // 0..127

    const float * src  = is_v ? v_tail : k_tail;
    const int     bits = is_v ? value_bits : key_bits;

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

    // Load: thread t owns row t (coalesced within the row).
    for (int c = 0; c < KVARN_N; c++) {
        tile[t * KVARN_N + c] = src[t * KVARN_N + c];
    }
    log_s_col[t]  = 0.0f;
    log_s_row[t]  = 0.0f;
    s_col_best[t] = 1.0f;
    s_row_best[t] = 1.0f;
    __syncthreads();

    // Initial imbalance (log_s all zero, so cur == tile).
    {
        const float col_std = kvarn_std_col_cur(tile, log_s_col, log_s_row, t);
        stat[t] = col_std;
        __syncthreads();
        if (t == 0) {
            float col_min = stat[0], col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
            imbalance_best = col_max / fmaxf(col_min, 1e-8f);
        }
        __syncthreads();

        const float row_std = kvarn_std_row_cur(tile, log_s_col, log_s_row, t);
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
            const float std = kvarn_clampf(kvarn_std_col_cur(tile, log_s_col, log_s_row, t), 1e-3f, 1e3f);
            log_s_col[t] = kvarn_clampf(log_s_col[t] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();

        // row pass: update log_s_row[t] using the UPDATED log_s_col
        {
            const float std = kvarn_clampf(kvarn_std_row_cur(tile, log_s_col, log_s_row, t), 1e-3f, 1e3f);
            log_s_row[t] = kvarn_clampf(log_s_row[t] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();

        // imbalance with both updated; keep best
        const float col_std2 = kvarn_std_col_cur(tile, log_s_col, log_s_row, t);
        stat[t] = col_std2;
        __syncthreads();
        float col_min, col_max;
        if (t == 0) {
            col_min = stat[0]; col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
        }
        __syncthreads();

        const float row_std2 = kvarn_std_row_cur(tile, log_s_col, log_s_row, t);
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
            const float v = tile[t * KVARN_N + c] / (s_col_best[c] * s_row_best[t]);
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

        // Pack this row's `bits`-wide values into its own byte range
        // (128*bits is always a whole number of bytes for bits in {4,5,6},
        // so each row's packed bytes never overlap another row's - safe to
        // write independently per thread with no synchronization).
        uint8_t * row_bytes = record + payload_off + (size_t) t * (bits * KVARN_N / 8);
        const size_t row_bytes_n = (size_t) (bits * KVARN_N + 7) / 8;
        for (size_t i = 0; i < row_bytes_n; i++) row_bytes[i] = 0;
        const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
        for (int c = 0; c < KVARN_N; c++) {
            const uint8_t value = q[c] & mask;
            const size_t bit_offset = (size_t) c * bits;
            for (int b = 0; b < bits; b++) {
                const size_t dst_bit = bit_offset + b;
                row_bytes[dst_bit / 8] |= (uint8_t) (((value >> b) & 1u) << (dst_bit % 8));
            }
        }

        // scale/zp/s_row stored SEPARATELY (not pre-multiplied) - see the
        // CPU reference (ggml-kvarn-quant.c's kvarn_quantize_tile) for why:
        // s_row can reach ~22026 (Sinkhorn's log-space clamp), and real
        // model activations can push `scale` large enough that the combined
        // product overflows fp16 and silently becomes +/-inf.
        const half scale_h = __float2half(scale);
        const half zp_h    = __float2half(lo);
        const half ramp_h  = __float2half(s_row_best[t]);
        memcpy(record + s_row_off   + (size_t) t * sizeof(half), &scale_h, sizeof(half));
        memcpy(record + zp_off      + (size_t) t * sizeof(half), &zp_h,    sizeof(half));
        memcpy(record + row_amp_off + (size_t) t * sizeof(half), &ramp_h,  sizeof(half));

        const half scol_h = __float2half(s_col_best[t]);
        memcpy(record + s_col_off + (size_t) t * sizeof(half), &scol_h, sizeof(half));
    }
}

// One thread per output row - fully elementwise, no shared memory needed.
__global__ void k_kvarn_materialize(
        const uint8_t * __restrict__ sealed,
        const float   * __restrict__ tail,
        const int64_t * __restrict__ idxs,
        int64_t idxs_n,
        float * __restrict__ out,
        int64_t n_kv,
        int bits,
        struct kvarn_tile_layout layout, int is_v) {
    const int64_t row = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_kv) return;

    // Read position from idxs's actual VALUE at execution time, not from
    // op_params - see ggml_kvarn_materialize's doc comment. Every thread
    // redundantly recomputes this (cheap, avoids any shared-memory/sync
    // overhead for a single scalar).
    const int64_t pos_end = idxs[idxs_n - 1] + 1;
    const int64_t n_total = pos_end < n_kv ? pos_end : n_kv;

    float * out_row = out + (size_t) row * KVARN_N;

    if (row >= n_total) {
        for (int c = 0; c < KVARN_N; c++) out_row[c] = 0.0f;
        return;
    }

    const int64_t n_sealed = n_total / KVARN_N;
    const int64_t group    = row / KVARN_N;
    const int     r        = (int) (row % KVARN_N);

    if (group < n_sealed) {
        const uint8_t * record = sealed + (size_t) group * layout.tile_bytes;

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

        for (int c = 0; c < KVARN_N; c++) {
            const size_t bit_offset = (size_t) (r * KVARN_N + c) * bits;
            uint8_t value = 0;
            for (int b = 0; b < bits; b++) {
                const size_t src_bit = bit_offset + b;
                value = (uint8_t) (value | (((payload[src_bit / 8] >> (src_bit % 8)) & 1u) << b));
            }

            half scol_h;
            memcpy(&scol_h, record + s_col_off + (size_t) c * sizeof(half), sizeof(half));
            const float other = __half2float(scol_h);

            out_row[c] = s_row * ((float) value * scale + zp) * other;
        }
    } else {
        // still-exact tail row
        const int tail_row = row - n_sealed * KVARN_N;
        const float * tail_src = tail + (size_t) tail_row * KVARN_N;
        for (int c = 0; c < KVARN_N; c++) out_row[c] = tail_src[c];
    }
}

// k_kvarn_store_seal / k_kvarn_store_tail
//
// Fixed-topology write (see ggml_kvarn_store's doc comment, ggml.h): unlike
// k_kvarn_seal above (which always processes exactly the two tensors it's
// handed), these kernels derive the starting position from `idxs[0]`'s
// actual VALUE at execution time - the same tensor llama.cpp's graph-reuse
// optimization already re-patches every call, unlike op_params (baked once
// at graph-build time). One (layer, side) call now launches these two
// kernels, sequentially on one stream (CUDA serializes same-stream launches
// automatically - no explicit sync needed), covering however many groups
// this call completes (0 or more) instead of building a variable number of
// GGML_OP_KVARN_SEAL nodes host-side.
//
// Split into two kernels rather than one, specifically to avoid a
// read-after-write hazard: sealing group 0 (if this call's tail_count0 > 0)
// needs to read the OLD tail rows before this call's leftover write
// overwrites them, and different CUDA blocks within one kernel launch have
// no ordering guarantee relative to each other - only relative to a
// separate, later kernel launch on the same stream.
//
// k_kvarn_store_tail only WRITES rows that come from `cur` (index >=
// tail_count0) - rows that come from the old tail (index < tail_count0)
// already hold the correct value in place and need no write at all, which
// sidesteps the hazard entirely rather than needing a same-kernel
// read-into-register-then-write-back dance.

// One block per (head, candidate group slot). blockIdx.y >= the actual
// number of groups this call completes is a normal, expected no-op (the
// grid is sized from max_groups_per_call, a host-computed UPPER bound).
__global__ void k_kvarn_store_seal(
        const float   * __restrict__ cur,    // [128, n_head_kv, n_tokens], rotated, contiguous
        const int64_t * __restrict__ idxs,   // [n_tokens]
        const float   * __restrict__ tail,   // [128, 128, n_head_kv], OLD data (read-only here)
        uint8_t       * __restrict__ sealed, // [tile_bytes, n_groups_max, n_head_kv]
        int64_t n_tokens, int bits, int sinkhorn_iters, int is_v,
        int64_t cur_head_floats, int64_t cur_token_floats, size_t sealed_head_stride,
        struct kvarn_tile_layout layout) {
    const int64_t h = blockIdx.x;
    const int64_t g = blockIdx.y;
    const int     t = threadIdx.x; // 0..127

    const int64_t pos0        = idxs[0];
    const int64_t tail_count0 = pos0 % KVARN_N;
    const int64_t n_sealed0   = pos0 / KVARN_N;
    const int64_t total_rows  = tail_count0 + n_tokens;
    const int64_t n_complete  = total_rows / KVARN_N;

    if (g >= n_complete) {
        return;
    }

    extern __shared__ float tile[]; // KVARN_N * KVARN_N, dynamic (64KB)
    __shared__ float log_s_col[KVARN_N];
    __shared__ float log_s_row[KVARN_N];
    __shared__ float s_col_best[KVARN_N];
    __shared__ float s_row_best[KVARN_N];
    __shared__ float stat[KVARN_N];
    __shared__ float imbalance_best;

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

    const float * tail_h = tail + (size_t) h * KVARN_N * KVARN_N;
    const float * cur_h  = cur + (size_t) h * cur_head_floats;

    // Load: thread t owns row t of THIS group's virtual 128-row tile,
    // sourced from the still-live old tail (only possible for g==0, since
    // g>=1's absolute row range starts at g*128 >= 128 > tail_count0) or
    // from this call's new tokens.
    const int64_t abs_row = g * KVARN_N + t;
    const float * src_row;
    if (abs_row < tail_count0) {
        src_row = tail_h + (size_t) abs_row * KVARN_N;
    } else {
        src_row = cur_h + (size_t) (abs_row - tail_count0) * cur_token_floats;
    }
    for (int c = 0; c < KVARN_N; c++) {
        tile[t * KVARN_N + c] = src_row[c];
    }
    log_s_col[t]  = 0.0f;
    log_s_row[t]  = 0.0f;
    s_col_best[t] = 1.0f;
    s_row_best[t] = 1.0f;
    __syncthreads();

    {
        const float col_std = kvarn_std_col_cur(tile, log_s_col, log_s_row, t);
        stat[t] = col_std;
        __syncthreads();
        if (t == 0) {
            float col_min = stat[0], col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
            imbalance_best = col_max / fmaxf(col_min, 1e-8f);
        }
        __syncthreads();

        const float row_std = kvarn_std_row_cur(tile, log_s_col, log_s_row, t);
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
        {
            const float std = kvarn_clampf(kvarn_std_col_cur(tile, log_s_col, log_s_row, t), 1e-3f, 1e3f);
            log_s_col[t] = kvarn_clampf(log_s_col[t] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();

        {
            const float std = kvarn_clampf(kvarn_std_row_cur(tile, log_s_col, log_s_row, t), 1e-3f, 1e3f);
            log_s_row[t] = kvarn_clampf(log_s_row[t] + logf(std), -0.3f, 10.0f);
        }
        __syncthreads();

        const float col_std2 = kvarn_std_col_cur(tile, log_s_col, log_s_row, t);
        stat[t] = col_std2;
        __syncthreads();
        float col_min, col_max;
        if (t == 0) {
            col_min = stat[0]; col_max = stat[0];
            for (int i = 1; i < KVARN_N; i++) { col_min = fminf(col_min, stat[i]); col_max = fmaxf(col_max, stat[i]); }
        }
        __syncthreads();

        const float row_std2 = kvarn_std_row_cur(tile, log_s_col, log_s_row, t);
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

    uint8_t * record = sealed + h * sealed_head_stride + (size_t) (n_sealed0 + g) * layout.tile_bytes;

    {
        float balanced[KVARN_N];
        float lo = 1e30f, hi = -1e30f;
        for (int c = 0; c < KVARN_N; c++) {
            const float v = tile[t * KVARN_N + c] / (s_col_best[c] * s_row_best[t]);
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

        uint8_t * row_bytes = record + payload_off + (size_t) t * (bits * KVARN_N / 8);
        const size_t row_bytes_n = (size_t) (bits * KVARN_N + 7) / 8;
        for (size_t i = 0; i < row_bytes_n; i++) row_bytes[i] = 0;
        const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
        for (int c = 0; c < KVARN_N; c++) {
            const uint8_t value = q[c] & mask;
            const size_t bit_offset = (size_t) c * bits;
            for (int b = 0; b < bits; b++) {
                const size_t dst_bit = bit_offset + b;
                row_bytes[dst_bit / 8] |= (uint8_t) (((value >> b) & 1u) << (dst_bit % 8));
            }
        }

        const half scale_h = __float2half(scale);
        const half zp_h    = __float2half(lo);
        const half ramp_h  = __float2half(s_row_best[t]);
        memcpy(record + s_row_off   + (size_t) t * sizeof(half), &scale_h, sizeof(half));
        memcpy(record + zp_off      + (size_t) t * sizeof(half), &zp_h,    sizeof(half));
        memcpy(record + row_amp_off + (size_t) t * sizeof(half), &ramp_h,  sizeof(half));

        const half scol_h = __float2half(s_col_best[t]);
        memcpy(record + s_col_off + (size_t) t * sizeof(half), &scol_h, sizeof(half));
    }
}

// One block per head, one thread per tail row. Only writes rows sourced
// from `cur` (>= tail_count0) - rows still sourced from the old tail
// already hold the right value in place and are left untouched, avoiding
// any read-after-write ordering concern within this kernel.
__global__ void k_kvarn_store_tail(
        const float   * __restrict__ cur,
        const int64_t * __restrict__ idxs,
        float         * __restrict__ tail,
        int64_t n_tokens, int64_t cur_head_floats, int64_t cur_token_floats) {
    const int64_t h = blockIdx.x;
    const int64_t r = threadIdx.x;

    const int64_t pos0           = idxs[0];
    const int64_t tail_count0    = pos0 % KVARN_N;
    const int64_t total_rows     = tail_count0 + n_tokens;
    const int64_t n_complete     = total_rows / KVARN_N;
    const int64_t tail_count_end = total_rows - n_complete * KVARN_N;

    if (r >= tail_count_end || r < tail_count0) {
        return;
    }

    const int64_t tok = n_complete * KVARN_N + r - tail_count0;
    const float * src = cur + (size_t) h * cur_head_floats + (size_t) tok * cur_token_floats;
    float       * dst = tail + (size_t) h * KVARN_N * KVARN_N + (size_t) r * KVARN_N;
    for (int c = 0; c < KVARN_N; c++) {
        dst[c] = src[c];
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

    const size_t shared_bytes = (size_t) KVARN_N * KVARN_N * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_kvarn_seal, shared_bytes);

    dim3 grid(1, 2); // y=0 -> K, y=1 -> V
    dim3 block(KVARN_N);
    k_kvarn_seal<<<grid, block, shared_bytes, stream>>>(
            (const float *) k_tail->data, (const float *) v_tail->data, (uint8_t *) dst->data,
            key_bits, value_bits, sinkhorn_iters, layout);
}

void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * sealed = dst->src[0];
    const ggml_tensor * tail   = dst->src[1];
    const ggml_tensor * idxs   = dst->src[2];

    GGML_ASSERT(sealed->type == GGML_TYPE_I8);
    GGML_ASSERT(tail->type == GGML_TYPE_F32);
    GGML_ASSERT(idxs->type == GGML_TYPE_I64);

    int32_t key_bits, value_bits, is_v;
    memcpy(&key_bits,   dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits, dst->op_params + 1, sizeof(int32_t));
    memcpy(&is_v,       dst->op_params + 2, sizeof(int32_t));

    const int64_t n_kv = dst->ne[1];
    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);
    const int bits = is_v ? value_bits : key_bits;

    cudaStream_t stream = ctx.stream();

    const int64_t block = 128;
    const int64_t grid  = (n_kv + block - 1) / block;
    k_kvarn_materialize<<<(unsigned) grid, (unsigned) block, 0, stream>>>(
            (const uint8_t *) sealed->data, (const float *) tail->data,
            (const int64_t *) idxs->data, idxs->ne[0],
            (float *) dst->data, n_kv, bits, layout, is_v);
}

void ggml_cuda_op_kvarn_store(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * cur    = dst->src[0];
    const ggml_tensor * idxs   = dst->src[1];
    const ggml_tensor * tail   = dst->src[2];
    const ggml_tensor * sealed = dst->src[3];

    GGML_ASSERT(cur->type == GGML_TYPE_F32 && idxs->type == GGML_TYPE_I64);
    GGML_ASSERT(tail->type == GGML_TYPE_F32 && sealed->type == GGML_TYPE_I8);

    int32_t key_bits, value_bits, sinkhorn_iters, is_v, max_groups_per_call;
    memcpy(&key_bits,            dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits,          dst->op_params + 1, sizeof(int32_t));
    memcpy(&sinkhorn_iters,      dst->op_params + 2, sizeof(int32_t));
    memcpy(&is_v,                dst->op_params + 3, sizeof(int32_t));
    memcpy(&max_groups_per_call, dst->op_params + 4, sizeof(int32_t));

    const int64_t n_head_kv = cur->ne[1];
    const int64_t n_tokens  = cur->ne[2];
    const int     bits      = is_v ? value_bits : key_bits;

    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);

    const int64_t cur_head_floats    = cur->nb[1] / sizeof(float);
    const int64_t cur_token_floats   = cur->nb[2] / sizeof(float);
    const size_t  sealed_head_stride = (size_t) sealed->nb[2];

    cudaStream_t stream = ctx.stream();

    const size_t shared_bytes = (size_t) KVARN_N * KVARN_N * sizeof(float);
    CUDA_SET_SHARED_MEMORY_LIMIT(k_kvarn_store_seal, shared_bytes);

    dim3 grid_seal((unsigned) n_head_kv, (unsigned) max_groups_per_call);
    k_kvarn_store_seal<<<grid_seal, KVARN_N, shared_bytes, stream>>>(
            (const float *) cur->data, (const int64_t *) idxs->data,
            (const float *) tail->data, (uint8_t *) sealed->data,
            n_tokens, bits, sinkhorn_iters, is_v,
            cur_head_floats, cur_token_floats, sealed_head_stride, layout);

    k_kvarn_store_tail<<<(unsigned) n_head_kv, KVARN_N, 0, stream>>>(
            (const float *) cur->data, (const int64_t *) idxs->data, (float *) dst->data,
            n_tokens, cur_head_floats, cur_token_floats);
}
