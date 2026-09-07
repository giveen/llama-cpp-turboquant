#include "kvarn-attn-decode.cuh"
#include "ggml-kvarn-quant.h"

#include <cuda_fp16.h>

// KVarN fused decode attention, split-K ("flash-decoding") version.
//
// v1 of this kernel used one block per Q head processing every sealed group
// sequentially - correct, but at long context that is one block doing
// hundreds of groups' worth of serial work while the rest of the GPU sits
// idle (measured: 3.7-4.3x slower than the plain materialize-then-dense-
// attend path even after removing an atomics-contention bug, see the
// project's plan notes). The fix ported here is the same one the beellama
// KVarN fork/upstream used to get a real win at depth: split the KV history
// across many independent blocks (a handful of sealed groups each) so the
// GPU actually has enough concurrent work to fill its SMs, then merge the
// per-split partial (running max, running sum, weighted-V) results with a
// second, tiny combine kernel using the standard online-softmax merge
// identity. This does NOT need beellama's tensor-core MMA kernel to be a
// real win - splitting alone turns "1 block always" into "one block per
// GROUPS_PER_SPLIT sealed groups", which is what actually starves the GPU
// at long context.
//
// Q must already be rotated (ggml_turbo_wht, direction=0) by the caller,
// same as turbo4's existing graph-level convention - K/V dequantize to the
// SAME rotated domain they were stored in (see cpy_kvarn), so QK^T is
// unaffected (rotation is orthogonal: (Rq).(Rk) = q.k) and the output only
// needs one inverse rotation at the end (also left to the caller, via
// ggml_turbo_wht direction=1) - not per cached position, because rotation
// commutes with the linear weighted-sum in attention: sum_i w_i*(R v_i) =
// R * (sum_i w_i*v_i).

#define KVARN_N 128

// How many sealed 128-token groups one partial-kernel block processes
// sequentially before handing off to the combine kernel. Small enough that
// long contexts still produce many concurrent blocks (good occupancy),
// large enough that per-block/per-split fixed overhead (combine kernel
// work, launch overhead) stays a small fraction of total work.
#define KVARN_GROUPS_PER_SPLIT 4

// One block's contribution: online-softmax state for one (q_head, split) pair.
__global__ void k_kvarn_attn_decode_partial(
        const float   * __restrict__ q,        // [128, n_head_q] rotated
        const uint8_t * __restrict__ sealed,    // [tile_bytes, n_groups_max, n_head_kv]
        const float   * __restrict__ k_tail,    // [128, 128, n_head_kv]
        const float   * __restrict__ v_tail,    // [128, 128, n_head_kv]
        const int64_t * __restrict__ idxs,      // [n_tokens] - element [idxs_n-1]+1 gives real content length
        int64_t idxs_n,
        float * __restrict__ partial_out,       // [n_splits, n_head_q, 128]
        float * __restrict__ partial_max,       // [n_splits, n_head_q]
        float * __restrict__ partial_sum,       // [n_splits, n_head_q]
        int key_bits, int value_bits,
        int n_head_kv, int n_group_broadcast, // n_head_q = n_head_kv * n_group_broadcast
        int n_groups_max,
        int n_splits_sealed_max, // FIXED upper bound (ceil(n_groups_max/GROUPS_PER_SPLIT)), not the real
                                  // n_sealed - blockIdx.y in [0, n_splits_sealed_max) => sealed groups (a
                                  // block whose g_begin is past the REAL n_sealed, computed on-device
                                  // below, naturally contributes a harmless zero partial - see the
                                  // g_end computation); == n_splits_sealed_max => dedicated tail split
        float kq_scale,
        struct kvarn_tile_layout layout) {
    const int q_head  = blockIdx.x;
    const int kv_head = q_head / n_group_broadcast;
    const int split_id = blockIdx.y;
    const int r        = threadIdx.x; // 0..127

    // Read position from idxs's actual VALUE at execution time, not from
    // op_params/host-computed args - see ggml_kvarn_store's doc comment for
    // why this matters under graph reuse. Every block redundantly
    // recomputes this (cheap, avoids a separate prologue kernel/shared
    // state), matching k_kvarn_store_seal/_tail's existing pattern.
    const int64_t n_total    = idxs[idxs_n - 1] + 1;
    const int64_t tail_count = n_total % KVARN_N;
    const int64_t n_sealed   = (n_total - tail_count) / KVARN_N;

    __shared__ float q_sh[KVARN_N];
    __shared__ float s_col_k[KVARN_N];
    __shared__ float s_col_v[KVARN_N];
    __shared__ float row_score[KVARN_N];
    __shared__ float out_acc[KVARN_N];
    __shared__ float running_max;
    __shared__ float running_sum;

    q_sh[r]    = q[(size_t) q_head * KVARN_N + r];
    out_acc[r] = 0.0f;
    if (r == 0) { running_max = -1e30f; running_sum = 0.0f; }
    __syncthreads();

    const size_t tile_bytes = layout.tile_bytes;
    const uint8_t * sealed_kv = sealed + (size_t) kv_head * (size_t) n_groups_max * tile_bytes;

    if (split_id < n_splits_sealed_max) {
        const int64_t g_begin = (int64_t) split_id * KVARN_GROUPS_PER_SPLIT;
        const int64_t g_end   = min(n_sealed, g_begin + KVARN_GROUPS_PER_SPLIT);

        for (int64_t g = g_begin; g < g_end; g++) {
            const uint8_t * record = sealed_kv + (size_t) g * tile_bytes;

            half h;
            memcpy(&h, record + layout.k_s_col_off + (size_t) r * sizeof(half), sizeof(half));
            s_col_k[r] = __half2float(h);
            memcpy(&h, record + layout.v_s_col_off + (size_t) r * sizeof(half), sizeof(half));
            s_col_v[r] = __half2float(h);
            __syncthreads();

            // This row's K dot product with Q (sequential over 128 channels).
            // scale/zp/row-amplitude are stored SEPARATELY, not pre-multiplied
            // - see ggml-kvarn-quant.c's kvarn_quantize_tile for why (fp16
            // overflow for real, less-well-conditioned model activations).
            half scale_h, zp_h, ramp_h;
            memcpy(&scale_h, record + layout.k_s_row_off   + (size_t) r * sizeof(half), sizeof(half));
            memcpy(&zp_h,    record + layout.k_zp_off      + (size_t) r * sizeof(half), sizeof(half));
            memcpy(&ramp_h,  record + layout.k_row_amp_off + (size_t) r * sizeof(half), sizeof(half));
            const float k_scale = __half2float(scale_h);
            const float k_zp    = __half2float(zp_h);
            const float k_ramp  = __half2float(ramp_h);

            const uint8_t * k_payload = record + layout.k_payload_off;
            float dot = 0.0f;
#pragma unroll 4
            for (int c = 0; c < KVARN_N; c++) {
                const size_t bit_offset = (size_t) (r * KVARN_N + c) * key_bits;
                uint8_t value = 0;
                for (int b = 0; b < key_bits; b++) {
                    const size_t bit = bit_offset + b;
                    value = (uint8_t) (value | (((k_payload[bit / 8] >> (bit % 8)) & 1u) << b));
                }
                const float k_val = k_ramp * ((float) value * k_scale + k_zp) * s_col_k[c];
                dot += q_sh[c] * k_val;
            }
            row_score[r] = dot * kq_scale;
            __syncthreads();

            // Block-wide max over this group's 128 scores.
            __shared__ float group_max;
            if (r == 0) {
                float m = row_score[0];
                for (int i = 1; i < KVARN_N; i++) m = fmaxf(m, row_score[i]);
                group_max = m;
            }
            __syncthreads();

            const float new_max = fmaxf(running_max, group_max);
            const float rescale = expf(running_max - new_max);
            out_acc[r] *= rescale; // each thread rescales its own output channel
            __shared__ float w_row[KVARN_N];
            w_row[r] = expf(row_score[r] - new_max);
            __syncthreads();
            if (r == 0) {
                float sum_local = 0.0f;
                for (int i = 0; i < KVARN_N; i++) sum_local += expf(row_score[i] - new_max);
                running_sum = running_sum * rescale + sum_local;
                running_max = new_max;
            }
            __syncthreads();

            // V accumulation without atomics: thread r owns OUTPUT COLUMN r
            // (reusing the same thread index) and sums every source row's
            // weighted, dequantized contribution to that one column itself.
            // For a fixed loop iteration rr, adjacent threads r/r+1 read
            // adjacent bit offsets ((rr*128+r)*value_bits), so this stays
            // reasonably coalesced despite the per-thread bit-level unpacking.
            const uint8_t * v_payload = record + layout.v_payload_off;
            float acc_local = 0.0f;
#pragma unroll 4
            for (int rr = 0; rr < KVARN_N; rr++) {
                half vscale_h, vzp_h, vramp_h;
                memcpy(&vscale_h, record + layout.v_s_row_off   + (size_t) rr * sizeof(half), sizeof(half));
                memcpy(&vzp_h,    record + layout.v_zp_off      + (size_t) rr * sizeof(half), sizeof(half));
                memcpy(&vramp_h,  record + layout.v_row_amp_off + (size_t) rr * sizeof(half), sizeof(half));
                const float v_scale = __half2float(vscale_h);
                const float v_zp    = __half2float(vzp_h);
                const float v_ramp  = __half2float(vramp_h);
                const size_t bit_offset = (size_t) (rr * KVARN_N + r) * value_bits;
                uint8_t value = 0;
                for (int b = 0; b < value_bits; b++) {
                    const size_t bit = bit_offset + b;
                    value = (uint8_t) (value | (((v_payload[bit / 8] >> (bit % 8)) & 1u) << b));
                }
                const float v_val = v_ramp * ((float) value * v_scale + v_zp) * s_col_v[r];
                acc_local += w_row[rr] * v_val;
            }
            out_acc[r] += acc_local;
            __syncthreads();
        }
    } else {
        // Dedicated tail split: exact rows, no dequant needed. n_total/
        // n_sealed already computed on-device at function scope above.
        const int64_t from_tail = n_total - n_sealed * KVARN_N;
        const float * k_tail_h = k_tail + (size_t) kv_head * KVARN_N * KVARN_N;
        const float * v_tail_h = v_tail + (size_t) kv_head * KVARN_N * KVARN_N;

        if (from_tail > 0) {
            float dot = 0.0f;
            const bool valid = r < from_tail;
            if (valid) {
#pragma unroll 4
                for (int c = 0; c < KVARN_N; c++) {
                    dot += q_sh[c] * k_tail_h[r * KVARN_N + c];
                }
            }
            row_score[r] = valid ? dot * kq_scale : -1e30f;
            __syncthreads();

            __shared__ float tail_max;
            if (r == 0) {
                float m = row_score[0];
                for (int i = 1; i < KVARN_N; i++) m = fmaxf(m, row_score[i]);
                tail_max = m;
            }
            __syncthreads();

            const float new_max = fmaxf(running_max, tail_max);
            const float rescale = expf(running_max - new_max);
            out_acc[r] *= rescale;
            __shared__ float w_row_tail[KVARN_N];
            w_row_tail[r] = valid ? expf(row_score[r] - new_max) : 0.0f;
            __syncthreads();
            if (r == 0) {
                float sum_local = 0.0f;
                for (int i = 0; i < from_tail; i++) sum_local += expf(row_score[i] - new_max);
                running_sum = running_sum * rescale + sum_local;
                running_max = new_max;
            }
            __syncthreads();

            float acc_local = 0.0f;
#pragma unroll 4
            for (int rr = 0; rr < KVARN_N; rr++) {
                acc_local += w_row_tail[rr] * v_tail_h[rr * KVARN_N + r];
            }
            out_acc[r] += acc_local;
            __syncthreads();
        }
    }

    const size_t out_idx = ((size_t) split_id * gridDim.x + q_head) * KVARN_N + r;
    partial_out[out_idx] = out_acc[r];
    if (r == 0) {
        const size_t ms_idx = (size_t) split_id * gridDim.x + q_head;
        partial_max[ms_idx] = running_max;
        partial_sum[ms_idx] = running_sum;
    }
}

// Merges every split's partial online-softmax state for one Q head into the
// final attention output, via the standard rescale-by-relative-max identity.
__global__ void k_kvarn_attn_decode_combine(
        const float * __restrict__ partial_out, // [n_splits, n_head_q, 128]
        const float * __restrict__ partial_max, // [n_splits, n_head_q]
        const float * __restrict__ partial_sum, // [n_splits, n_head_q]
        float * __restrict__ out,               // [128, n_head_q]
        int n_splits, int n_head_q) {
    const int q_head = blockIdx.x;
    const int c       = threadIdx.x; // 0..127

    __shared__ float global_max;
    __shared__ float denom;

    if (c == 0) {
        float m = -1e30f;
        for (int s = 0; s < n_splits; s++) {
            m = fmaxf(m, partial_max[(size_t) s * n_head_q + q_head]);
        }
        global_max = m;
    }
    __syncthreads();

    float acc = 0.0f;
#pragma unroll 4
    for (int s = 0; s < n_splits; s++) {
        const float pm = partial_max[(size_t) s * n_head_q + q_head];
        const float w  = expf(pm - global_max);
        acc += w * partial_out[((size_t) s * n_head_q + q_head) * KVARN_N + c];
    }

    if (c == 0) {
        float d = 0.0f;
        for (int s = 0; s < n_splits; s++) {
            const float pm = partial_max[(size_t) s * n_head_q + q_head];
            d += expf(pm - global_max) * partial_sum[(size_t) s * n_head_q + q_head];
        }
        denom = fmaxf(d, 1e-20f);
    }
    __syncthreads();

    out[(size_t) q_head * KVARN_N + c] = acc / denom;
}

void ggml_cuda_op_kvarn_attn_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * sealed = dst->src[1];
    const ggml_tensor * k_tail = dst->src[2];
    const ggml_tensor * v_tail = dst->src[3];
    const ggml_tensor * idxs   = dst->src[4];

    int32_t key_bits, value_bits, n_head_kv, n_group_broadcast;
    float   kq_scale;
    memcpy(&key_bits,          dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits,        dst->op_params + 1, sizeof(int32_t));
    memcpy(&n_head_kv,         dst->op_params + 2, sizeof(int32_t));
    memcpy(&n_group_broadcast, dst->op_params + 3, sizeof(int32_t));
    memcpy(&kq_scale,          dst->op_params + 4, sizeof(float));

    const int n_head_q = n_head_kv * n_group_broadcast;
    const int n_groups_max = (int) sealed->ne[1]; // sealed is [tile_bytes, n_groups_max, n_head_kv]
    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);

    // FIXED upper bound on sealed splits, derived from the layer's
    // allocated group CAPACITY (a per-layer construction-time constant,
    // never varies per call) rather than the real n_sealed - reading the
    // real value would need a host sync on `idxs` (CUDA-resident), which
    // this hot per-layer-per-token dispatch path deliberately avoids (see
    // the graph-reuse plan doc). Blocks whose range is past the real
    // n_sealed (computed on-device inside the kernel) naturally contribute
    // a harmless zero partial - see k_kvarn_attn_decode_partial.
    const int n_splits_sealed_max = (n_groups_max + KVARN_GROUPS_PER_SPLIT - 1) / KVARN_GROUPS_PER_SPLIT;
    const int n_splits_total      = n_splits_sealed_max + 1; // +1 dedicated tail split, always launched

    cudaStream_t stream = ctx.stream();

    ggml_cuda_pool_alloc<float> partial_out(ctx.pool(), (size_t) n_splits_total * n_head_q * KVARN_N);
    ggml_cuda_pool_alloc<float> partial_max(ctx.pool(), (size_t) n_splits_total * n_head_q);
    ggml_cuda_pool_alloc<float> partial_sum(ctx.pool(), (size_t) n_splits_total * n_head_q);

    dim3 grid_partial(n_head_q, n_splits_total);
    dim3 block_partial(KVARN_N);
    k_kvarn_attn_decode_partial<<<grid_partial, block_partial, 0, stream>>>(
            (const float *) q->data, (const uint8_t *) sealed->data,
            (const float *) k_tail->data, (const float *) v_tail->data,
            (const int64_t *) idxs->data, idxs->ne[0],
            partial_out.get(), partial_max.get(), partial_sum.get(),
            key_bits, value_bits, n_head_kv, n_group_broadcast,
            n_groups_max, n_splits_sealed_max, kq_scale, layout);

    dim3 grid_combine(n_head_q);
    dim3 block_combine(KVARN_N);
    k_kvarn_attn_decode_combine<<<grid_combine, block_combine, 0, stream>>>(
            partial_out.get(), partial_max.get(), partial_sum.get(),
            (float *) dst->data, n_splits_total, n_head_q);
}
