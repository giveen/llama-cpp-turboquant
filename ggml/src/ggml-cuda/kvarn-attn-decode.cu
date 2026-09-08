#include "kvarn-attn-decode.cuh"
#include "ggml-kvarn-quant.h"

#include <cuda_fp16.h>

// KVarN fused decode attention, split-K ("flash-decoding") version.
//
// v1 of this kernel used one block per Q head processing every sealed group
// sequentially - correct, but at long context that left too little parallel
// work. The current kernel uses one block per KV head and computes the whole
// compile-time GQA group together, so each compressed tile is staged once and
// reused by all query heads that share it. Split-K still divides the sealed
// history across blocks and the combine kernel merges their partial results.
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
#define KVARN_GROUPS_PER_SPLIT 1

template<int Bits>
__device__ __forceinline__ uint32_t kvarn_unpack_row_value(
        const uint8_t * row, int index) {
    static_assert(Bits >= 2 && Bits <= 6, "unsupported KVarN bit width");

    const int bit_offset = index * Bits;
    const int aligned_byte_offset = (bit_offset >> 3) & ~3;
    const int shift = bit_offset - (aligned_byte_offset * 8);
    
    const uint32_t * p32 = (const uint32_t *)(row + aligned_byte_offset);
    uint32_t word0 = p32[0];
    uint32_t word1 = p32[1];
    
    return __funnelshift_r(word0, word1, shift) & ((1u << Bits) - 1u);
}
template<int Bits>
__device__ __forceinline__ uint32_t kvarn_unpack_warp_v(const uint32_t * warp_row_words, int lane) {
    constexpr int WORDS_PER_WARP = Bits;
    uint32_t my_word = (lane < WORDS_PER_WARP) ? warp_row_words[lane] : 0;

    const int bit_offset = lane * Bits;
    const int w0_idx     = bit_offset >> 5;
    const int shift      = bit_offset & 31;

    uint32_t word0 = __shfl_sync(0xFFFFFFFF, my_word, w0_idx);
    if constexpr (Bits == 2 || Bits == 4) {
        return (word0 >> shift) & ((1u << Bits) - 1u);
    } else {
        uint32_t word1 = __shfl_sync(0xFFFFFFFF, my_word, min(w0_idx + 1, 31));
        return __funnelshift_r(word0, word1, shift) & ((1u << Bits) - 1u);
    }
}


// One block's contribution: online-softmax state for one KV head and its GQA group.
template<int KeyBits, int ValueBits, int Gqa>
__launch_bounds__(Gqa * KVARN_N, (Gqa <= 4 ? 2 : 1))
__global__ void k_kvarn_attn_decode_partial(
        const float   * __restrict__ q,        // [128, n_head_q] rotated
        const uint8_t * __restrict__ sealed,    // [tile_bytes, n_groups_max, n_head_kv]
        const float   * __restrict__ k_tail,    // [128, 128, n_head_kv]
        const float   * __restrict__ v_tail,    // [128, 128, n_head_kv]
        float * __restrict__ partial_out,       // [n_splits, n_head_q, 128]
        float * __restrict__ partial_max,       // [n_splits, n_head_q]
        float * __restrict__ partial_sum,       // [n_splits, n_head_q]
        int n_groups_max, int n_total, int tail_count,
        int n_splits_sealed, int tail_capacity,
        float kq_scale,
        struct kvarn_tile_layout layout) {
    const int split_id = blockIdx.y;
    const int q_slot  = threadIdx.x / KVARN_N;
    const int r        = threadIdx.x % KVARN_N; // 0..127
    const int warp_in_block = threadIdx.x / 32;
    const int lane          = threadIdx.x % 32;
    const int q_head   = blockIdx.x * Gqa + q_slot;
    const int kv_head  = blockIdx.x;

    __shared__ float q_sh[Gqa * KVARN_N];
    __shared__ float s_col_k[KVARN_N];
    __shared__ float s_col_v[KVARN_N];
    __shared__ float k_scale_sh[KVARN_N];
    __shared__ float k_zp_sh[KVARN_N];
    __shared__ float k_ramp_sh[KVARN_N];
    __shared__ float v_scale_sh[KVARN_N];
    __shared__ float v_zp_sh[KVARN_N];
    __shared__ float v_ramp_sh[KVARN_N];
    constexpr int k_row_words = (KeyBits * KVARN_N / 8) / 4;
    constexpr int k_stride_words = k_row_words + 1; // +4 bytes padding to avoid bank conflicts
    constexpr int v_row_words = (ValueBits * KVARN_N / 8) / 4;
    constexpr int v_stride_words = v_row_words + 1;
    __shared__ uint32_t k_payload_sh_u32[k_stride_words * KVARN_N];
    __shared__ uint32_t v_payload_sh_u32[v_stride_words * KVARN_N];
    __shared__ float row_score[Gqa * KVARN_N];
    __shared__ float out_acc[Gqa * KVARN_N];
    __shared__ float w_row[Gqa * KVARN_N];
    __shared__ float running_max[Gqa];
    __shared__ float running_sum[Gqa];
    __shared__ float group_max[Gqa];
    __shared__ float warp_max[Gqa * 4]; // 4 warps per q_slot
    __shared__ float warp_sum[Gqa * 4];

    q_sh[q_slot * KVARN_N + r] = q[(size_t) q_head * KVARN_N + r];
    out_acc[q_slot * KVARN_N + r] = 0.0f;
    if (r == 0) {
        running_max[q_slot] = -1e30f;
        running_sum[q_slot] = 0.0f;
    }
    __syncthreads();

    const size_t tile_bytes = layout.tile_bytes;
    const uint8_t * sealed_kv = sealed + (size_t) kv_head * (size_t) n_groups_max * tile_bytes;

    if (split_id < n_splits_sealed) {
        const int n_sealed = (n_total - tail_count) / KVARN_N;
        const int g_begin = split_id * KVARN_GROUPS_PER_SPLIT;
        const int g_end   = min(n_sealed, g_begin + KVARN_GROUPS_PER_SPLIT);

        for (int g = g_begin; g < g_end; g++) {
            const uint8_t * record = sealed_kv + (size_t) g * tile_bytes;

            if (q_slot == 0) {
                // Four K metadata arrays (s_row, zp, s_col, row_amp) are contiguous 128-half
                // blocks; one stride-indexed load per element avoids 8 separate memcpy calls.
                const half * km = (const half *)(record + layout.k_s_row_off) + r;
                const half * vm = (const half *)(record + layout.v_s_row_off) + r;
                k_scale_sh[r] = __half2float(km[0 * KVARN_N]); // k_s_row
                k_zp_sh[r]    = __half2float(km[1 * KVARN_N]); // k_zp
                s_col_k[r]    = __half2float(km[2 * KVARN_N]); // k_s_col
                k_ramp_sh[r]  = __half2float(km[3 * KVARN_N]); // k_row_amp
                v_scale_sh[r] = __half2float(vm[0 * KVARN_N]); // v_s_row
                v_zp_sh[r]    = __half2float(vm[1 * KVARN_N]); // v_zp
                s_col_v[r]    = __half2float(vm[2 * KVARN_N]); // v_s_col
                v_ramp_sh[r]  = __half2float(vm[3 * KVARN_N]); // v_row_amp
            }
            __syncthreads();

            const uint32_t * k_payload_u32 = (const uint32_t *)(record + layout.k_payload_off);
            const uint32_t * v_payload_u32 = (const uint32_t *)(record + layout.v_payload_off);
            for (int i = threadIdx.x; i < KVARN_N * k_row_words; i += blockDim.x) {
                int r = i / k_row_words;
                int c = i % k_row_words;
                k_payload_sh_u32[r * k_stride_words + c] = k_payload_u32[i];
            }
            for (int i = threadIdx.x; i < KVARN_N * v_row_words; i += blockDim.x) {
                int r = i / v_row_words;
                int c = i % v_row_words;
                v_payload_sh_u32[r * v_stride_words + c] = v_payload_u32[i];
            }
            __syncthreads();
            const float k_scale_comb = k_ramp_sh[r] * k_scale_sh[r];
            const float k_zp_comb    = k_ramp_sh[r] * k_zp_sh[r];
            const float * q_vec      = q_sh + q_slot * KVARN_N;
            const uint32_t * k_row_words_ptr = k_payload_sh_u32 + (size_t) r * k_stride_words;
            float dot = 0.0f;

            if constexpr (KeyBits == 4) {
#pragma unroll
                for (int w = 0; w < 16; w++) {
                    const uint32_t packed = k_row_words_ptr[w];
                    const int c_base = w * 8;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        const uint32_t val = (packed >> (i * 4)) & 0xFu;
                        const int c = c_base + i;
                        const float k_val = (k_scale_comb * (float) val + k_zp_comb) * s_col_k[c];
                        dot += q_vec[c] * k_val;
                    }
                }
            } else {
#pragma unroll 4
                for (int c = 0; c < KVARN_N; c++) {
                    const uint32_t value = kvarn_unpack_row_value<KeyBits>((const uint8_t *) k_row_words_ptr, c);
                    const float k_val = (k_scale_comb * (float) value + k_zp_comb) * s_col_k[c];
                    dot += q_vec[c] * k_val;
                }
            }
            const int row_idx = q_slot * KVARN_N + r;
            row_score[row_idx] = dot * kq_scale;
            __syncthreads();

            // Warp-parallel reduction for group max (4 warps per q_slot, 128 threads).
            float m = row_score[row_idx];
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFF, m, offset));
            }
            // Each warp now has max of its 32 lanes; write lane 0 to smem,
            // then one thread reduces across the 4 warp maxes.
            // (warp_in_block and lane already declared at function head)
            if (lane == 0) warp_max[warp_in_block] = m;
            __syncthreads();
            if (r == 0) {
                const int w0 = q_slot * 4;
                float gm = warp_max[w0];
                for (int w = 1; w < 4; w++) gm = fmaxf(gm, warp_max[w0 + w]);
                group_max[q_slot] = gm;
            }
            __syncthreads();

            const float new_max = fmaxf(running_max[q_slot], group_max[q_slot]);
            const float rescale = expf(running_max[q_slot] - new_max);
            out_acc[row_idx] *= rescale;
            w_row[row_idx] = expf(row_score[row_idx] - new_max);
            __syncthreads();

            // Warp-parallel reduction for group sum.
            float s = w_row[row_idx];
#pragma unroll
            for (int offset = 16; offset >= 1; offset >>= 1) {
                s += __shfl_xor_sync(0xFFFFFFFF, s, offset);
            }
            if (lane == 0) warp_sum[warp_in_block] = s;
            __syncthreads();
            if (r == 0) {
                const int w0 = q_slot * 4;
                float gs = warp_sum[w0];
                for (int w = 1; w < 4; w++) gs += warp_sum[w0 + w];
                running_sum[q_slot] = running_sum[q_slot] * rescale + gs;
                running_max[q_slot] = new_max;
            }
            const int warp_id_in_row = r / 32;
            const size_t v_row_offset = (size_t) warp_id_in_row * ValueBits;
            const float * w_q        = w_row + q_slot * KVARN_N;

            float acc_local = 0.0f;
#pragma unroll 4
            for (int rr = 0; rr < KVARN_N; rr++) {
                const uint32_t * warp_row_words = v_payload_sh_u32 + (size_t) rr * v_stride_words + v_row_offset;
                const uint32_t value = kvarn_unpack_warp_v<ValueBits>(warp_row_words, lane);
                const float wv_scale = w_q[rr] * (v_ramp_sh[rr] * v_scale_sh[rr]);
                const float wv_zp    = w_q[rr] * (v_ramp_sh[rr] * v_zp_sh[rr]);
                acc_local += (float) value * wv_scale + wv_zp;
            }
            out_acc[row_idx] += acc_local * s_col_v[r];
            __syncthreads();
        }
    } else {
        // Dedicated tail split(s): exact rows from k_tail/v_tail, no dequant needed.
        const int n_sealed   = (n_total - tail_count) / KVARN_N;
        const int from_tail  = n_total - n_sealed * KVARN_N;
        const int tail_split = split_id - n_splits_sealed;
        const int r_offset   = tail_split * KVARN_N;
        const int valid_rows = min(KVARN_N, max(0, from_tail - r_offset));
        const size_t tail_head_stride = (size_t) tail_capacity * KVARN_N;
        const float * k_tail_h = k_tail + (size_t) kv_head * tail_head_stride + (size_t) r_offset * KVARN_N;
        const float * v_tail_h = v_tail + (size_t) kv_head * tail_head_stride + (size_t) r_offset * KVARN_N;

        if (valid_rows > 0) {
            const int tail_row_idx = q_slot * KVARN_N + r;
            float dot = 0.0f;
            const bool valid = r < valid_rows;
            if (valid) {
#pragma unroll 4
                for (int c = 0; c < KVARN_N; c++) {
                    dot += q_sh[q_slot * KVARN_N + c] * k_tail_h[r * KVARN_N + c];
                }
            }
            row_score[tail_row_idx] = valid ? dot * kq_scale : -1e30f;
            __syncthreads();

            if (r == 0) {
                float m = row_score[q_slot * KVARN_N];
                for (int i = 1; i < KVARN_N; i++) m = fmaxf(m, row_score[q_slot * KVARN_N + i]);
                group_max[q_slot] = m;
            }
            __syncthreads();

            const float new_max = fmaxf(running_max[q_slot], group_max[q_slot]);
            const float rescale = expf(running_max[q_slot] - new_max);
            out_acc[tail_row_idx] *= rescale;
            w_row[tail_row_idx] = valid ? expf(row_score[tail_row_idx] - new_max) : 0.0f;
            __syncthreads();
            if (r == 0) {
                float sum_local = 0.0f;
                for (int i = 0; i < valid_rows; i++) sum_local += w_row[q_slot * KVARN_N + i];
                running_sum[q_slot] = running_sum[q_slot] * rescale + sum_local;
                running_max[q_slot] = new_max;
            }
            __syncthreads();

            float acc_local = 0.0f;
#pragma unroll 4
            for (int rr = 0; rr < KVARN_N; rr++) {
                acc_local += w_row[q_slot * KVARN_N + rr] * v_tail_h[rr * KVARN_N + r];
            }
            out_acc[tail_row_idx] += acc_local;
            __syncthreads();
        }
    }

    const size_t out_idx = ((size_t) split_id * gridDim.x * Gqa + q_head) * KVARN_N + r;
    partial_out[out_idx] = out_acc[q_slot * KVARN_N + r];
    if (r == 0) {
        const size_t ms_idx = (size_t) split_id * gridDim.x * Gqa + q_head;
        partial_max[ms_idx] = running_max[q_slot];
        partial_sum[ms_idx] = running_sum[q_slot];
    }
}

// Merges every split's partial online-softmax state for one Q head into the
// final attention output, via the standard rescale-by-relative-max identity.
__launch_bounds__(KVARN_N, 4)
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

template<int KeyBits, int ValueBits, int Gqa>
static void ggml_cuda_op_kvarn_attn_decode_launch(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * q, const ggml_tensor * sealed,
        const ggml_tensor * k_tail, const ggml_tensor * v_tail,
        int n_head_kv, int n_groups_max,
        int n_total, int tail_count, int n_splits_sealed, int tail_capacity,
        float kq_scale, const struct kvarn_tile_layout & layout, int n_splits_total, cudaStream_t stream) {
    const int n_head_q = n_head_kv * Gqa;

    ggml_cuda_pool_alloc<float> partial_out(ctx.pool(), (size_t) n_splits_total * n_head_q * KVARN_N);
    ggml_cuda_pool_alloc<float> partial_max(ctx.pool(), (size_t) n_splits_total * n_head_q);
    ggml_cuda_pool_alloc<float> partial_sum(ctx.pool(), (size_t) n_splits_total * n_head_q);

    dim3 grid_partial(n_head_kv, n_splits_total);
    dim3 block_partial(KVARN_N * Gqa);
    k_kvarn_attn_decode_partial<KeyBits, ValueBits, Gqa><<<grid_partial, block_partial, 0, stream>>>(
            (const float *) q->data, (const uint8_t *) sealed->data,
            (const float *) k_tail->data, (const float *) v_tail->data,
            partial_out.get(), partial_max.get(), partial_sum.get(),
            n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout);
    dim3 grid_combine(n_head_q);
    dim3 block_combine(KVARN_N);
    k_kvarn_attn_decode_combine<<<grid_combine, block_combine, 0, stream>>>(
            partial_out.get(), partial_max.get(), partial_sum.get(),
            (float *) dst->data, n_splits_total, n_head_q);
}

template<int KeyBits, int ValueBits>
static void ggml_cuda_op_kvarn_attn_decode_dispatch(
        ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * q, const ggml_tensor * sealed,
        const ggml_tensor * k_tail, const ggml_tensor * v_tail,
        int n_head_kv, int n_group_broadcast, int n_groups_max,
        int n_total, int tail_count, int n_splits_sealed, int tail_capacity,
        float kq_scale, const struct kvarn_tile_layout & layout, int n_splits_total, cudaStream_t stream) {
    switch (n_group_broadcast) {
        case 1: ggml_cuda_op_kvarn_attn_decode_launch<KeyBits, ValueBits, 1>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 2: ggml_cuda_op_kvarn_attn_decode_launch<KeyBits, ValueBits, 2>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 4: ggml_cuda_op_kvarn_attn_decode_launch<KeyBits, ValueBits, 4>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 8: ggml_cuda_op_kvarn_attn_decode_launch<KeyBits, ValueBits, 8>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        default: GGML_ABORT("unsupported KVarN GQA factor");
    }
}

void ggml_cuda_op_kvarn_attn_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * q      = dst->src[0];
    const ggml_tensor * sealed = dst->src[1];
    const ggml_tensor * k_tail = dst->src[2];
    const ggml_tensor * v_tail = dst->src[3];

    int32_t key_bits, value_bits, n_head_kv, n_group_broadcast, n_total, tail_count;
    float   kq_scale;
    memcpy(&key_bits,          dst->op_params + 0, sizeof(int32_t));
    memcpy(&value_bits,        dst->op_params + 1, sizeof(int32_t));
    memcpy(&n_head_kv,         dst->op_params + 2, sizeof(int32_t));
    memcpy(&n_group_broadcast, dst->op_params + 3, sizeof(int32_t));
    memcpy(&n_total,           dst->op_params + 4, sizeof(int32_t));
    memcpy(&tail_count,        dst->op_params + 5, sizeof(int32_t));
    memcpy(&kq_scale,          dst->op_params + 6, sizeof(float));

    const int n_groups_max = (int) sealed->ne[1];
    const struct kvarn_tile_layout layout = kvarn_make_layout(key_bits, value_bits);
    const int tail_capacity = (int) k_tail->ne[1];
    const int n_sealed = (n_total - tail_count) / KVARN_N;
    const int from_tail = n_total - n_sealed * KVARN_N;
    const int n_splits_sealed = (n_sealed + KVARN_GROUPS_PER_SPLIT - 1) / KVARN_GROUPS_PER_SPLIT;
    const int n_splits_tail = from_tail > 0 ? (from_tail + KVARN_N - 1) / KVARN_N : 0;
    const int n_splits_total = n_splits_sealed + n_splits_tail;
    cudaStream_t stream = ctx.stream();

    switch (key_bits * 10 + value_bits) {
        case 22: ggml_cuda_op_kvarn_attn_decode_dispatch<2, 2>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 32: ggml_cuda_op_kvarn_attn_decode_dispatch<3, 2>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 33: ggml_cuda_op_kvarn_attn_decode_dispatch<3, 3>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 34: ggml_cuda_op_kvarn_attn_decode_dispatch<3, 4>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 42: ggml_cuda_op_kvarn_attn_decode_dispatch<4, 2>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 43: ggml_cuda_op_kvarn_attn_decode_dispatch<4, 3>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 44: ggml_cuda_op_kvarn_attn_decode_dispatch<4, 4>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 53: ggml_cuda_op_kvarn_attn_decode_dispatch<5, 3>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 54: ggml_cuda_op_kvarn_attn_decode_dispatch<5, 4>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 55: ggml_cuda_op_kvarn_attn_decode_dispatch<5, 5>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        case 66: ggml_cuda_op_kvarn_attn_decode_dispatch<6, 6>(ctx, dst, q, sealed, k_tail, v_tail, n_head_kv, n_group_broadcast, n_groups_max, n_total, tail_count, n_splits_sealed, tail_capacity, kq_scale, layout, n_splits_total, stream); break;
        default: GGML_ABORT("unsupported KVarN decode bit-width pair");
    }
}
