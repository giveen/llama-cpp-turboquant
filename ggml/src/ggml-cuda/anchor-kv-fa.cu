// AnchorKV FA integration implementation.
//
// Defines the functions declared in anchor-kv-fa.h. This file has access
// to CUDA APIs directly (compiled by nvcc), so it can launch kernels and
// access the compressed data static state.

#include "common.cuh"
#include "anchor-kv-common.cuh"
#include "anchor-kv-fa.h"

// Static state set by the KV cache (anchor-kv-decompress.cu)
#include "anchor-kv-fa-state.h"

// -------------------------------------------------------------------------
// Fused AnchorKV FA decode kernel
// -------------------------------------------------------------------------

// Lloyd-Max 2-bit centroids (constant memory)
__constant__ float ANCHOR_FA_CENTROIDS[4] = {-1.510138f, -0.452823f, 0.452823f, 1.510138f};

// WHT signs (constant memory, up to 128 elements)
__constant__ float ANCHOR_FA_WHT_SIGNS[128];

// Forward WHT (registers)
__device__ void anchor_fa_wht_fwd(float * x, int d, const float * signs) {
    for (int i = 0; i < d; i++) x[i] *= signs[i];
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    float s = rsqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= s;
}

// Inverse WHT (registers)
__device__ void anchor_fa_wht_inv(float * x, int d, const float * signs) {
    float s = rsqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= s;
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < d; i++) x[i] *= signs[i];
}

// Dequantize a 2-bit residual
__device__ void anchor_fa_dequant_2bit(
    const uint8_t * codes, int d, float scale, const float * signs, float * out
) {
    for (int i = 0; i < d; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        out[i] = ANCHOR_FA_CENTROIDS[idx];
    }
    anchor_fa_wht_inv(out, d, signs);
    for (int i = 0; i < d; i++) out[i] *= scale;
}

// Reconstruct a single K/V vector from compressed data
__device__ void anchor_fa_reconstruct(
    const float * anchor_vecs, int anchor_idx, float gamma, int slot,
    const uint8_t * res_codes, const float * res_scales,
    const float * signs, int D, float * out
) {
    for (int d = 0; d < D; d++) {
        out[d] = gamma * anchor_vecs[anchor_idx * D + d];
    }
    if (slot >= 0) {
        float deq[128];
        size_t cpr = (size_t)D / 4;
        anchor_fa_dequant_2bit(&res_codes[slot * cpr], D, res_scales[slot], signs, deq);
        for (int d = 0; d < D; d++) out[d] += deq[d];
    }
}

// Fused AnchorKV FA decode kernel.
// One block per KV tile. blockDim = D (one thread per dimension).
template <int D>
__global__ void anchor_fa_decode_kernel(
    const float * Q,
    anchor_kv_data_t data,
    float * out,
    int tile_start, int tile_size
) {
    extern __shared__ float smem[];
    float * sk = smem;             // reconstructed K for current position
    float * sv = smem + D;         // reconstructed V for current position
    float * s_weights = smem + 2*D; // softmax weights for tile

    int tid = threadIdx.x;
    if (tid >= D) return;

    float scale = rsqrtf((float)D);

    // WHT signs (registers/local)
    float signs[128];
    for (int i = 0; i < D; i++) signs[i] = ANCHOR_FA_WHT_SIGNS[i];

    // Pass 1: compute scores for this tile
    float max_score = -1e30f;
    float rowsum = 0.0f;

    for (int i = 0; i < tile_size; i++) {
        int t = tile_start + i;
        if (t >= data.S) break;

        anchor_fa_reconstruct(
            data.anchor_keys, data.k_anchor_of[t], data.k_gamma[t],
            data.k_slot_of[t], data.k_res_codes, data.k_res_scales,
            signs, D, sk
        );

        float score = 0.0f;
        for (int d = 0; d < D; d++) score += Q[d] * sk[d];
        score *= scale;

        float new_max = fmaxf(max_score, score);
        rowsum = rowsum * expf(max_score - new_max) + expf(score - new_max);
        max_score = new_max;
        s_weights[i] = score;
    }

    // Pass 2: reconstruct V and accumulate
    for (int d = 0; d < D; d++) out[d] = 0.0f;

    for (int i = 0; i < tile_size; i++) {
        int t = tile_start + i;
        if (t >= data.S) break;

        anchor_fa_reconstruct(
            data.anchor_values, data.v_anchor_of[t], data.v_gamma[t],
            data.v_slot_of[t], data.v_res_codes, data.v_res_scales,
            signs, D, sv
        );

        float w = expf(s_weights[i] - max_score) / rowsum;
        for (int d = 0; d < D; d++) out[d] += w * sv[d];
    }
}

bool anchor_kv_fa_enabled() {
    // Read the shared flag set by the KV cache
    return anchor_kv_fa_is_enabled();
}

const anchor_kv_data_t & anchor_kv_fa_get_state() {
    return anchor_kv_fa_read_state();
}

void anchor_kv_fa_decode_launch(
    cudaStream_t stream,
    const float * d_Q,
    const anchor_kv_data_t * data,
    float * d_out,
    int D
) {
    const int TILE_SIZE = 64;
    int n_tiles = (data->S + TILE_SIZE - 1) / TILE_SIZE;

    // Allocate accumulation buffer
    float * d_acc;
    cudaMalloc(&d_acc, n_tiles * D * sizeof(float));
    cudaMemset(d_acc, 0, n_tiles * D * sizeof(float));

    dim3 grid(n_tiles);
    dim3 block(D);
    size_t smem_size = 3 * D * sizeof(float);

    anchor_fa_decode_kernel<128><<<grid, block, smem_size, stream>>>(
        d_Q, *data, d_acc, 0, TILE_SIZE
    );

    // Copy first tile result (single-tile approximation for reference)
    cudaMemcpy(d_out, d_acc, D * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_acc);
}
