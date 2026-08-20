// AnchorKV fused FlashAttention decode kernel.
//
// This kernel reconstructs KV from compressed AnchorKV data inside the
// FA inner loop, so dense KV never exists in global memory.
//
// For decode (single query), each thread block processes a tile of KV
// positions. For each position in the tile:
//   1. Read anchor index + coefficient
//   2. Fetch anchor value from shared memory
//   3. Fetch and dequantize residual (2-bit WHT)
//   4. Reconstruct: gamma * anchor + residual
//   5. Compute Q*K^T and accumulate V*softmax
//
// This is a simplified reference kernel. The production version would
// use MMA instructions and pipeline stages like the existing turbo kernel.

#include "common.cuh"
#include <cuda_fp16.h>
#include <cstdint>

// AnchorKV compressed data layout (per head, in global memory):
struct anchor_kv_data_t {
    const float * anchor_keys;      // [k, D]
    const float * anchor_values;    // [k, D]
    const int   * k_anchor_of;      // [S] anchor index per position (K side)
    const int   * v_anchor_of;      // [S] anchor index per position (V side)
    const float * k_gamma;          // [S] projection coefficient (K side)
    const float * v_gamma;          // [S] projection coefficient (V side)
    const int   * k_slot_of;        // [S] residual slot index (-1 if none)
    const int   * v_slot_of;        // [S] residual slot index (-1 if none)
    const uint8_t * k_res_codes;    // [N_K * D/4] packed residual codes
    const float * k_res_scales;     // [N_K] per-token scale
    const uint8_t * v_res_codes;    // [N_V * D/4] packed residual codes
    const float * v_res_scales;     // [N_V] per-token scale
    int S, D, k, n_K, n_V;
};

// Lloyd-Max 2-bit centroids (constant memory)
__constant__ float ANCHOR_CENTROIDS[4] = {-1.510138f, -0.452823f, 0.452823f, 1.510138f};

// WHT signs (constant memory, 128 elements)
__constant__ float ANCHOR_WHT_SIGNS[128];

// Forward WHT (registers only, D<=128)
__device__ void anchor_wht_fwd(float *x, int d) {
    for (int i = 0; i < d; i++) x[i] *= ANCHOR_WHT_SIGNS[i];
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

// Inverse WHT
__device__ void anchor_wht_inv(float *x, int d) {
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
    for (int i = 0; i < d; i++) x[i] *= ANCHOR_WHT_SIGNS[i];
}

// Dequantize 2-bit residual
__device__ void dequant_2bit(const uint8_t * codes, int d, float scale, float * out) {
    for (int i = 0; i < d; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        out[i] = ANCHOR_CENTROIDS[idx];
    }
    anchor_wht_inv(out, d);
    for (int i = 0; i < d; i++) out[i] *= scale;
}

// Reconstruct one KV vector from compressed data
__device__ void reconstruct_kv(
    const float * anchor_vecs,  // [k, D]
    int anchor_idx,
    float gamma,
    int slot,
    const uint8_t * res_codes,
    const float * res_scales,
    int D,
    float * out               // [D] reconstructed vector
) {
    // Projection: out = gamma * anchor
    for (int d = 0; d < D; d++) {
        out[d] = gamma * anchor_vecs[anchor_idx * D + d];
    }

    // Add residual if present
    if (slot >= 0) {
        float deq[128];
        size_t cpr = (size_t)D / 4;
        dequant_2bit(&res_codes[slot * cpr], D, res_scales[slot], deq);
        for (int d = 0; d < D; d++) out[d] += deq[d];
    }
}

// Fused AnchorKV FA decode kernel (simplified reference).
// One block per KV tile. blockDim = D (one thread per dimension).
// GridDim = ceil(S / TILE_SIZE) for the KV sequence.
//
// For decode: Q is [1, D] (single query), output is [1, D].
template <int D>
__global__ void anchor_kv_fa_decode_kernel(
    // Query
    const float * Q,           // [D] query vector
    // AnchorKV compressed data
    const anchor_kv_data_t data,
    // Output
    float * out,               // [D] attention output
    // Parameters
    int tile_start, int tile_size
) {
    extern __shared__ float smem[];
    // smem layout: [D] for reconstructed K, [D] for reconstructed V, [D] for softmax weights

    float * sk = smem;                    // reconstructed K for current position
    float * sv = smem + D;                // reconstructed V for current position
    float * s_weights = smem + 2 * D;     // softmax weights for the tile

    int tid = threadIdx.x;  // dimension index
    if (tid >= D) return;

    float scale = rsqrtf((float)D);
    float max_score = -1e30f;
    float rowsum = 0.0f;

    // Pass 1: compute attention scores for this tile
    for (int i = 0; i < tile_size; i++) {
        int t = tile_start + i;
        if (t >= data.S) break;

        // Reconstruct K for this position
        reconstruct_kv(
            data.anchor_keys, data.k_anchor_of[t], data.k_gamma[t],
            data.k_slot_of[t], data.k_res_codes, data.k_res_scales,
            D, sk
        );

        // Compute Q . K_t
        float score = 0.0f;
        for (int d = 0; d < D; d++) {
            score += Q[d] * sk[d];
        }
        score *= scale;

        // Online softmax
        float new_max = fmaxf(max_score, score);
        rowsum = rowsum * expf(max_score - new_max) + expf(score - new_max);
        max_score = new_max;
        s_weights[i] = score;
    }

    // Pass 2: reconstruct V and accumulate output
    for (int d = 0; d < D; d++) out[d] = 0.0f;

    for (int i = 0; i < tile_size; i++) {
        int t = tile_start + i;
        if (t >= data.S) break;

        // Reconstruct V for this position
        reconstruct_kv(
            data.anchor_values, data.v_anchor_of[t], data.v_gamma[t],
            data.v_slot_of[t], data.v_res_codes, data.v_res_scales,
            D, sv
        );

        // Softmax weight
        float w = expf(s_weights[i] - max_score) / rowsum;

        // Accumulate: out += w * V_t
        for (int d = 0; d < D; d++) {
            out[d] += w * sv[d];
        }
    }
}

// Host-side launcher
extern "C" void anchor_kv_fa_decode_launch(
    cudaStream_t stream,
    const float * d_Q,           // [D] query
    const anchor_kv_data_t & data,
    float * d_out,               // [D] output
    int D
) {
    const int TILE_SIZE = 64;  // positions per tile
    int n_tiles = (data.S + TILE_SIZE - 1) / TILE_SIZE;

    // Temporary accumulation buffer
    float * d_acc;
    cudaMalloc(&d_acc, n_tiles * D * sizeof(float));
    cudaMemset(d_acc, 0, n_tiles * D * sizeof(float));

    // Launch one block per tile
    dim3 grid(n_tiles);
    dim3 block(D);
    size_t smem_size = 3 * D * sizeof(float);

    anchor_kv_fa_decode_kernel<128><<<grid, block, smem_size, stream>>>(
        d_Q, data, d_acc, 0, TILE_SIZE
    );

    // Reduce across tiles (simple sequential reduction)
    // TODO: parallel reduction
    float * h_acc = (float *)malloc(n_tiles * D * sizeof(float));
    cudaMemcpy(h_acc, d_acc, n_tiles * D * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemset(d_out, 0, D * sizeof(float));
    for (int i = 0; i < n_tiles * D; i++) {
        // Simple: just take the first tile's result for now
        // Real implementation would do proper reduction
    }
    // For reference: just copy first tile
    cudaMemcpy(d_out, d_acc, D * sizeof(float), cudaMemcpyDeviceToDevice);

    free(h_acc);
    cudaFree(d_acc);
}
