// AnchorKV GPU decompression kernel.
//
// Converts compressed AnchorKV representation -> dense f16 KV cache.
// This is the first step toward GPU integration: the dense KV is then
// consumed by the existing FlashAttention kernel.
//
// The compressed format per head (stored in global memory):
//   anchor_keys:    [k, D] in fp32     -- exact anchor K vectors
//   anchor_values:  [k, D] in fp32     -- exact anchor V vectors
//   anchor_ids:     [S] in int32       -- which anchor each position maps to (K side)
//   anchor_ids_v:   [S] in int32       -- which anchor each position maps to (V side)
//   gamma_k:        [S] in fp32        -- projection coefficient (K side)
//   gamma_v:        [S] in fp32        -- projection coefficient (V side)
//   residual_mask_k: [ceil(S/64)] in uint64 -- bitmask for K residuals
//   residual_mask_v: [ceil(S/64)] in uint64 -- bitmask for V residuals
//   res_codes_k:    [N_K * D/4] in uint8   -- packed 2-bit K residual codes
//   res_scales_k:   [N_K] in fp32          -- per-token absmax scale
//   res_codes_v:    [N_V * D/4] in uint8   -- packed 2-bit V residual codes
//   res_scales_v:   [N_V] in fp32          -- per-token absmax scale
//   res_positions_v: [N_V] in uint16       -- position index per V residual slot
//
// Output: dense K/V in f16, shape [D, S] per head

#include "common.cuh"
#include "anchor-kv-common.cuh"

#include <cuda_fp16.h>
#include <cstdint>

// Lloyd-Max 2-bit centroids for unit Gaussian (from anchor-kv.cpp)
__constant__ float LLOYD_CENTROIDS_2BIT_D[4] = {
    -1.510138f, -0.452823f, 0.452823f, 1.510138f
};

// WHT signs (generated from seed=42, stored in constant memory)
// For D=128, we need 128 signs. Generated at kernel launch time.
__constant__ float WHT_SIGNS_D[128];

// Forward WHT (in-place, registers only for D<=128)
__device__ void anchor_wht_forward(float *x, int d) {
    // Apply signs
    for (int i = 0; i < d; i++) x[i] *= WHT_SIGNS_D[i];
    // Butterfly
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float inv_sqrt_d = rsqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= inv_sqrt_d;
}

// Inverse WHT (in-place)
__device__ void anchor_wht_inverse(float *x, int d) {
    float inv_sqrt_d = rsqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= inv_sqrt_d;
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < d; i++) x[i] *= WHT_SIGNS_D[i];
}

// Dequantize one 2-bit residual from packed codes
__device__ void dequant_residual_2bit(
    const uint8_t * codes, int d, float scale, float * out
) {
    // Unpack codes
    for (int i = 0; i < d; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        out[i] = LLOYD_CENTROIDS_2BIT_D[idx];
    }
    // Inverse WHT
    anchor_wht_inverse(out, d);
    // Scale
    for (int i = 0; i < d; i++) out[i] *= scale;
}

// Kernel: decompress one KV side (K or V) for all positions in a head
// Grid: (S, 1, 1) -- one thread per position
// Each thread reconstructs D elements for one position
__global__ void anchor_kv_decompress_kernel(
    // Compressed inputs
    const float * anchor_vecs,   // [k, D] anchor K or V vectors
    const int   * anchor_ids,    // [S] anchor index per position
    const float * gamma,         // [S] projection coefficient
    const int   * slot_of,       // [S] precomputed slot index (-1 if no residual)
    const uint8_t * res_codes,   // [N_res * D/4] packed residual codes
    const float * res_scales,    // [N_res] per-token absmax scale
    int S, int D, int k,
    // Output
    float * out_dense            // [S * D] dense output in fp32
) {
    int t = blockIdx.x;  // position index
    if (t >= S) return;

    int anchor_idx = anchor_ids[t];
    float g = gamma[t];

    // Reconstruct: out = gamma * anchor
    for (int d = 0; d < D; d++) {
        out_dense[t * D + d] = g * anchor_vecs[anchor_idx * D + d];
    }

    // Add residual if this position has one (O(1) lookup via precomputed slot)
    int slot = slot_of[t];
    if (slot >= 0) {
        // Dequantize residual
        float deq[128];
        size_t codes_per_res = (size_t)D / 4;
        dequant_residual_2bit(
            &res_codes[slot * codes_per_res],
            D, res_scales[slot],
            deq
        );

        // Add to reconstruction
        for (int d = 0; d < D; d++) {
            out_dense[t * D + d] += deq[d];
        }
    }
}

// Static global for FA kernel to access compressed data
// Set by anchor_kv_set_current_layer() before each FA call
anchor_kv_data_t g_anchor_kv_current_data = {};
static int g_anchor_kv_enabled = 0;

extern "C" void anchor_kv_set_current_layer(
    const float * d_anchor_keys, const float * d_anchor_values,
    const int * d_k_anchor_of, const int * d_v_anchor_of,
    const float * d_k_gamma, const float * d_v_gamma,
    const int * d_k_slot_of, const int * d_v_slot_of,
    const uint8_t * d_k_res_codes, const float * d_k_res_scales,
    const uint8_t * d_v_res_codes, const float * d_v_res_scales,
    int S, int D, int k, int n_K, int n_V
) {
    g_anchor_kv_current_data.anchor_keys = d_anchor_keys;
    g_anchor_kv_current_data.anchor_values = d_anchor_values;
    g_anchor_kv_current_data.k_anchor_of = d_k_anchor_of;
    g_anchor_kv_current_data.v_anchor_of = d_v_anchor_of;
    g_anchor_kv_current_data.k_gamma = d_k_gamma;
    g_anchor_kv_current_data.v_gamma = d_v_gamma;
    g_anchor_kv_current_data.k_slot_of = d_k_slot_of;
    g_anchor_kv_current_data.v_slot_of = d_v_slot_of;
    g_anchor_kv_current_data.k_res_codes = d_k_res_codes;
    g_anchor_kv_current_data.k_res_scales = d_k_res_scales;
    g_anchor_kv_current_data.v_res_codes = d_v_res_codes;
    g_anchor_kv_current_data.v_res_scales = d_v_res_scales;
    g_anchor_kv_current_data.S = S;
    g_anchor_kv_current_data.D = D;
    g_anchor_kv_current_data.k = k;
    g_anchor_kv_current_data.n_K = n_K;
    g_anchor_kv_current_data.n_V = n_V;
    g_anchor_kv_enabled = 1;
}

extern "C" void anchor_kv_clear_current_layer() {
    g_anchor_kv_enabled = 0;
}

extern "C" int anchor_kv_is_enabled() {
    return g_anchor_kv_enabled;
}

extern "C" void anchor_kv_decompress_gpu(
    cudaStream_t stream,
    // Compressed data (device pointers)
    const float * d_anchor_keys,   // [n_heads, k, D]
    const float * d_anchor_values, // [n_heads, k, D]
    const int   * d_k_anchor_of,   // [n_heads, S]
    const int   * d_v_anchor_of,   // [n_heads, S]
    const float * d_k_gamma,       // [n_heads, S]
    const float * d_v_gamma,       // [n_heads, S]
    const int   * d_k_slot_of,     // [n_heads, S] precomputed slot indices
    const int   * d_v_slot_of,     // [n_heads, S] precomputed slot indices
    const uint8_t * d_k_res_codes, // [n_heads, N_K * D/4]
    const float * d_k_res_scales,  // [n_heads, N_K]
    const uint8_t * d_v_res_codes, // [n_heads, N_V * D/4]
    const float * d_v_res_scales,  // [n_heads, N_V]
    // Output
    float * d_out_k,               // [n_heads, S * D]
    float * d_out_v,               // [n_heads, S * D]
    // Dimensions
    int S, int D, int k, int n_heads,
    int n_K, int n_V
) {
    dim3 grid(S);
    dim3 block(1);

    for (int h = 0; h < n_heads; h++) {
        // Decompress K
        anchor_kv_decompress_kernel<<<grid, block, 0, stream>>>(
            d_anchor_keys + (size_t)h * k * D,
            d_k_anchor_of + (size_t)h * S,
            d_k_gamma + (size_t)h * S,
            d_k_slot_of + (size_t)h * S,
            d_k_res_codes + (size_t)h * n_K * (D / 4),
            d_k_res_scales + (size_t)h * n_K,
            S, D, k,
            d_out_k + (size_t)h * S * D
        );

        // Decompress V
        anchor_kv_decompress_kernel<<<grid, block, 0, stream>>>(
            d_anchor_values + (size_t)h * k * D,
            d_v_anchor_of + (size_t)h * S,
            d_v_gamma + (size_t)h * S,
            d_v_slot_of + (size_t)h * S,
            d_v_res_codes + (size_t)h * n_V * (D / 4),
            d_v_res_scales + (size_t)h * n_V,
            S, D, k,
            d_out_v + (size_t)h * S * D
        );
    }
}
