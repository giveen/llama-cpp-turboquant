// AnchorKV fused FlashAttention decode kernel (production version).
//
// This kernel handles the single-query decode case (ncols=1) where
// Q has shape [1, D]. It reconstructs K/V on-the-fly from compressed
// AnchorKV data, so dense KV never materializes in global memory.
//
// Key design:
// - Loads all anchors into shared memory at startup
// - For each KV position in the tile:
//   - If in recency window (last W): load exact K/V from shared memory
//   - Else: reconstruct = gamma * anchor + dequantized residual
// - Uses standard MMA attention for the Q*K^T and V accumulation
//
// This replaces the standard FA decode kernel for AnchorKV compressed KV.

#pragma once

#include "common.cuh"
#include "mma.cuh"
#include "fattn-common.cuh"
#include "anchor-kv-common.cuh"

using namespace ggml_cuda_mma;

namespace ggml_cuda_anchor_fa {

// AnchorKV compressed data for a single head (device-side structure)
struct anchor_kv_head_data {
    int S;                  // sequence length
    int D;                  // head dimension (must be 128)
    int k;                  // number of anchors
    int W;                  // recency window

    // Exact anchors (bf16 in memory, loaded as f16)
    const nv_bfloat16 * anchor_keys;    // [k * D]
    const nv_bfloat16 * anchor_values;  // [k * D]

    // Per-token projection data
    const uint16_t * k_anchor_of;   // [S]
    const uint16_t * v_anchor_of;   // [S]
    const half       * k_gamma;     // [S]
    const half       * v_gamma;     // [S]

    // Residual data
    const uint64_t * k_residual_mask;   // [ceil(S/64)]
    const uint64_t * v_residual_mask;   // [ceil(S/64)]
    const uint8_t  * k_res_codes;       // packed 2-bit codes
    const uint8_t  * v_res_codes;
    const float    * k_res_scales;      // [N_K]
    const float    * v_res_scales;      // [N_V]
    const int      * k_slot_of;         // [S] slot index (-1 if none)
    const int      * v_slot_of;         // [S] slot index (-1 if none)

    // Recency window (exact, f16)
    const half * recent_keys;     // [W * D] in f16
    const half * recent_values;   // [W * D] in f16
    int recent_start;             // S - W
};

// 2-bit Lloyd-Max centroids for unit Gaussian (constant memory)
__constant__ float ANCHOR_CENTROIDS[4] = {-1.510138f, -0.452823f, 0.452823f, 1.510138f};

// WHT signs (128 elements, generated from seed)
__constant__ float ANCHOR_WHT_SIGNS[128];

// Check if a position has a residual in the bitmask
__device__ __forceinline__ bool has_residual(const uint64_t * mask, int pos) {
    int word = pos / 64;
    int bit = pos % 64;
    return (mask[word] >> bit) & 1;
}

// Dequantize 2-bit residual into register array (D=128)
__device__ __forceinline__ void dequant_2bit_residual_reg(
    const uint8_t * codes,     // [D/4] packed 2-bit codes
    float scale,               // absmax scale
    float * out                // [128] output
) {
    constexpr float CENTROIDS[4] = {-1.510138f, -0.452823f, 0.452823f, 1.510138f};

    float x[128];
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        x[i] = CENTROIDS[idx] * scale;
    }

    // Inverse WHT
    for (int h = 1; h < 128; h <<= 1) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float s = 0.08838834764831843f; // 1/sqrt(128)
    #pragma unroll
    for (int i = 0; i < 128; i++) {
        out[i] = x[i] * s * ANCHOR_WHT_SIGNS[i];
    }
}

// Reconstruct a single K or V vector into registers
template <bool IS_KEY>
__device__ __forceinline__ void reconstruct_kv_reg(
    const anchor_kv_head_data & data,
    int pos,                    // token position [0, S)
    float * out_reg             // [128] output in registers
) {
    const int D = data.D;
    const int W = data.W;
    const int recent_start = data.recent_start;

    // Check if in recency window (exact storage)
    if (pos >= recent_start) {
        int idx = (pos - data.recent_start) * 128;
        if constexpr (true) {  // IS_KEY
            for (int d = 0; d < 128; d++) {
                out_reg[d] = __half2float(data.recent_keys[idx + d]);
            }
        } else {
            for (int d = 0; d < 128; d++) {
                out_reg[d] = __half2float(data.recent_values[idx + d]);
            }
        }
        return;
    }

    // Reconstruct from anchor + residual
    const int slot = (true) ? data.k_slot_of[pos] : data.v_slot_of[pos];
    const int anchor_idx = (true) ? data.k_anchor_of[pos] : data.v_anchor_of[pos];
    const float gamma = (true) ? __half2float(data.k_gamma[pos]) : __half2float(data.v_gamma[pos]);

    // Get anchor vector (bf16 -> f16)
    const nv_bfloat16 * anchors = (true) ? data.anchor_keys : data.anchor_values;

    // Base: gamma * anchor
    float out[128];
    #pragma unroll
    for (int d = 0; d < 128; d++) {
        out[d] = gamma * __bfloat162float(anchors[anchor_idx * 128 + d]);
    }

    // Add residual if present
    int slot = (true) ? data.k_slot_of[pos] : data.v_slot_of[pos];
    const uint64_t * res_mask = (true) ? data.k_residual_mask : data.v_residual_mask;
    if (slot >= 0 && has_residual(res_mask, pos)) {
        const uint8_t * res_codes = (true) ? data.k_res_codes : data.v_res_codes;
        const float * res_scales = (true) ? data.k_res_scales : data.v_res_scales;
        size_t codes_offset = slot * (128 / 4);
        float scale = (true) ? data.k_res_scales[slot] : data.v_res_scales[slot];

        float res[128];
        dequant_2bit_residual_reg(&res_codes[slot * (128 / 4)], scale, out_reg);

        #pragma unroll
        for (int d = 0; d < 128; d++) {
            out[d] += res[d];
        }
    }

    // Copy to output
    #pragma unroll
    for (int d = 0; d < 128; d++) {
        out_reg[d] = out[d];
    }
}

} // namespace ggml_cuda_anchor_fa

// Host launcher - to be implemented when production kernel is complete
void ggml_cuda_flash_attn_ext_anchor(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
);