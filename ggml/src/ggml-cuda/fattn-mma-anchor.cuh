// AnchorKV FlashAttention decode kernel (production version).
//
// This kernel handles the single-query decode case (ncols=1) where
// Q has shape [1, D]. It reconstructs K/V on-the-fly from compressed
// AnchorKV data, so dense KV never materializes in global memory.
//
// This replaces the standard FA decode kernel for AnchorKV compressed KV.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
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

// Launch the AnchorKV fused decode kernel
void ggml_cuda_flash_attn_ext_anchor_launch(
    cudaStream_t stream,
    const float * d_Q,           // [batch * n_heads * D]
    const void * d_anchor_k,     // array of anchor_kv_head_data [batch * n_heads]
    const void * d_anchor_v,
    float * d_out,
    int D, int S, int n_heads, int batch_size
);

// Dispatch for AnchorKV FA.
// Detects AnchorKV-compressed KV and routes to the fused kernel.
void ggml_cuda_flash_attn_ext_anchor(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
);

} // namespace ggml_cuda_anchor_fa