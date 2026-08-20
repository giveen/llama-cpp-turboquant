// Shared AnchorKV data structures for CUDA kernels.
// Used by both the decompress kernel and the FA fused kernel.

#pragma once

#include <cstdint>

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
