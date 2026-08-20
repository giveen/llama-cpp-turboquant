// AnchorKV FlashAttention integration header.
//
// Declares the C++ (not extern "C") functions used by the FA dispatch
// to route to the fused AnchorKV kernel. Using normal C++ linkage avoids
// the extern "C" + C++-reference mismatch that broke previous attempts.

#pragma once

#include "anchor-kv-common.cuh"

// Returns true if AnchorKV compressed data is set for the current layer
bool anchor_kv_fa_enabled();

// Returns the current layer's compressed data (device pointers)
const anchor_kv_data_t & anchor_kv_fa_get_state();

// Launch the fused AnchorKV FA decode kernel (single-query decode mode).
// Reconstructs K/V from compressed data on-the-fly; never reads dense KV.
void anchor_kv_fa_decode_launch(
    cudaStream_t stream,
    const float * d_Q,          // [D] query on device
    const anchor_kv_data_t * data, // compressed layer data on device
    float * d_out,              // [D] output on device
    int D
);
