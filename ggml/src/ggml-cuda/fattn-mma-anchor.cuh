// AnchorKV FlashAttention decode kernel (CPU-side launcher).
//
// This kernel reads compressed AnchorKV data and reconstructs K/V on-the-fly
// during the FlashAttention inner loop. The compressed format stores:
//   - Anchor K/V vectors (exact)
//   - Per-token projection coefficients (gamma)
//   - Per-token anchor indices
//   - Sparse 2-bit residual codes + scales
//
// For the first implementation, we decompress the full KV cache to f16 on GPU
// before running standard FA. This validates the pipeline. The fused on-the-fly
// reconstruction kernel is a follow-up optimization.

#pragma once

#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"

// Launch standard f16 FA on decompressed KV data.
// The caller is responsible for decompressing AnchorKV -> f16 before calling this.
template <int DKQ, int DV, int ncols1, int ncols2>
void ggml_cuda_flash_attn_ext_anchor_decompress_and_run(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    // For now, route to the standard f16 MMA path.
    // The AnchorKV decompression happens in the graph builder (llama-graph.cpp)
    // which decompresses KV to f16 before the FA op.
    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, ncols1, ncols2>(ctx, dst);
}

// Dispatch for AnchorKV FA.
// Detects AnchorKV-compressed KV and routes to appropriate kernel.
void ggml_cuda_flash_attn_ext_anchor(
    ggml_backend_cuda_context & ctx,
    ggml_tensor * dst
) {
    // For the initial implementation, if KV has been decompressed to f16
    // by the graph builder, just run standard f16 FA.
    // The AnchorKV-specific dispatch will be added when the fused kernel is ready.
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
}
