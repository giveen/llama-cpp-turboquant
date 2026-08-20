// AnchorKV: CUDA forward for GGML_OP_ANCHOR_DECOMPRESS.
// Reconstructs one dense KV layer from the compressed anchor-residual
// representation directly into the pre-allocated f16 scratch buffer.

#pragma once

#include "common.cuh"

void ggml_cuda_anchor_decompress(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
