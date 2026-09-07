#pragma once

#include "common.cuh"

// KVarN fused decode attention: for a single query per head, computes
// softmax(Q.K^T / sqrt(d)) . V directly against kvarn_sealed/kvarn_k_tail/
// kvarn_v_tail, without materializing the full dequantized history to
// global memory first (see ggml_cuda_op_kvarn_materialize in kvarn.cu,
// which this bypasses for the decode case specifically - prefill/multi-
// token ubatches still use that path). See kvarn-attn-decode.cu for the
// rotation-commutes-with-attention argument this relies on.

void ggml_cuda_op_kvarn_attn_decode(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
