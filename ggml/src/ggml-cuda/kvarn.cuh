#pragma once

#include "common.cuh"

// KVarN: variance-normalized KV-cache quantization CUDA kernels.
// dst->src[0] = k_tail/sealed, dst->src[1] = v_tail/tail - see
// ggml_kvarn_seal/ggml_kvarn_materialize in ggml.c for exact tensor shapes
// and op_params layout; kernels here mirror ggml-kvarn-quant.c's CPU
// reference math exactly (see that file's comments for the algorithm).

void ggml_cuda_op_kvarn_seal(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_kvarn_materialize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
