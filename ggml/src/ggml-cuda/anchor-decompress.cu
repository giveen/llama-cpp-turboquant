// AnchorKV: CUDA kernel for GGML_OP_ANCHOR_DECOMPRESS.
//
// Reconstructs one dense KV layer from the compressed anchor-residual
// representation (see ggml_anchor_decompress in ggml.c for the tensor
// layouts) into a pre-allocated f16 scratch buffer shared across layers.
//
// The math must match the CPU reference exactly (src/anchor-kv.cpp):
//   out = gamma * anchor_vec + dequantized_2bit_residual (if slot >= 0)
// where the residual is unpacked to Lloyd-Max centroids, inverse-WHT'd
// with the deterministic seed-42 signs, and scaled by the absmax scale.

#include "anchor-decompress.cuh"
#include "common.cuh"

#include <cstdint>
#include <cuda_fp16.h>

// Lloyd-Max 2-bit centroids for a unit Gaussian source (src/anchor-kv.cpp)
__constant__ float ANCHOR_DECOMPRESS_CENTROIDS[4] = {
    -1.510138f, -0.452823f, 0.452823f, 1.510138f
};

// deterministic WHT signs (LCG, seed 42 - must match anchor_kv_get_wht_signs)
__constant__ float ANCHOR_DECOMPRESS_SIGNS[128];

static bool g_anchor_decompress_signs_ready = false;

static void anchor_decompress_init_signs() {
    if (g_anchor_decompress_signs_ready) {
        return;
    }

    float signs[128];
    uint64_t state = 42;
    for (int i = 0; i < 128; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        signs[i] = ((state >> 11) & 1) ? 1.0f : -1.0f;
    }

    cudaMemcpyToSymbol(ANCHOR_DECOMPRESS_SIGNS, signs, sizeof(signs), 0, cudaMemcpyHostToDevice);
    g_anchor_decompress_signs_ready = true;
}

// inverse WHT: 1/sqrt(d) -> butterfly -> signs (must match anchor_kv_wht_inverse)
__device__ void anchor_decompress_wht_inverse(float * x, int d) {
    const float inv_sqrt_d = rsqrtf((float) d);
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
    for (int i = 0; i < d; i++) x[i] *= ANCHOR_DECOMPRESS_SIGNS[i];
}

// One thread per KV position. Reconstructs every head of that position.
__global__ void anchor_decompress_kernel(
    const float * __restrict__ anchors,       // [n_heads, 2, k, D] f32
    const int32_t * __restrict__ anchor_of,   // [2, n_heads, S] i32
    const float * __restrict__ gamma,         // [2, n_heads, S] f32
    const int32_t * __restrict__ slot_of,     // [2, n_heads, S] i32
    const uint8_t * __restrict__ k_res_codes, // [n_heads, n_K, D/4] u8
    const float * __restrict__ k_res_scales,  // [n_heads, n_K] f32
    const uint8_t * __restrict__ v_res_codes, // [n_heads, n_V, D/4] u8
    const float * __restrict__ v_res_scales,  // [n_heads, n_V] f32
    __half * __restrict__ dst,                // [n_embd_gqa, kv_size] f16
    int S, int D, int k, int n_heads, int n_K, int n_V,
    int64_t n_embd, int64_t kv_size, int side) {

    const int64_t t = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    // Only positions [0, S) hold compressed data. Positions >= S are never
    // touched here: the scratch is zeroed once at allocation, and decode steps
    // append new tokens at S, S+1, ... via cpy_k/cpy_v. Zeroing t >= S here
    // would wipe previously appended tokens on every graph replay.
    if (t >= S) {
        return;
    }

    const int cpr = D / 4;

    for (int h = 0; h < n_heads; h++) {
        const int a    = anchor_of[(side * n_heads + h) * S + t];
        const float g  = gamma[(side * n_heads + h) * S + t];
        const int slot = slot_of[(side * n_heads + h) * S + t];

        const float * anchor_vec = anchors + ((h * 2 + side) * k + a) * D;

        if (slot >= 0) {
            const int n_res = (side == 0) ? n_K : n_V;
            const uint8_t * codes  = (side == 0 ? k_res_codes  : v_res_codes)  + (h * n_res + slot) * cpr;
            const float   * scales = (side == 0 ? k_res_scales : v_res_scales) +  h * n_res + slot;

            float deq[128];
            for (int d = 0; d < D; d++) {
                const int idx = (codes[d / 4] >> ((d % 4) * 2)) & 0x3;
                deq[d] = ANCHOR_DECOMPRESS_CENTROIDS[idx];
            }
            anchor_decompress_wht_inverse(deq, D);
            const float scale = scales[0];
            for (int d = 0; d < D; d++) {
                const float val = g * anchor_vec[d] + deq[d] * scale;
                dst[t * n_embd + h * D + d] = __float2half(val);
            }
        } else {
            for (int d = 0; d < D; d++) {
                const float val = g * anchor_vec[d];
                dst[t * n_embd + h * D + d] = __float2half(val);
            }
        }
    }
}

void ggml_cuda_anchor_decompress(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * anchors      = dst->src[0];
    const ggml_tensor * anchor_of    = dst->src[1];
    const ggml_tensor * gamma        = dst->src[2];
    const ggml_tensor * slot_of      = dst->src[3];
    const ggml_tensor * k_res_codes  = dst->src[4];
    const ggml_tensor * k_res_scales = dst->src[5];
    const ggml_tensor * v_res_codes  = dst->src[6];
    const ggml_tensor * v_res_scales = dst->src[7];

    int is_k = 0;
    memcpy(&is_k, dst->op_params, sizeof(int));

    const int n_heads = (int) anchors->ne[0];
    const int k       = (int) anchors->ne[2];
    const int D       = (int) anchors->ne[3];
    const int S       = (int) anchor_of->ne[2];
    const int n_K     = (int) k_res_codes->ne[1];
    const int n_V     = (int) v_res_codes->ne[1];
    const int64_t n_embd   = dst->ne[0];
    const int64_t kv_size  = dst->ne[1];

    anchor_decompress_init_signs();

    const dim3 block = 256;
    const dim3 grid  = (unsigned int) ((S + block.x - 1) / block.x);

    const int side = is_k ? 0 : 1;

    anchor_decompress_kernel<<<grid, block, 0, ctx.stream()>>>(
        (const float *)   anchors->data,
        (const int32_t *) anchor_of->data,
        (const float *)   gamma->data,
        (const int32_t *) slot_of->data,
        (const uint8_t *) k_res_codes->data,
        (const float *)   k_res_scales->data,
        (const uint8_t *) v_res_codes->data,
        (const float *)   v_res_scales->data,
        (__half *)        dst->data,
        S, D, k, n_heads, n_K, n_V,
        n_embd, kv_size, side);
}
