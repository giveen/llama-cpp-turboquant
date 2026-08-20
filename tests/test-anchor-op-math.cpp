/*
 * Test: GGML_OP_ANCHOR_DECOMPRESS math vs reference anchor_kv_decompress_head.
 *
 * Build:
 *   g++ -O2 -std=c++17 -I../src -I../include -I../ggml/include \
 *       -o test-anchor-op-math test-anchor-op-math.cpp ../src/anchor-kv.cpp -lm
 *
 * Run:
 *   ./test-anchor-op-math
 *
 * The test replicates the layout/arithmetic of ggml_compute_forward_anchor_decompress_f16
 * (ggml/src/ggml-cpu/ops.cpp) in plain vectors, so a mismatch here means the op code
 * diverges from the reference codec.
 */

#include "anchor-kv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ---- exact copies of the op helpers (must stay in sync with ops.cpp) ----

static const float op_lloyd_centroids[4] = {
    -1.510138f, -0.452823f, 0.452823f, 1.510138f
};

static void op_get_wht_signs(float * signs, int d) {
    uint64_t state = 42;
    for (int i = 0; i < d; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        signs[i] = ((state >> 11) & 1) ? 1.0f : -1.0f;
    }
}

static void op_wht_inverse(float * x, int d, const float * signs) {
    const float inv_sqrt_d = 1.0f / sqrtf((float) d);
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
    for (int i = 0; i < d; i++) x[i] *= signs[i];
}

// replicate the op for one layer; outputs are [n_embd_gqa * kv_size] f16 values
static void op_decompress_layer(
        const anchor_kv_layer & layer,
        int n_heads, int S, int D,
        float * out_k, /* [n_embd_gqa * S] */
        float * out_v) {
    int k = layer.heads[0].k;

    // build the uploaded tensor layouts (same as llama_kv_cache::anchor_kv_upload_layer)
    std::vector<ggml_bf16_t> anchors(n_heads * 2 * k * D, ggml_fp32_to_bf16(0.0f));
    std::vector<int32_t> anchor_of(2 * n_heads * S, 0);
    std::vector<ggml_fp16_t> gamma(2 * n_heads * S, ggml_fp32_to_fp16(0.0f));
    std::vector<int32_t> slot_of(2 * n_heads * S, -1);

    int n_K_max = 0, n_V_max = 0;
    for (int h = 0; h < n_heads; h++) {
        n_K_max = std::max(n_K_max, layer.heads[h].n_K);
        n_V_max = std::max(n_V_max, layer.heads[h].n_V);
    }
    const int cpr = D / 4;
    std::vector<uint8_t> k_codes((size_t) n_heads * n_K_max * cpr, 0);
    std::vector<float> k_scales((size_t) n_heads * n_K_max, 0.0f);
    std::vector<uint8_t> v_codes((size_t) n_heads * n_V_max * cpr, 0);
    std::vector<float> v_scales((size_t) n_heads * n_V_max, 0.0f);

    for (int h = 0; h < n_heads; h++) {
        const anchor_kv_head & head = layer.heads[h];
        for (int a = 0; a < head.k; a++) {
            ggml_fp32_to_bf16_row(&head.anchor_keys[a * D],
                    &anchors[((h * 2 + 0) * k + a) * D], D);
            ggml_fp32_to_bf16_row(&head.anchor_values[a * D],
                    &anchors[((h * 2 + 1) * k + a) * D], D);
        }
        for (int t = 0; t < S; t++) {
            anchor_of[(0 * n_heads + h) * S + t] = head.k_anchor_of[t];
            anchor_of[(1 * n_heads + h) * S + t] = head.v_anchor_of[t];
            gamma[(0 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.k_gamma[t]);
            gamma[(1 * n_heads + h) * S + t]     = ggml_fp32_to_fp16(head.v_gamma[t]);
            slot_of[(0 * n_heads + h) * S + t]   = head.k_slot_of[t];
            slot_of[(1 * n_heads + h) * S + t]   = head.v_slot_of[t];
        }
        for (int i = 0; i < head.n_K; i++) {
            memcpy(&k_codes[(size_t) (h * n_K_max + i) * cpr], &head.k_res_codes[i * cpr], cpr);
            k_scales[h * n_K_max + i] = head.k_res_scales[i];
        }
        for (int i = 0; i < head.n_V; i++) {
            memcpy(&v_codes[(size_t) (h * n_V_max + i) * cpr], &head.v_res_codes[i * cpr], cpr);
            v_scales[h * n_V_max + i] = head.v_res_scales[i];
        }
    }

    float signs[128];
    op_get_wht_signs(signs, D);

    const int n_embd = n_heads * D;

    // replicate the op's loop over heads/tokens (single thread: ith=0, nth=1)
    for (int64_t t = 0; t < S; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int side = 0; side < 2; side++) {
                const int a = anchor_of[(side * n_heads + h) * S + t];
                const float g = ggml_fp16_to_fp32(gamma[(side * n_heads + h) * S + t]);
                const int slot = slot_of[(side * n_heads + h) * S + t];
                const ggml_bf16_t * anchor_vec = &anchors[((h * 2 + side) * k + a) * D];

                float * out = side == 0 ? out_k : out_v;

                if (slot >= 0) {
                    const uint8_t * codes  = (side == 0 ? k_codes.data()  : v_codes.data())  + (h * (side == 0 ? n_K_max : n_V_max) + slot) * cpr;
                    const float   * scales = (side == 0 ? k_scales.data() : v_scales.data()) +  h * (side == 0 ? n_K_max : n_V_max) + slot;

                    float deq[128];
                    for (int d = 0; d < D; d++) {
                        const int idx = (codes[d / 4] >> ((d % 4) * 2)) & 0x3;
                        deq[d] = op_lloyd_centroids[idx];
                    }
                    op_wht_inverse(deq, D, signs);
                    const float scale = scales[0];
                    for (int d = 0; d < D; d++) {
                        out[t * n_embd + h * D + d] = g * ggml_bf16_to_fp32(anchor_vec[d]) + deq[d] * scale;
                    }
                } else {
                    for (int d = 0; d < D; d++) {
                        out[t * n_embd + h * D + d] = g * ggml_bf16_to_fp32(anchor_vec[d]);
                    }
                }
            }
        }
    }
}

static float mse(const float * a, const float * b, int n) {
    double sum = 0;
    for (int i = 0; i < n; i++) {
        double d = (double) a[i] - b[i];
        sum += d * d;
    }
    return (float) (sum / n);
}

static float max_abs(const float * a, const float * b, int n) {
    float m = 0;
    for (int i = 0; i < n; i++) m = std::max(m, fabsf(a[i] - b[i]));
    return m;
}

int main() {
    const int S = 33; // not a multiple of anything, exercises all paths
    const int D = 128;
    const int n_heads = 4;
    // NOTE: theta=0.1f (the original value here) makes anchor_kv_max_residuals()
    // round down to 0 residuals at this S, so the WHT/Lloyd-Max quantization and
    // residual-slot-lookup code paths in anchor_kv_decompress_head() / the
    // GGML_OP_ANCHOR_DECOMPRESS op never actually run - the test "passes" without
    // testing anything meaningful. theta=0.54f keeps S=33 (still an odd,
    // non-power-of-2 size) but pushes the retained-byte budget just past the
    // point where residuals get allocated - N=15 (n_K=7, n_V=8 for head 0),
    // safely below the 17 non-anchor positions available at this S/W so both
    // sides get a non-empty share (see the n_K/n_V check below).
    const float theta = 0.54f;

    // deterministic pseudo-random data
    std::vector<float> keys((size_t) n_heads * S * D);
    std::vector<float> values((size_t) n_heads * S * D);
    uint64_t rng = 12345;
    for (size_t i = 0; i < keys.size(); i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        keys[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        values[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
    }

    anchor_kv_params params;
    params.theta = theta;
    params.W = 16;
    params.k_frac = 16;
    params.rho = 0.5f;
    params.kappa = 8;

    anchor_kv_layer layer = anchor_kv_compress(
        keys.data(), values.data(), S, D, n_heads, params);

    printf("layer: %d heads, k=%d anchors, n_K=%d n_V=%d (head 0)\n",
           layer.n_heads, layer.heads[0].k, layer.heads[0].n_K, layer.heads[0].n_V);

    // Guard against this test silently degrading back into a trivial pass: if
    // n_K or n_V is 0 for any head, the residual/quantization code path below
    // is never exercised and a matching mse of 0 would prove nothing. Fail
    // loudly here instead of only relying on someone eyeballing the printed
    // n_K/n_V values.
    int total_K = 0;
    int total_V = 0;
    for (int h = 0; h < n_heads; h++) {
        total_K += layer.heads[h].n_K;
        total_V += layer.heads[h].n_V;
    }
    if (total_K == 0 || total_V == 0) {
        fprintf(stderr,
            "FAIL: total residuals are n_K=%d n_V=%d - residual/quantization "
            "path not exercised. Adjust the test parameters.\n", total_K, total_V);
        return 1;
    }

    // reference reconstruction
    std::vector<float> ref_k((size_t) n_heads * S * D);
    std::vector<float> ref_v((size_t) n_heads * S * D);
    for (int h = 0; h < n_heads; h++) {
        anchor_kv_decompress_head(layer.heads[h], &ref_k[(size_t) h * S * D], &ref_v[(size_t) h * S * D]);
    }

    // op replication: outputs head-major interleaved [n_heads][S][D] -> [t][h*D+d]
    std::vector<float> op_k((size_t) n_heads * S * D);
    std::vector<float> op_v((size_t) n_heads * S * D);
    op_decompress_layer(layer, n_heads, S, D, op_k.data(), op_v.data());

    // op outputs are laid out [t * n_embd + h * D + d]; reference is [h * S * D + t * D + d].
    // Re-map reference to op layout for comparison.
    std::vector<float> ref_k_op((size_t) n_heads * S * D);
    std::vector<float> ref_v_op((size_t) n_heads * S * D);
    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t < S; t++) {
            memcpy(&ref_k_op[(size_t) t * (n_heads * D) + h * D], &ref_k[(size_t) h * S * D + t * D], D * sizeof(float));
            memcpy(&ref_v_op[(size_t) t * (n_heads * D) + h * D], &ref_v[(size_t) h * S * D + t * D], D * sizeof(float));
        }
    }

    float mse_k = mse(op_k.data(), ref_k_op.data(), op_k.size());
    float mse_v = mse(op_v.data(), ref_v_op.data(), op_v.size());
    float max_k = max_abs(op_k.data(), ref_k_op.data(), op_k.size());
    float max_v = max_abs(op_v.data(), ref_v_op.data(), op_v.size());

    printf("K: mse=%.3e max_abs=%.3e\n", mse_k, max_k);
    printf("V: mse=%.3e max_abs=%.3e\n", mse_v, max_v);

    // also check reconstruction quality vs original (sanity: should be small-ish)
    float mse_orig_k = mse(op_k.data(), keys.data(), op_k.size());
    float mse_orig_v = mse(op_v.data(), values.data(), op_v.size());
    printf("orig: K mse=%.3e V mse=%.3e (lossy codec sanity)\n", mse_orig_k, mse_orig_v);

    // Runtime anchors and coefficients are bf16/f16, so compare within the
    // expected storage-rounding error rather than requiring f32 identity.
    const bool pass = mse_k < 1e-5f && mse_v < 1e-5f && max_k < 1e-2f && max_v < 1e-2f;
    printf(pass ? "PASS: op matches reference\n" : "FAIL: op diverges from reference\n");
    return pass ? 0 : 1;
}
