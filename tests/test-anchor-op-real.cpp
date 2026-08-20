/*
 * Test: GGML_OP_ANCHOR_DECOMPRESS math vs reference, exact runtime params.
 *
 * Build:
 *   g++ -O2 -std=c++17 -I../src -I../include -I../ggml/include \
 *       -o test-anchor-op-real test-anchor-op-real.cpp ../src/anchor-kv.cpp -lm
 *
 * Run (needs /tmp/anchor-kv-k.bin, /tmp/anchor-kv-v.bin from extract-kv-cache.py):
 *   ./test-anchor-op-real [S]
 */

#include "anchor-kv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

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

static void op_decompress_layer(
        const anchor_kv_layer & layer,
        int n_heads, int S, int D,
        float * out_k, float * out_v) {
    int k = layer.heads[0].k;

    std::vector<float> anchors(n_heads * 2 * k * D, 0.0f);
    std::vector<int32_t> anchor_of(2 * n_heads * S, 0);
    std::vector<float> gamma(2 * n_heads * S, 0.0f);
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
            memcpy(&anchors[((h * 2 + 0) * k + a) * D], &head.anchor_keys[a * D],  D * sizeof(float));
            memcpy(&anchors[((h * 2 + 1) * k + a) * D], &head.anchor_values[a * D], D * sizeof(float));
        }
        for (int t = 0; t < S; t++) {
            anchor_of[(0 * n_heads + h) * S + t] = head.k_anchor_of[t];
            anchor_of[(1 * n_heads + h) * S + t] = head.v_anchor_of[t];
            gamma[(0 * n_heads + h) * S + t]     = head.k_gamma[t];
            gamma[(1 * n_heads + h) * S + t]     = head.v_gamma[t];
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

    for (int64_t t = 0; t < S; t++) {
        for (int h = 0; h < n_heads; h++) {
            for (int side = 0; side < 2; side++) {
                const int a = anchor_of[(side * n_heads + h) * S + t];
                const float g = gamma[(side * n_heads + h) * S + t];
                const int slot = slot_of[(side * n_heads + h) * S + t];
                const float * anchor_vec = &anchors[((h * 2 + side) * k + a) * D];

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
                        out[t * n_embd + h * D + d] = g * anchor_vec[d] + deq[d] * scale;
                    }
                } else {
                    for (int d = 0; d < D; d++) {
                        out[t * n_embd + h * D + d] = g * anchor_vec[d];
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

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na)*sqrtf(nb) + 1e-10f);
}

int main(int argc, char ** argv) {
    // NOTE: the previous default (S=32) - and S=512, also previously used for
    // manual runs - both make anchor_kv_max_residuals() round down to 0
    // residuals at the runtime defaults (theta=0.1f, W=ANCHOR_KV_W=32), so the
    // WHT/Lloyd-Max quantization and residual-slot-lookup code paths never ran
    // and the test passed without exercising anything meaningful. S=4096 is
    // the smallest of the previously-tried sizes that actually allocates
    // residuals (n_K/n_V > 0, see the guard below), so it is now the default
    // for unattended/standing runs. Pass an explicit S on the command line to
    // still exercise other sizes (e.g. S=32 or S=512) for adhoc/edge-case
    // checks - those will now fail loudly instead of silently passing, which
    // is the point: it forces you to notice they aren't testing the residual
    // path rather than rediscovering that fact by accident later.
    int S = argc > 1 ? atoi(argv[1]) : 4096;
    const int D = 128;
    const int n_heads = 1;

    // exact runtime defaults (llama-kv-cache.cpp sets only theta)
    anchor_kv_params params;
    params.theta = 0.1f;

    // load real KV
    std::vector<float> keys((size_t) S * D);
    std::vector<float> values((size_t) S * D);
    FILE * fk = fopen("/tmp/anchor-kv-k.bin", "rb");
    FILE * fv = fopen("/tmp/anchor-kv-v.bin", "rb");
    if (!fk || !fv) { fprintf(stderr, "missing KV files\n"); return 1; }
    fread(keys.data(), sizeof(float), (size_t) S * D, fk);
    fread(values.data(), sizeof(float), (size_t) S * D, fv);
    fclose(fk); fclose(fv);

    anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, params);

    printf("S=%d: k=%d n_K=%d n_V=%d\n", S, layer.heads[0].k,
           layer.heads[0].n_K, layer.heads[0].n_V);

    // Guard against this test silently degrading back into a trivial pass: if
    // n_K or n_V is 0, the residual/quantization code path below is never
    // exercised and a matching mse of 0 would prove nothing. Fail loudly here
    // instead of relying on someone eyeballing the printed n_K/n_V values (this
    // is exactly how the original slot-index bug went unnoticed at S=32/512).
    for (int h = 0; h < n_heads; h++) {
        if (layer.heads[h].n_K == 0 || layer.heads[h].n_V == 0) {
            fprintf(stderr,
                "FAIL: head %d has n_K=%d n_V=%d at S=%d - residual/quantization "
                "path not exercised (theta/W/S combination rounds the residual "
                "budget to 0; see anchor_kv_max_residuals()). Use a larger S "
                "(S=4096 is known to work at the runtime-default theta/W) or "
                "adjust the test parameters so residuals are actually "
                "allocated.\n",
                h, layer.heads[h].n_K, layer.heads[h].n_V, S);
            return 1;
        }
    }

    // reference reconstruction (head-major: [h][t][d])
    std::vector<float> ref_k((size_t) n_heads * S * D);
    std::vector<float> ref_v((size_t) n_heads * S * D);
    for (int h = 0; h < n_heads; h++) {
        anchor_kv_decompress_head(layer.heads[h], &ref_k[(size_t) h * S * D], &ref_v[(size_t) h * S * D]);
    }

    // op replication: [t][h*D+d]
    std::vector<float> op_k((size_t) n_heads * S * D);
    std::vector<float> op_v((size_t) n_heads * S * D);
    op_decompress_layer(layer, n_heads, S, D, op_k.data(), op_v.data());

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
    float cos_k = cosine_sim(op_k.data(), ref_k_op.data(), op_k.size());
    float cos_v = cosine_sim(op_v.data(), ref_v_op.data(), op_v.size());

    printf("op vs ref: K mse=%.3e max=%.3e cos=%.6f | V mse=%.3e max=%.3e cos=%.6f\n",
           mse_k, max_k, cos_k, mse_v, max_v, cos_v);

    // reconstruction quality vs original
    float mse_orig_k = mse(op_k.data(), keys.data(), op_k.size());
    float mse_orig_v = mse(op_v.data(), values.data(), op_v.size());
    float cos_orig_k = cosine_sim(op_k.data(), keys.data(), op_k.size());
    float cos_orig_v = cosine_sim(op_v.data(), values.data(), op_v.size());
    printf("op vs orig: K mse=%.4f cos=%.6f | V mse=%.4f cos=%.6f\n",
           mse_orig_k, cos_orig_k, mse_orig_v, cos_orig_v);

    const bool pass = mse_k < 1e-6f && mse_v < 1e-6f;
    printf(pass ? "PASS: op matches reference\n" : "FAIL: op diverges from reference\n");
    return pass ? 0 : 1;
}
