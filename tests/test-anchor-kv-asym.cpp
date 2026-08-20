/*
 * Test: AnchorKV asymmetric (V-only / K-only) compression round-trip.
 *
 * Build:
 *   g++ -O2 -std=c++17 -I../src -I../include -I../ggml/include \
 *       -o test-anchor-kv-asym test-anchor-kv-asym.cpp ../src/anchor-kv.cpp -lm
 *
 * Run:
 *   ./test-anchor-kv-asym
 *
 * Verifies that when only one side is compressed the other side's storage is
 * left empty and the reference decompressor does not touch the uncompressed
 * side's output buffer.
 */

#include "anchor-kv.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return dot / (sqrtf(na)*sqrtf(nb) + 1e-10f);
}

int main() {
    const int S = 256;
    const int D = 128;
    const int n_heads = 2;

    std::vector<float> keys((size_t) n_heads * S * D);
    std::vector<float> values((size_t) n_heads * S * D);
    uint64_t rng = 12345;
    for (size_t i = 0; i < keys.size(); i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        keys[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        values[i] = ((float) ((rng >> 33) & 0xffff) / 32768.0f - 1.0f) * 2.0f;
    }

    anchor_kv_params p;
    // theta chosen so the per-side budget lands below the non-anchor position
    // count (P = S - W), otherwise the selection loop caps n_K/n_V at P and the
    // budget formula is never the binding constraint.
    p.theta = 0.2f;
    p.W = 16;
    p.k_frac = 16;
    p.rho = 0.5f;
    p.kappa = 8;

    // V-only
    {
        p.compress_k = false;
        p.compress_v = true;
        anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, p);
        const anchor_kv_head & h = layer.heads[0];
        printf("V-only: k=%d n_K=%d n_V=%d anchor_keys=%zu anchor_values=%zu\n",
               h.k, h.n_K, h.n_V, h.anchor_keys.size(), h.anchor_values.size());
        if (h.n_K != 0 || h.anchor_keys.size() != 0 || h.n_V == 0 || h.anchor_values.size() == 0) {
            printf("FAIL: V-only flags/storage wrong\n");
            return 1;
        }
        // per-side budget, not the combined two-side budget
        const int n_v_side  = anchor_kv_max_residuals_side(S, D, p.W, p.theta, true);
        const int n_combined = anchor_kv_max_residuals(S, D, p.W, p.theta);
        printf("V-only budget: n_V=%d (side) vs combined=%d\n", n_v_side, n_combined);
        if (h.n_V != n_v_side) {
            printf("FAIL: V-only n_V=%d does not match per-side budget %d\n", h.n_V, n_v_side);
            return 1;
        }
        if (h.n_V >= n_combined) {
            printf("FAIL: V-only still uses the combined budget (n_V=%d >= %d)\n", h.n_V, n_combined);
            return 1;
        }
        std::vector<float> out_k((size_t) S * D, -123.0f);
        std::vector<float> out_v((size_t) S * D, 0.0f);
        anchor_kv_decompress_head(h, out_k.data(), out_v.data());
        for (size_t i = 0; i < out_k.size(); i++) {
            if (out_k[i] != -123.0f) { printf("FAIL: V-only wrote out_k\n"); return 1; }
        }
        float cos = cosine_sim(values.data(), out_v.data(), (int) out_v.size());
        printf("V-only recon cos=%.6f\n", cos);
        if (cos < 0.5f) { printf("FAIL: V-only recon too poor\n"); return 1; }
    }

    // K-only
    {
        p.compress_k = true;
        p.compress_v = false;
        anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, p);
        const anchor_kv_head & h = layer.heads[0];
        printf("K-only: k=%d n_K=%d n_V=%d anchor_keys=%zu anchor_values=%zu\n",
               h.k, h.n_K, h.n_V, h.anchor_keys.size(), h.anchor_values.size());
        if (h.n_V != 0 || h.anchor_values.size() != 0 || h.n_K == 0 || h.anchor_keys.size() == 0) {
            printf("FAIL: K-only flags/storage wrong\n");
            return 1;
        }
        const int n_k_side = anchor_kv_max_residuals_side(S, D, p.W, p.theta, false);
        printf("K-only budget: n_K=%d (side)\n", n_k_side);
        if (h.n_K != n_k_side) {
            printf("FAIL: K-only n_K=%d does not match per-side budget %d\n", h.n_K, n_k_side);
            return 1;
        }
        std::vector<float> out_k((size_t) S * D, 0.0f);
        std::vector<float> out_v((size_t) S * D, -123.0f);
        anchor_kv_decompress_head(h, out_k.data(), out_v.data());
        for (size_t i = 0; i < out_v.size(); i++) {
            if (out_v[i] != -123.0f) { printf("FAIL: K-only wrote out_v\n"); return 1; }
        }
        float cos = cosine_sim(keys.data(), out_k.data(), (int) out_k.size());
        printf("K-only recon cos=%.6f\n", cos);
        if (cos < 0.5f) { printf("FAIL: K-only recon too poor\n"); return 1; }
    }

    printf("PASS: asymmetric V-only/K-only roundtrip\n");
    return 0;
}
