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

    // The paper's long-context budget must not activate while the cache
    // contains only a few positions beyond the exact recency window.
    {
        anchor_kv_params short_p;
        if (!anchor_kv_budget_ready(ANCHOR_KV_W, D, short_p) ||
            anchor_kv_budget_ready(48, D, short_p)) {
            printf("FAIL: short-cache budget gate is wrong\\n");
            return 1;
        }
        if (!anchor_kv_budget_ready(16384, D, short_p)) {
            printf("FAIL: viable long-context budget was rejected\\n");
            return 1;
        }
    }

    // V-only
    {
        p.compress_k = false;
        p.compress_v = true;
        anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, p);
        const anchor_kv_head & h = layer.heads[0];
        int n_v_actual = 0;
        bool v_storage = false;
        for (const anchor_kv_head & head : layer.heads) {
            if (head.n_K != 0 || !head.anchor_keys.empty()) {
                printf("FAIL: V-only stored K data\n");
                return 1;
            }
            n_v_actual += head.n_V;
            v_storage |= !head.anchor_values.empty();
        }
        printf("V-only: k=%d head0_n_V=%d total_n_V=%d\n", h.k, h.n_V, n_v_actual);
        if (n_v_actual == 0 || !v_storage) {
            printf("FAIL: V-only flags/storage wrong\n");
            return 1;
        }
        // per-side layer budget, not one combined budget per head
        const int n_v_side  = anchor_kv_max_residuals_side_layer(S, D, p.W, p.theta, true, n_heads, p.k_frac);
        const int n_combined = anchor_kv_max_residuals_layer(S, D, p.W, p.theta, n_heads, p.k_frac);
        printf("V-only budget: total_n_V=%d (side) vs combined=%d\n", n_v_actual, n_combined);
        if (n_v_actual != n_v_side) {
            printf("FAIL: V-only total n_V=%d does not match layer budget %d\n", n_v_actual, n_v_side);
            return 1;
        }
        if (n_v_actual >= n_combined) {
            printf("FAIL: V-only still uses the combined budget (n_V=%d >= %d)\n", n_v_actual, n_combined);
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
        int n_k_actual = 0;
        bool k_storage = false;
        for (const anchor_kv_head & head : layer.heads) {
            if (head.n_V != 0 || !head.anchor_values.empty()) {
                printf("FAIL: K-only stored V data\n");
                return 1;
            }
            n_k_actual += head.n_K;
            k_storage |= !head.anchor_keys.empty();
        }
        printf("K-only: k=%d head0_n_K=%d total_n_K=%d\n", h.k, h.n_K, n_k_actual);
        if (n_k_actual == 0 || !k_storage) {
            printf("FAIL: K-only flags/storage wrong\n");
            return 1;
        }
        const int n_k_side = anchor_kv_max_residuals_side_layer(S, D, p.W, p.theta, false, n_heads, p.k_frac);
        printf("K-only budget: total_n_K=%d (side)\n", n_k_side);
        if (n_k_actual != n_k_side) {
            printf("FAIL: K-only total n_K=%d does not match layer budget %d\n", n_k_actual, n_k_side);
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

    // Asymmetric K/V thetas (both sides compressed, different ratios)
    {
        p.compress_k = true;
        p.compress_v = true;
        p.theta = 0.2f;
        p.theta_k = 0.15f;
        p.theta_v = 0.18f;
        anchor_kv_layer layer = anchor_kv_compress(keys.data(), values.data(), S, D, n_heads, p);
        const anchor_kv_head & h = layer.heads[0];
        int n_k_actual = 0;
        int n_v_actual = 0;
        for (const anchor_kv_head & head : layer.heads) {
            n_k_actual += head.n_K;
            n_v_actual += head.n_V;
        }
        const int n_k_side = anchor_kv_max_residuals_side_layer(S, D, p.W, p.theta_k, false, n_heads, p.k_frac);
        const int n_v_side = anchor_kv_max_residuals_side_layer(S, D, p.W, p.theta_v, true, n_heads, p.k_frac);
        printf("Asym theta: k=%d total_n_K=%d (budget %d) total_n_V=%d (budget %d)\n",
               h.k, n_k_actual, n_k_side, n_v_actual, n_v_side);
        if (n_k_actual != n_k_side || n_v_actual != n_v_side) {
            printf("FAIL: asymmetric theta budget mismatch (n_K=%d vs %d, n_V=%d vs %d)\n",
                   n_k_actual, n_k_side, n_v_actual, n_v_side);
            return 1;
        }
        if (n_k_side >= n_v_side) {
            printf("FAIL: theta_k=0.15 should give fewer K residuals than theta_v=0.18 (%d >= %d)\n",
                   n_k_side, n_v_side);
            return 1;
        }
    }

    printf("PASS: asymmetric V-only/K-only roundtrip\n");
    return 0;
}
