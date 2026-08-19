/*
 * Test AnchorKV compression pass against real model KV data.
 *
 * Reads binary KV files from extract-kv-cache.py, compresses with the
 * production compression pass, decompresses, and measures quality.
 *
 * Build:
 *   g++ -O2 -std=c++17 -I../src -I../include -I../ggml/include \
 *       -o test-anchor-kv-compress test-anchor-kv-compress.cpp ../src/anchor-kv.cpp -lm
 *
 * Run:
 *   ./test-anchor-kv-compress --seq-len 3001 --head-dim 128 --theta 0.1
 */

#include "anchor-kv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

static float mse(const float * a, const float * b, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum / n;
}

int main(int argc, char ** argv) {
    int S = 3001;
    int D = 128;
    float theta = 0.1f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) S = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) D = atoi(argv[++i]);
        else if (strcmp(argv[i], "--theta") == 0 && i + 1 < argc) theta = atof(argv[++i]);
    }

    printf("AnchorKV Compression Pass Test\n");
    printf("==============================\n");
    printf("S=%d D=%d theta=%.3f\n\n", S, D, theta);

    /* Load KV data */
    std::vector<float> keys(S * D);
    std::vector<float> values(S * D);

    FILE * fk = fopen("/tmp/anchor-kv-k.bin", "rb");
    FILE * fv = fopen("/tmp/anchor-kv-v.bin", "rb");
    if (!fk || !fv) {
        fprintf(stderr, "Error: cannot open KV files. Run extract-kv-cache.py first.\n");
        return 1;
    }
    size_t nk = fread(keys.data(), sizeof(float), S * D, fk);
    size_t nv = fread(values.data(), sizeof(float), S * D, fv);
    fclose(fk); fclose(fv);

    if (nk != (size_t)(S * D) || nv != (size_t)(S * D)) {
        fprintf(stderr, "Error: file size mismatch\n");
        return 1;
    }
    printf("Loaded %d x %d KV data\n", S, D);

    /* Compress */
    anchor_kv_params params;
    params.theta = theta;

    printf("Compressing (theta=%.3f, W=%d, k_frac=%d)...\n", params.theta, params.W, params.k_frac);
    anchor_kv_layer layer = anchor_kv_compress(
        keys.data(), values.data(), S, D, 1, params
    );

    const anchor_kv_head & head = layer.heads[0];
    printf("  Anchors: %d\n", head.k);
    printf("  K residuals: %d, V residuals: %d\n", head.n_K, head.n_V);

    /* Debug: show anchor positions and first few values */
    printf("  Anchor positions: ");
    for (int i = 0; i < std::min(head.k, 10); i++) printf("%d ", head.anchor_positions[i]);
    printf("...\n");

    /* Check anchor_of for anchor positions */
    printf("  v_anchor_of for anchor positions: ");
    for (int i = 0; i < std::min(head.k, 10); i++) {
        int t = head.anchor_positions[i];
        printf("(%d->%d) ", t, head.v_anchor_of[t]);
    }
    printf("\n");
    printf("  k_anchor_of for anchor positions: ");
    for (int i = 0; i < std::min(head.k, 10); i++) {
        int t = head.anchor_positions[i];
        printf("(%d->%d) ", t, head.k_anchor_of[t]);
    }
    printf("\n");
    printf("  Last 5 anchor positions: ");
    for (int i = std::max(0, head.k - 5); i < head.k; i++) printf("%d ", head.anchor_positions[i]);
    printf("\n");

    /* Check anchor_values norm */
    float anch_norm = 0;
    for (int i = 0; i < D; i++) anch_norm += head.anchor_values[i] * head.anchor_values[i];
    printf("  anchor_values[0] norm: %.4f\n", sqrtf(anch_norm));
    float orig_norm = 0;
    int t0 = head.anchor_positions[0];
    for (int d = 0; d < D; d++) orig_norm += values[t0 * D + d] * values[t0 * D + d];
    printf("  orig values at pos %d norm: %.4f\n", t0, sqrtf(orig_norm));

    /* Decompress */
    std::vector<float> recon_k(S * D);
    std::vector<float> recon_v(S * D);
    anchor_kv_decompress_head(head, recon_k.data(), recon_v.data());

    /* Measure quality */
    float cos_k = cosine_sim(keys.data(), recon_k.data(), S * D);
    float cos_v = cosine_sim(values.data(), recon_v.data(), S * D);
    float mse_k = mse(keys.data(), recon_k.data(), S * D);
    float mse_v = mse(values.data(), recon_v.data(), S * D);

    printf("\nResults\n");
    printf("=======\n");
    printf("Key   MSE: %.8f  Cosine: %.6f\n", mse_k, cos_k);
    printf("Value MSE: %.8f  Cosine: %.6f\n", mse_v, cos_v);

    /* Debug: trace a single non-anchor position */
    {
        int t = 500;
        while (t < S && head.v_anchor_of[t] == t) t++;  /* skip anchors */
        int a_v = head.v_anchor_of[t];
        float proj_cos = 0, po_r = 0, po_o = 0;
        for (int d = 0; d < D; d++) {
            float proj = head.v_gamma[t] * head.anchor_values[a_v * D + d];
            proj_cos += proj * values[t * D + d];
            po_r += proj * proj;
            po_o += values[t * D + d] * values[t * D + d];
        }
        float rc = 0, rn = 0, ro = 0;
        for (int d = 0; d < D; d++) {
            rc += recon_v[t * D + d] * values[t * D + d];
            rn += recon_v[t * D + d] * recon_v[t * D + d];
            ro += values[t * D + d] * values[t * D + d];
        }
        printf("\n[debug] t=%d v_anchor=%d v_gamma=%.4f\n", t, a_v, head.v_gamma[t]);
        printf("  V proj-only cos: %.6f  V recon cos: %.6f\n",
               proj_cos / (sqrtf(po_o) * sqrtf(po_r) + 1e-10f),
               rc / (sqrtf(ro) * sqrtf(rn) + 1e-10f));
        printf("  orig: %.6f %.6f %.6f  (norm=%.4f)\n", values[t*D], values[t*D+1], values[t*D+2],
               sqrtf(ro));
        printf("  recon: %.6f %.6f %.6f  (norm=%.4f)\n", recon_v[t*D], recon_v[t*D+1], recon_v[t*D+2],
               sqrtf(rn));
        printf("  gamma=%.6f\n", head.v_gamma[t]);

        /* Check: is the scale wrong? */
        float scale = sqrtf(ro) / (sqrtf(rn) + 1e-10f);
        printf("  orig_norm/recon_norm = %.6f\n", scale);
    }

    /* Also check: average reconstruction quality across non-anchor positions */
    {
        float sum_cos = 0; int cnt = 0;
        for (int t = 0; t < S; t++) {
            if (head.v_anchor_of[t] == t) continue;  /* skip anchors */
            float dot = 0, nr = 0, no = 0;
            for (int d = 0; d < D; d++) {
                dot += recon_v[t * D + d] * values[t * D + d];
                nr += recon_v[t * D + d] * recon_v[t * D + d];
                no += values[t * D + d] * values[t * D + d];
            }
            sum_cos += dot / (sqrtf(no) * sqrtf(nr) + 1e-10f);
            cnt++;
        }
        printf("  Non-anchor avg V cosine: %.6f (count=%d)\n", sum_cos / cnt, cnt);

    /* Check anchor positions */
    {
        float sum_cos = 0; int cnt = 0;
        for (int i = 0; i < head.k; i++) {
            int t = head.anchor_positions[i];
            float dot = 0, nr = 0, no = 0;
            for (int d = 0; d < D; d++) {
                dot += recon_v[t * D + d] * values[t * D + d];
                nr += recon_v[t * D + d] * recon_v[t * D + d];
                no += values[t * D + d] * values[t * D + d];
            }
            float c = dot / (sqrtf(no) * sqrtf(nr) + 1e-10f);
            sum_cos += c;
            cnt++;
            if (c < 0.99f) {
                printf("  [!] anchor t=%d (idx=%d) cos=%.6f orig_norm=%.4f recon_norm=%.4f\n",
                       t, i, c, sqrtf(no), sqrtf(nr));
                printf("      orig: %.4f %.4f %.4f\n", values[t*D], values[t*D+1], values[t*D+2]);
                printf("      recon: %.4f %.4f %.4f\n", recon_v[t*D], recon_v[t*D+1], recon_v[t*D+2]);
            }
        }
        printf("  Anchor avg V cosine: %.6f (count=%d)\n", sum_cos / cnt, cnt);
    }
    }

    /* Per-head stats */
    float k_norm_orig = 0, k_norm_recon = 0;
    for (int i = 0; i < S * D; i++) {
        k_norm_orig += keys[i] * keys[i];
        k_norm_recon += recon_k[i] * recon_k[i];
    }
    printf("K norm ratio: %.4f\n", sqrtf(k_norm_recon) / sqrtf(k_norm_orig));

    /* Pass/fail */
    int pass = 1;
    if (cos_k < 0.99f) { printf("\nFAIL: K cosine %.6f < 0.99\n", cos_k); pass = 0; }
    if (cos_v < 0.99f) { printf("\nFAIL: V cosine %.6f < 0.99\n", cos_v); pass = 0; }
    if (pass) printf("\nPASS\n");

    return pass ? 0 : 1;
}
