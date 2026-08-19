/*
 * AnchorKV CPU Decode Validation
 *
 * Tests the full pipeline: compress -> reconstruct tile-by-tile -> attention
 * -> compare against dense attention output.
 *
 * This validates that AnchorKV produces correct attention output when used
 * in the actual inference loop, which is the prerequisite for GPU kernel work.
 *
 * Build:
 *   g++ -O2 -std=c++17 -I../src -I../include -I../ggml/include \
 *       -o test-anchor-kv-decode test-anchor-kv-decode.cpp ../src/anchor-kv.cpp -lm
 *
 * Run:
 *   ./test-anchor-kv-decode --seq-len 3001 --head-dim 128 --theta 0.1
 */

#include "anchor-kv.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

/* ---------- Dense attention (reference) ---------- */

static void dense_attention(
    const float * Q,    /* [D] single query */
    const float * K,    /* [S * D] keys */
    const float * V,    /* [S * D] values */
    float * out,        /* [D] output */
    int S, int D
) {
    float scale = 1.0f / sqrtf((float)D);

    /* Compute attention scores: s_t = Q . K_t / sqrt(D) */
    std::vector<float> scores(S);
    for (int t = 0; t < S; t++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += Q[d] * K[t * D + d];
        }
        scores[t] = dot * scale;
    }

    /* Softmax */
    float max_s = *std::max_element(scores.begin(), scores.end());
    float sum_exp = 0.0f;
    for (int t = 0; t < S; t++) {
        scores[t] = expf(scores[t] - max_s);
        sum_exp += scores[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < S; t++) scores[t] *= inv_sum;

    /* Output: out = sum_t alpha_t * V_t */
    memset(out, 0, D * sizeof(float));
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            out[d] += scores[t] * V[t * D + d];
        }
    }
}

/* ---------- AnchorKV attention (tile-by-tile reconstruction) ---------- */

/*
 * This simulates what the GPU decode kernel would do:
 * For each tile of the sequence, reconstruct K/V from the compressed
 * representation, then compute attention scores and output.
 *
 * The tile-by-tile approach means we never materialize the full dense
 * K/V -- we reconstruct only what we need for each tile.
 */
static void anchorkv_attention(
    const float * Q,                        /* [D] single query */
    const anchor_kv_head & head,            /* compressed representation */
    float * out,                            /* [D] output */
    int D,
    int tile_size                           /* positions per tile */
) {
    float scale = 1.0f / sqrtf((float)D);
    int S = head.S;

    /* Build is_anchor set */
    std::vector<int> is_anchor(S, 0);
    for (int i = 0; i < head.k; i++) {
        is_anchor[head.anchor_positions[i]] = 1;
    }

    /* Build residual slot lookups */
    std::vector<int> k_slot(S, -1);
    std::vector<int> v_slot(S, -1);
    {
        int k_idx = 0, v_idx = 0;
        for (int t = 0; t < S; t++) {
            if (head.k_residual_mask[t / 64] & (1ULL << (t % 64))) {
                k_slot[t] = k_idx++;
            }
            if (head.v_residual_mask[t / 64] & (1ULL << (t % 64))) {
                v_slot[t] = v_idx++;
            }
        }
    }

    /* Process tile-by-tile */
    std::vector<float> tile_k(tile_size * D);
    std::vector<float> tile_v(tile_size * D);
    std::vector<float> tile_scores(tile_size);

    /* Accumulate output using log-sum-exp for numerical stability */
    float max_score = -1e30f;
    std::vector<float> all_scores(S);

    /* First pass: compute all scores to find max (for softmax stability) */
    for (int t0 = 0; t0 < S; t0 += tile_size) {
        int t_end = std::min(t0 + tile_size, S);
        int tile_len = t_end - t0;

        /* Reconstruct K for this tile */
        for (int i = 0; i < tile_len; i++) {
            int t = t0 + i;
            if (is_anchor[t]) {
                int a = head.k_anchor_of[t];
                memcpy(&tile_k[i * D], &head.anchor_keys[a * D], D * sizeof(float));
            } else {
                int a = head.k_anchor_of[t];
                for (int d = 0; d < D; d++) {
                    tile_k[i * D + d] = head.k_gamma[t] * head.anchor_keys[a * D + d];
                }
                if (k_slot[t] >= 0) {
                    float deq[128];
                    size_t cpr = (size_t)D / 4;
                    anchor_kv_dequantize_residual(
                        &head.k_res_codes[k_slot[t] * cpr],
                        D, head.k_res_scales[k_slot[t]],
                        deq
                    );
                    for (int d = 0; d < D; d++) {
                        tile_k[i * D + d] += deq[d];
                    }
                }
            }
        }

        /* Compute scores for this tile */
        for (int i = 0; i < tile_len; i++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += Q[d] * tile_k[i * D + d];
            }
            all_scores[t0 + i] = dot * scale;
        }
    }

    /* Softmax */
    max_score = *std::max_element(all_scores.begin(), all_scores.begin() + S);
    float sum_exp = 0.0f;
    for (int t = 0; t < S; t++) {
        all_scores[t] = expf(all_scores[t] - max_score);
        sum_exp += all_scores[t];
    }
    float inv_sum = 1.0f / sum_exp;
    for (int t = 0; t < S; t++) all_scores[t] *= inv_sum;

    /* Second pass: accumulate output from V tiles */
    memset(out, 0, D * sizeof(float));
    for (int t0 = 0; t0 < S; t0 += tile_size) {
        int t_end = std::min(t0 + tile_size, S);
        int tile_len = t_end - t0;

        /* Reconstruct V for this tile */
        for (int i = 0; i < tile_len; i++) {
            int t = t0 + i;
            if (is_anchor[t]) {
                int a = head.v_anchor_of[t];
                memcpy(&tile_v[i * D], &head.anchor_values[a * D], D * sizeof(float));
            } else {
                int a = head.v_anchor_of[t];
                for (int d = 0; d < D; d++) {
                    tile_v[i * D + d] = head.v_gamma[t] * head.anchor_values[a * D + d];
                }
                if (v_slot[t] >= 0) {
                    float deq[128];
                    size_t cpr = (size_t)D / 4;
                    anchor_kv_dequantize_residual(
                        &head.v_res_codes[v_slot[t] * cpr],
                        D, head.v_res_scales[v_slot[t]],
                        deq
                    );
                    for (int d = 0; d < D; d++) {
                        tile_v[i * D + d] += deq[d];
                    }
                }
            }
        }

        /* Accumulate: out += sum_i alpha_{t0+i} * V_{t0+i} */
        for (int i = 0; i < tile_len; i++) {
            float alpha = all_scores[t0 + i];
            for (int d = 0; d < D; d++) {
                out[d] += alpha * tile_v[i * D + d];
            }
        }
    }
}

/* ---------- Metrics ---------- */

static float cosine_sim(const float * a, const float * b, int n) {
    float dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    return dot / (sqrtf(na) * sqrtf(nb) + 1e-10f);
}

static float max_abs_error(const float * a, const float * b, int n) {
    float mx = 0;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/* ---------- Main ---------- */

int main(int argc, char ** argv) {
    int S = 3001;
    int D = 128;
    float theta = 0.1f;
    int n_queries = 10;
    int tile_size = 64;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) S = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) D = atoi(argv[++i]);
        else if (strcmp(argv[i], "--theta") == 0 && i + 1 < argc) theta = atof(argv[++i]);
        else if (strcmp(argv[i], "--tile-size") == 0 && i + 1 < argc) tile_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--n-queries") == 0 && i + 1 < argc) n_queries = atoi(argv[++i]);
    }

    printf("AnchorKV CPU Decode Validation\n");
    printf("==============================\n");
    printf("S=%d D=%d theta=%.3f tile=%d queries=%d\n\n", S, D, theta, tile_size, n_queries);

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

    printf("Compressing...\n");
    anchor_kv_layer layer = anchor_kv_compress(
        keys.data(), values.data(), S, D, 1, params
    );
    const anchor_kv_head & head = layer.heads[0];
    printf("  Anchors: %d, K_res: %d, V_res: %d\n\n", head.k, head.n_K, head.n_V);

    /* Generate random queries */
    std::vector<float> queries(n_queries * D);
    {
        uint64_t rng = 42;
        for (int i = 0; i < n_queries * D; i++) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            queries[i] = (float)((int32_t)(rng >> 32)) / (float)(1ULL << 31);
        }
        /* Normalize each query */
        for (int q = 0; q < n_queries; q++) {
            float norm = 0;
            for (int d = 0; d < D; d++) norm += queries[q*D+d] * queries[q*D+d];
            norm = sqrtf(norm);
            for (int d = 0; d < D; d++) queries[q*D+d] /= norm;
        }
    }

    /* Run attention and compare */
    float total_cos = 0, total_mae = 0;
    float worst_cos = 1.0f, worst_mae = 0;

    printf("Query | Dense vs AnchorKV\n");
    printf("------+------------------\n");

    for (int q = 0; q < n_queries; q++) {
        float * Q = &queries[q * D];
        std::vector<float> out_dense(D);
        std::vector<float> out_anchor(D);

        dense_attention(Q, keys.data(), values.data(), out_dense.data(), S, D);
        anchorkv_attention(Q, head, out_anchor.data(), D, tile_size);

        float cos = cosine_sim(out_dense.data(), out_anchor.data(), D);
        float mae = max_abs_error(out_dense.data(), out_anchor.data(), D);

        printf("  %2d  | cos=%.6f  mae=%.6f\n", q, cos, mae);

        total_cos += cos;
        total_mae += mae;
        if (cos < worst_cos) worst_cos = cos;
        if (mae > worst_mae) worst_mae = mae;
    }

    printf("\nSummary\n");
    printf("=======\n");
    printf("Average cosine: %.6f\n", total_cos / n_queries);
    printf("Average MAE:    %.6f\n", total_mae / n_queries);
    printf("Worst cosine:   %.6f\n", worst_cos);
    printf("Worst MAE:      %.6f\n", worst_mae);

    /* Pass/fail */
    int pass = 1;
    if (worst_cos < 0.99f) {
        printf("\nFAIL: worst cosine %.6f < 0.99\n", worst_cos);
        pass = 0;
    }
    if (worst_mae > 0.01f) {
        printf("\nFAIL: worst MAE %.6f > 0.01\n", worst_mae);
        pass = 0;
    }
    if (pass) printf("\nPASS\n");

    return pass ? 0 : 1;
}
