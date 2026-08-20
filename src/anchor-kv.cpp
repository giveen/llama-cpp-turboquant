/*
 * AnchorKV: CPU compression pass
 *
 * Implements the full Algorithm 1 from arXiv:2608.02901v1.
 * Takes dense K/V cache tensors and produces the compressed
 * anchor-residual representation.
 */

#include "anchor-kv.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <numeric>

/* ---------- Lloyd-Max 4-level codebook for unit Gaussian ---------- */

static const float LLOYD_CENTROIDS_2BIT[4] = {
    -1.510138f, -0.452823f, 0.452823f, 1.510138f
};

static const float LLOYD_THRESHOLDS_2BIT[3] = {
    -0.981627f, 0.0f, 0.981627f
};

static int nearest_centroid_2bit(float val) {
    if (val < LLOYD_THRESHOLDS_2BIT[0]) return 0;
    if (val < LLOYD_THRESHOLDS_2BIT[1]) return 1;
    if (val < LLOYD_THRESHOLDS_2BIT[2]) return 2;
    return 3;
}

/* ---------- WHT ---------- */

void anchor_kv_get_wht_signs(float * signs, int d, uint32_t seed) {
    uint64_t state = seed;
    for (int i = 0; i < d; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        signs[i] = ((state >> 11) & 1) ? 1.0f : -1.0f;
    }
}

static void anchor_kv_wht_forward(float * x, int d, const float * signs) {
    for (int i = 0; i < d; i++) x[i] *= signs[i];
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= inv_sqrt_d;
}

static void anchor_kv_wht_inverse(float * x, int d, const float * signs) {
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
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

/* ---------- Residual quantization: WHT + absmax + Lloyd-Max ---------- */

static float quantize_residual_2bit(
    const float * residual, int d,
    const float * signs,
    uint8_t * codes_out   /* [d/4] packed output */
) {
    float buf[128];
    assert(d <= 128);
    memcpy(buf, residual, d * sizeof(float));

    anchor_kv_wht_forward(buf, d, signs);

    float absmax = 0.0f;
    for (int i = 0; i < d; i++) {
        float a = fabsf(buf[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    float inv_scale = 1.0f / absmax;
    for (int i = 0; i < d; i++) buf[i] *= inv_scale;

    size_t n_codes = (size_t)d / 4;
    memset(codes_out, 0, n_codes);
    for (int i = 0; i < d; i++) {
        int idx = nearest_centroid_2bit(buf[i]);
        codes_out[i / 4] |= (idx & 0x3) << ((i % 4) * 2);
    }

    return absmax;
}

static void dequantize_residual_2bit(
    const uint8_t * codes, int d, float scale,
    const float * signs,
    float * out
) {
    for (int i = 0; i < d; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        out[i] = LLOYD_CENTROIDS_2BIT[idx];
    }
    anchor_kv_wht_inverse(out, d, signs);
    for (int i = 0; i < d; i++) out[i] *= scale;
}

/* ---------- Byte budget (Eq. 9, Table 1) ---------- */

int anchor_kv_max_residuals(int S, int D, int W, float theta) {
    int k = S / ANCHOR_KV_K_FRAC;
    if (k < W) k = W;
    int P = S - W;

    /* Uncompressed: 2 sides * S * D * 2 bytes (bf16) */
    float Mfull = 2.0f * S * D * 2.0f;

    /* Base metadata (Eq. 13, single head) */
    int ba = 4;    /* int32 anchor index */
    int bgamma = 4; /* fp32 coefficient */
    float Mbase = 4.0f * k * D           /* anchor keys + values (fp32 for CPU ref) */
                + 8.0f * (k - W)         /* anchor positions (int32) */
                + 2.0f * P * (ba + bgamma) /* per-token index + coefficient x2 sides */
                + 24.0f * ((P + 63) / 64)  /* residual masks */
                + 8.0f + 4.0f * P;         /* head offsets + position ids */

    float cK = D / 4.0f + 4.0f;  /* key residual: codes + fp32 scale */
    float cV = D / 4.0f + 5.0f;  /* value residual: codes + fp32 scale + uint16 pos */

    float budget = theta * Mfull - Mbase;
    if (budget <= 0) return 0;

    float N = floorf(budget / ((cK + cV) / 2.0f));
    return (int)N;
}

/* ---------- Utility scoring (Eq. 6) ---------- */

struct utility_scores_t {
    std::vector<float> uK;
    std::vector<float> uV;
};

static utility_scores_t score_utilities(
    const float * residual_K, const float * residual_V,
    const float * values,
    const float * gamma_K, const float * gamma_V,
    const int * anchor_of_K, const int * anchor_of_V,
    const int * is_anchor,
    int S, int D
) {
    utility_scores_t util;
    util.uK.assign(S, 0.0f);
    util.uV.assign(S, 0.0f);

    /* Compute attention output y = (1/S) * sum_t V_t */
    std::vector<float> y(D, 0.0f);
    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            y[d] += values[t * D + d] / S;
        }
    }

    for (int t = 0; t < S; t++) {
        if (is_anchor[t]) continue;

        float alpha_t = 1.0f / S;

        /* ||V_t - y||^2 */
        float v_minus_y_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = values[t * D + d] - y[d];
            v_minus_y_sq += diff * diff;
        }

        /* Key utility: alpha_t^2 * ||r_t^K||^2 / D * ||V_t - y||^2 */
        float rK_norm_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            rK_norm_sq += residual_K[t * D + d] * residual_K[t * D + d];
        }
        util.uK[t] = alpha_t * alpha_t * (rK_norm_sq / D) * v_minus_y_sq;

        /* Value utility: alpha_t^2 * ||r_t^V||^2 */
        float rV_norm_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            rV_norm_sq += residual_V[t * D + d] * residual_V[t * D + d];
        }
        util.uV[t] = alpha_t * alpha_t * rV_norm_sq;
    }

    return util;
}

void anchor_kv_dequantize_residual(
    const uint8_t * codes, int D, float scale, float * out
) {
    std::vector<float> signs(D);
    anchor_kv_get_wht_signs(signs.data(), D, ANCHOR_KV_SEED);
    dequantize_residual_2bit(codes, D, scale, signs.data(), out);
}

/* ---------- Per-head compression ---------- */

static anchor_kv_head compress_head(
    const float * keys,     /* [S * D] */
    const float * values,   /* [S * D] */
    int S, int D,
    const anchor_kv_params & params
) {
    anchor_kv_head head;
    head.S = S;
    head.D = D;
    head.theta = params.theta;

    int k_budget = S / params.k_frac;
    if (k_budget < params.W) k_budget = params.W;
    head.k = k_budget;

    /* Generate WHT signs */
    std::vector<float> signs(D);
    anchor_kv_get_wht_signs(signs.data(), D, ANCHOR_KV_SEED);

    /* --- Step 1: Select anchors --- */

    /* Score non-window positions using observation queries (Eq. 14) */
    int P = S - params.W;
    int n_scored = head.k - params.W;
    int n_top = (int)(params.rho * n_scored);
    int n_random = n_scored - n_top;

    std::vector<float> scores(P, 0.0f);
    float inv_sqrt_d = 1.0f / sqrtf((float)D);

    for (int t = 0; t < P; t++) {
        float score_sum = 0.0f;
        for (int q = S - params.W; q < S; q++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += keys[q * D + d] * keys[t * D + d];
            }
            score_sum += expf(dot * inv_sqrt_d);
        }
        scores[t] = score_sum / params.W;
    }

    /* Average pool with kernel kappa */
    std::vector<float> pooled(P, 0.0f);
    for (int t = 0; t < P; t++) {
        float sum = 0.0f;
        int cnt = 0;
        for (int j = t - params.kappa / 2; j <= t + params.kappa / 2; j++) {
            if (j >= 0 && j < P) { sum += scores[j]; cnt++; }
        }
        pooled[t] = sum / cnt;
    }

    /* Build anchor set: window + top scored + random */
    std::vector<int> is_anchor(S, 0);
    std::vector<int> anchor_positions;

    for (int t = S - params.W; t < S; t++) {
        is_anchor[t] = 1;
        anchor_positions.push_back(t);
    }

    /* Select top n_top by score */
    std::vector<int> remaining(P);
    std::iota(remaining.begin(), remaining.end(), 0);
    int n_rem = P;

    for (int i = 0; i < n_top && n_rem > 0; i++) {
        int best = 0;
        for (int j = 1; j < n_rem; j++) {
            if (pooled[remaining[j]] > pooled[remaining[best]]) best = j;
        }
        int pos = remaining[best];
        is_anchor[pos] = 1;
        anchor_positions.push_back(pos);
        remaining[best] = remaining[--n_rem];
    }

    /* Random remaining anchors */
    uint64_t rng = ANCHOR_KV_SEED + 1000;
    for (int i = 0; i < n_random && n_rem > 0; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int idx = (int)(rng % (uint64_t)n_rem);
        int pos = remaining[idx];
        is_anchor[pos] = 1;
        anchor_positions.push_back(pos);
        remaining[idx] = remaining[--n_rem];
    }

    head.anchor_positions.assign(anchor_positions.begin(), anchor_positions.end());

    /* Store exact anchor K/V */
    head.anchor_keys.resize(head.k * D);
    head.anchor_values.resize(head.k * D);
    for (int a = 0; a < head.k; a++) {
        int pos = anchor_positions[a];
        memcpy(&head.anchor_keys[a * D], &keys[pos * D], D * sizeof(float));
        memcpy(&head.anchor_values[a * D], &values[pos * D], D * sizeof(float));
    }

    /* Reverse lookup: token position -> anchor index */
    std::vector<int> pos_to_anchor_idx(S, -1);
    for (int a = 0; a < head.k; a++) {
        pos_to_anchor_idx[anchor_positions[a]] = a;
    }

    /* --- Step 2: Compute projections (Eq. 1-2) --- */

    head.k_anchor_of.resize(S);
    head.v_anchor_of.resize(S);
    head.k_gamma.resize(S, 0.0f);
    head.v_gamma.resize(S, 0.0f);

    std::vector<float> residual_K(S * D);
    std::vector<float> residual_V(S * D);

    auto compute_proj = [&](const float * data, int * anchor_of, float * gamma, float * residual) {
        for (int t = 0; t < S; t++) {
            if (is_anchor[t]) {
                anchor_of[t] = pos_to_anchor_idx[t];
                gamma[t] = 1.0f;
                memset(&residual[t * D], 0, D * sizeof(float));
                continue;
            }

            /* Find nearest anchor by cosine similarity */
            float data_norm = 0.0f;
            for (int d = 0; d < D; d++) data_norm += data[t * D + d] * data[t * D + d];
            data_norm = sqrtf(data_norm);
            if (data_norm < 1e-10f) data_norm = 1e-10f;

            float best_sim = -1.0f;
            int best_idx = 0;
            for (int a = 0; a < head.k; a++) {
                float dot = 0.0f, anch_norm = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += data[t * D + d] * head.anchor_keys[a * D + d]; /* placeholder, use actual anchor data */
                    anch_norm += head.anchor_keys[a * D + d] * head.anchor_keys[a * D + d];
                }
                /* Use correct anchor data based on which side we're computing */
                /* This is called separately for K and V with different anchor data */

                anch_norm = sqrtf(anch_norm);
                if (anch_norm < 1e-10f) anch_norm = 1e-10f;
                float sim = fabsf(dot) / (data_norm * anch_norm);
                if (sim > best_sim) { best_sim = sim; best_idx = a; }
            }

            anchor_of[t] = best_idx;

            /* Projection coefficient (Eq. 2) */
            float anch_norm_sq = 0.0f;
            for (int d = 0; d < D; d++) {
                anch_norm_sq += head.anchor_keys[best_idx * D + d] * head.anchor_keys[best_idx * D + d];
            }
            if (anch_norm_sq < 1e-20f) anch_norm_sq = 1e-20f;

            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += data[t * D + d] * head.anchor_keys[best_idx * D + d];
            }
            gamma[t] = dot / anch_norm_sq;

            /* Residual */
            float res_sq = 0.0f;
            for (int d = 0; d < D; d++) {
                float proj_val = gamma[t] * head.anchor_keys[best_idx * D + d];
                residual[t * D + d] = data[t * D + d] - proj_val;
                res_sq += residual[t * D + d] * residual[t * D + d];
            }
        }
    };

    /* K projection */
    compute_proj(keys, head.k_anchor_of.data(), head.k_gamma.data(), residual_K.data());

    /* V projection (recompute with V anchors) */
    {
        for (int t = 0; t < S; t++) {
            if (is_anchor[t]) {
                head.v_anchor_of[t] = pos_to_anchor_idx[t];
                head.v_gamma[t] = 1.0f;
                memset(&residual_V[t * D], 0, D * sizeof(float));
                continue;
            }

            float data_norm = 0.0f;
            for (int d = 0; d < D; d++) data_norm += values[t * D + d] * values[t * D + d];
            data_norm = sqrtf(data_norm);
            if (data_norm < 1e-10f) data_norm = 1e-10f;

            float best_sim = -1.0f;
            int best_idx = 0;
            for (int a = 0; a < head.k; a++) {
                float dot = 0.0f, anch_norm = 0.0f;
                for (int d = 0; d < D; d++) {
                    dot += values[t * D + d] * head.anchor_values[a * D + d];
                    anch_norm += head.anchor_values[a * D + d] * head.anchor_values[a * D + d];
                }
                anch_norm = sqrtf(anch_norm);
                if (anch_norm < 1e-10f) anch_norm = 1e-10f;
                float sim = fabsf(dot) / (data_norm * anch_norm);
                if (sim > best_sim) { best_sim = sim; best_idx = a; }
            }

            head.v_anchor_of[t] = best_idx;

            float anch_norm_sq = 0.0f;
            for (int d = 0; d < D; d++) {
                anch_norm_sq += head.anchor_values[best_idx * D + d] * head.anchor_values[best_idx * D + d];
            }
            if (anch_norm_sq < 1e-20f) anch_norm_sq = 1e-20f;

            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += values[t * D + d] * head.anchor_values[best_idx * D + d];
            }
            head.v_gamma[t] = dot / anch_norm_sq;

            float res_sq = 0.0f;
            for (int d = 0; d < D; d++) {
                float proj_val = head.v_gamma[t] * head.anchor_values[best_idx * D + d];
                residual_V[t * D + d] = values[t * D + d] - proj_val;
                res_sq += residual_V[t * D + d] * residual_V[t * D + d];
            }
        }
    }

    /* --- Step 3: Score utilities (Eq. 6) --- */

    utility_scores_t util = score_utilities(
        residual_K.data(), residual_V.data(), values,
        head.k_gamma.data(), head.v_gamma.data(),
        head.k_anchor_of.data(), head.v_anchor_of.data(),
        is_anchor.data(), S, D
    );

    /* --- Step 4: Compute residual budget (Eq. 9) --- */

    int N = anchor_kv_max_residuals(S, D, params.W, params.theta);
    head.n_K = N / 2;
    head.n_V = N - head.n_K;

    /* --- Step 5: Select residuals by utility --- */

    /* Collect non-anchor positions */
    std::vector<int> non_anchors;
    for (int t = 0; t < S; t++) {
        if (!is_anchor[t]) non_anchors.push_back(t);
    }

    /* Select top N_K by key utility */
    std::vector<int> sel_K;
    {
        std::vector<int> rem = non_anchors;
        int n_r = (int)rem.size();
        for (int i = 0; i < head.n_K && n_r > 0; i++) {
            int best = 0;
            for (int j = 1; j < n_r; j++) {
                if (util.uK[rem[j]] > util.uK[rem[best]]) best = j;
            }
            sel_K.push_back(rem[best]);
            rem[best] = rem[--n_r];
        }
        /* Update n_K to actual count */
        head.n_K = (int)sel_K.size();
    }

    /* Select top N_V by value utility from remaining */
    std::vector<int> sel_V;
    {
        std::vector<int> rem;
        for (int t = 0; t < S; t++) {
            if (!is_anchor[t]) {
                bool in_K = false;
                for (int s : sel_K) { if (s == t) { in_K = true; break; } }
                if (!in_K) rem.push_back(t);
            }
        }
        int n_r = (int)rem.size();
        for (int i = 0; i < head.n_V && n_r > 0; i++) {
            int best = 0;
            for (int j = 1; j < n_r; j++) {
                if (util.uV[rem[j]] > util.uV[rem[best]]) best = j;
            }
            sel_V.push_back(rem[best]);
            rem[best] = rem[--n_r];
        }
        head.n_V = (int)sel_V.size();
    }

    /* --- Step 6: Build residual masks and quantize --- */

    int n_mask_words = (S + 63) / 64;
    head.k_residual_mask.assign(n_mask_words, 0);
    head.v_residual_mask.assign(n_mask_words, 0);

    for (int t : sel_K) {
        head.k_residual_mask[t / 64] |= (1ULL << (t % 64));
    }
    for (int t : sel_V) {
        head.v_residual_mask[t / 64] |= (1ULL << (t % 64));
    }

    /* Quantize key residuals */
    size_t codes_per_res = (size_t)D / 4;
    head.k_res_codes.resize(sel_K.size() * codes_per_res);
    head.k_res_scales.resize(sel_K.size());

    for (int i = 0; i < (int)sel_K.size(); i++) {
        int t = sel_K[i];
        head.k_res_scales[i] = quantize_residual_2bit(
            &residual_K[t * D], D, signs.data(),
            &head.k_res_codes[i * codes_per_res]
        );
    }

    /* Quantize value residuals */
    head.v_res_codes.resize(sel_V.size() * codes_per_res);
    head.v_res_scales.resize(sel_V.size());
    head.v_slot_positions.resize(sel_V.size());

    for (int i = 0; i < (int)sel_V.size(); i++) {
        int t = sel_V[i];
        head.v_res_scales[i] = quantize_residual_2bit(
            &residual_V[t * D], D, signs.data(),
            &head.v_res_codes[i * codes_per_res]
        );
        head.v_slot_positions[i] = (uint16_t)t;
    }

    /* Precompute slot indices for GPU decompression */
    head.k_slot_of.assign(S, -1);
    head.v_slot_of.assign(S, -1);
    for (int i = 0; i < (int)sel_K.size(); i++) {
        head.k_slot_of[sel_K[i]] = i;
    }
    for (int i = 0; i < (int)sel_V.size(); i++) {
        head.v_slot_of[sel_V[i]] = i;
    }

    return head;
}

/* ---------- Public API ---------- */

anchor_kv_layer anchor_kv_compress(
    const float * keys,
    const float * values,
    int S, int D, int n_heads,
    const anchor_kv_params & params
) {
    anchor_kv_layer layer;
    layer.n_heads = n_heads;
    layer.heads.resize(n_heads);

    for (int h = 0; h < n_heads; h++) {
        const float * h_keys   = keys + (size_t)h * S * D;
        const float * h_values = values + (size_t)h * S * D;
        layer.heads[h] = compress_head(h_keys, h_values, S, D, params);
    }

    return layer;
}

void anchor_kv_decompress_head(
    const anchor_kv_head & head,
    float * out_k,
    float * out_v
) {
    int S = head.S;
    int D = head.D;

    std::vector<float> signs(D);
    anchor_kv_get_wht_signs(signs.data(), D, ANCHOR_KV_SEED);

    /* Build is_anchor set from anchor_positions */
    std::vector<int> is_anchor(S, 0);
    for (int i = 0; i < head.k; i++) {
        is_anchor[head.anchor_positions[i]] = 1;
    }

    /* Position -> residual slot lookup: residual codes are stored in
     * utility-selection order (sel_K/sel_V in compress_head), not ascending
     * position order, so the slot index must come from the precomputed
     * k_slot_of/v_slot_of arrays rather than from counting bitmask bits. */
    const int * k_slot = head.k_slot_of.data();
    const int * v_slot = head.v_slot_of.data();

    /* Reconstruct K */
    for (int t = 0; t < S; t++) {
        if (is_anchor[t]) {
            /* Anchor: stored exactly */
            int a = head.k_anchor_of[t];
            memcpy(&out_k[t * D], &head.anchor_keys[a * D], D * sizeof(float));
        } else {
            int a = head.k_anchor_of[t];
            for (int d = 0; d < D; d++) {
                out_k[t * D + d] = head.k_gamma[t] * head.anchor_keys[a * D + d];
            }
            if (k_slot[t] >= 0) {
                float deq[128];
                size_t codes_per_res = (size_t)D / 4;
                dequantize_residual_2bit(
                    &head.k_res_codes[k_slot[t] * codes_per_res],
                    D, head.k_res_scales[k_slot[t]],
                    signs.data(), deq
                );
                for (int d = 0; d < D; d++) {
                    out_k[t * D + d] += deq[d];
                }
            }
        }
    }

    /* Reconstruct V */
    for (int t = 0; t < S; t++) {
        if (is_anchor[t]) {
            int a = head.v_anchor_of[t];
            memcpy(&out_v[t * D], &head.anchor_values[a * D], D * sizeof(float));
        } else {
            int a = head.v_anchor_of[t];
            for (int d = 0; d < D; d++) {
                out_v[t * D + d] = head.v_gamma[t] * head.anchor_values[a * D + d];
            }
            if (v_slot[t] >= 0) {
                float deq[128];
                size_t codes_per_res = (size_t)D / 4;
                dequantize_residual_2bit(
                    &head.v_res_codes[v_slot[t] * codes_per_res],
                    D, head.v_res_scales[v_slot[t]],
                    signs.data(), deq
                );
                for (int d = 0; d < D; d++) {
                    out_v[t * D + d] += deq[d];
                }
            }
        }
    }
}
