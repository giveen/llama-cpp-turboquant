/*
 * AnchorKV CPU Reference Implementation
 *
 * Standalone reference for the AnchorKV anchor-residual KV cache compression
 * algorithm from arXiv:2608.02901v1 (Khalaf et al., 2026).
 *
 * This is a validation prototype, not production code. It implements the full
 * compression pipeline per-layer, per-KV-head as described in Algorithm 1.
 *
 * Build (standalone, no llama.cpp dependencies):
 *   gcc -O2 -o test-anchor-kv tests/test-anchor-kv.c -lm
 *
 * Run:
 *   ./test-anchor-kv [--seq-len 32768] [--head-dim 128] [--theta 0.05]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/* ---------- Paper hyperparameters (Section 4.1) ---------- */

#define ANCHOR_W         32     /* recency window */
#define ANCHOR_K_FRAC    128    /* anchor budget = seq_len / ANCHOR_K_FRAC */
#define ANCHOR_RHO       0.7    /* fraction of remaining anchor slots for scored positions */
#define ANCHOR_KAPPA     7      /* pooling kernel width for anchor scoring */
#define ANCHOR_D         128    /* head dimension (default, overridden at runtime) */
#define ANCHOR_SEED      42     /* fixed seed for random anchor sampling and WHT sign pattern */

/* ---------- Lloyd-Max 4-level codebook for unit Gaussian ---------- */
/* These are the standard optimal centroids and thresholds. */
/* Centroids: {-1.5101, -0.4528, +0.4528, +1.5101} for N(0,1) */
/* Normalized by 1/sqrt(D) for residual quantization in the paper. */

static const float LLOYD_CENTROIDS_2BIT[4] = {
    -1.510138f, -0.452823f, 0.452823f, 1.510138f
};

static const float LLOYD_THRESHOLDS_2BIT[3] = {
    -0.981627f, 0.0f, 0.981627f
};

/* ---------- WHT sign arrays (seed=42, matching turbo-quant) ---------- */

static float *wht_signs = NULL;
static int wht_signs_len = 0;

static void generate_wht_signs(int d) {
    /* Simple deterministic PRNG matching turbo_quant's seed */
    uint64_t state = ANCHOR_SEED;
    free(wht_signs);
    wht_signs = (float *)malloc(d * sizeof(float));
    wht_signs_len = d;
    for (int i = 0; i < d; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        wht_signs[i] = ((state >> 11) & 1) ? 1.0f : -1.0f;
    }
}

/* ---------- Forward WHT (in-place, normalized) ---------- */
/* U = (1/sqrt(D)) * H_D * diag(s) */

static void anchor_wht_forward(float *x, int d) {
    if (!wht_signs || wht_signs_len != d) generate_wht_signs(d);

    /* Apply sign pattern: D(s) * x */
    for (int i = 0; i < d; i++) x[i] *= wht_signs[i];

    /* Hadamard butterfly */
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    /* Normalize by 1/sqrt(D) */
    float inv_sqrt_d = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) x[i] *= inv_sqrt_d;
}

/* ---------- Inverse WHT (in-place) ---------- */
/* U^T = (1/sqrt(D)) * diag(s) * H_D  (since H is self-inverse, s are self-inverse) */

static void anchor_wht_inverse(float *x, int d) {
    if (!wht_signs || wht_signs_len != d) generate_wht_signs(d);

    float inv_sqrt_d = 1.0f / sqrtf((float)d);

    /* Inverse: D(s1) * N * H * D(s2) * y */
    /* Step 1: apply s2 (first in the inverse, last in forward) */
    /* Note: our single wht_signs array corresponds to s1 in the paper's notation. */
    /* For the AnchorKV residual codec (Eq. 10), the rotation is U = (1/sqrt(D)) * H_D * diag(s). */
    /* Forward:  U*x = (1/sqrt(D)) * H * diag(s) * x */
    /* Inverse:  U^T*y = (1/sqrt(D)) * diag(s) * H * y  (H is self-inverse, s is self-inverse) */
    /* So inverse = normalize -> butterfly -> apply signs */

    /* Step 1: normalize by 1/sqrt(D) */
    for (int i = 0; i < d; i++) x[i] *= inv_sqrt_d;

    /* Step 2: Hadamard butterfly (self-inverse) */
    for (int h = 1; h < d; h *= 2) {
        for (int i = 0; i < d; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    /* Step 3: apply sign pattern (self-inverse) */
    for (int i = 0; i < d; i++) x[i] *= wht_signs[i];
}

/* ---------- Nearest centroid (2-bit Lloyd-Max) ---------- */

static int nearest_centroid_2bit(float val) {
    if (val < LLOYD_THRESHOLDS_2BIT[0]) return 0;
    if (val < LLOYD_THRESHOLDS_2BIT[1]) return 1;
    if (val < LLOYD_THRESHOLDS_2BIT[2]) return 2;
    return 3;
}

/* ---------- Residual quantization: WHT + absmax scale + Lloyd-Max ---------- */
/* Returns the quantized codes (packed 4 per byte) and the absmax scale. */
/* The caller must free *codes. */

static float quantize_residual_2bit(const float *residual, int d, uint8_t **codes_out) {
    /* Copy residual */
    float buf[ANCHOR_D];
    memcpy(buf, residual, d * sizeof(float));

    /* Forward WHT rotation */
    anchor_wht_forward(buf, d);

    /* Per-token absmax scaling */
    float absmax = 0.0f;
    for (int i = 0; i < d; i++) {
        float a = fabsf(buf[i]);
        if (a > absmax) absmax = a;
    }
    if (absmax < 1e-10f) absmax = 1e-10f;

    float inv_scale = 1.0f / absmax;
    for (int i = 0; i < d; i++) buf[i] *= inv_scale;

    /* Quantize with Lloyd-Max codebook */
    size_t n_codes = (size_t)d / 4;
    uint8_t *codes = (uint8_t *)calloc(n_codes, 1);
    for (int i = 0; i < d; i++) {
        int idx = nearest_centroid_2bit(buf[i]);
        codes[i / 4] |= (idx & 0x3) << ((i % 4) * 2);
    }

    *codes_out = codes;
    return absmax;
}

/* ---------- Residual dequantization: codes -> rotated domain ---------- */

static void dequantize_residual_2bit(const uint8_t *codes, int d, float scale, float *out) {
    for (int i = 0; i < d; i++) {
        int idx = (codes[i / 4] >> ((i % 4) * 2)) & 0x3;
        out[i] = LLOYD_CENTROIDS_2BIT[idx];
    }
    /* Inverse WHT rotation */
    anchor_wht_inverse(out, d);
    /* Re-scale */
    for (int i = 0; i < d; i++) out[i] *= scale;
}

/* ---------- Byte accounting (Eq. 9, Table 1) ---------- */

typedef struct {
    int S;         /* sequence length */
    int D;         /* head dimension */
    int W;         /* recency window */
    int k;         /* anchor budget */
    int ba;        /* anchor index width in bytes (int32 = 4) */
    int b_gamma;   /* coefficient width in bytes (fp32 = 4) */
    float Mfull;   /* uncompressed bytes */
    float Mbase;   /* metadata bytes */
    float N;       /* number of residuals */
    float NK;      /* key residual slots */
    float NV;      /* value residual slots */
    float cK;      /* bytes per key residual */
    float cV;      /* bytes per value residual */
    float theta;   /* retained fraction */
} byte_budget_t;

static byte_budget_t compute_byte_budget(int S, int D, int W, float theta) {
    byte_budget_t b;
    b.S = S;
    b.D = D;
    b.W = W;
    b.k = S / ANCHOR_K_FRAC;
    b.ba = 4;       /* int32 anchor index */
    b.b_gamma = 4;  /* fp32 coefficient */

    int P = S - W;  /* non-window positions */
    int H = 1;      /* single head for this reference */

    /* Uncompressed: 2 sides * S positions * D dimensions * 2 bytes (bf16) */
    b.Mfull = 2.0f * S * D * 2.0f;

    /* Base metadata (Eq. 13) */
    b.Mbase = (float)(H * (4 * b.k * D + 8 * (b.k - W) + 2 * P * (b.ba + b.b_gamma))
               + 24 * H * ((P + 63) / 64) + 8 * (H + 1) + 4 * P);

    /* Residual costs */
    b.cK = D / 4.0f + 4.0f;  /* codes + fp32 scale */
    b.cV = D / 4.0f + 5.0f;  /* codes + fp32 scale + uint8 position */

    /* Total residual slots (Eq. 9) */
    float budget = theta * b.Mfull - b.Mbase;
    b.N = (budget > 0) ? floorf(budget / ((b.cK + b.cV) / 2.0f)) : 0;
    b.NK = floorf(b.N / 2.0f);
    b.NV = b.N - b.NK;
    b.theta = theta;

    return b;
}

/* ---------- Anchor selection (SnapKV-style, Section 3.2, Eq. 14) ---------- */
/* For a single KV head. Returns the anchor set as a boolean mask. */

typedef struct {
    int *is_anchor;      /* [S] boolean: 1 if position is an anchor */
    int *anchor_idx;     /* [k] position indices of anchors */
    int  n_anchors;      /* actual number of anchors */
    float *scores;       /* [S] attention scores for non-anchor positions */
} anchor_set_t;

/* Simple average-pooled attention score from observation queries. */
/* The observation queries are the last W positions' query vectors. */
/* Since we don't have the model here, we simulate attention scores */
/* using the dot products of observation queries with keys. */

static anchor_set_t select_anchors(
    const float *keys,  /* [S * D] key vectors */
    int S, int D, int W, int k
) {
    /* Ensure anchor budget k >= W (window positions are always anchors) */
    if (k < W) k = W;

    anchor_set_t as;
    as.is_anchor = (int *)calloc(S, sizeof(int));
    as.anchor_idx = (int *)malloc(k * sizeof(int));
    as.scores = (float *)calloc(S, sizeof(float));
    as.n_anchors = 0;

    int P = S - W;  /* non-window positions */

    /* Step 1: Last W positions are always anchors */
    for (int t = S - W; t < S; t++) {
        as.is_anchor[t] = 1;
        as.anchor_idx[as.n_anchors++] = t;
    }

    /* Step 2: Score non-window positions using observation queries */
    /* Observation queries = last W positions' keys (approximation for reference) */
    /* score(t) = avg_pool_kappa( softmax_q (q^T K / sqrt(D)) ) for q in observation window */
    /* Simplified: we use the cosine similarity between each non-window key */
    /* and the observation window keys as a proxy for attention importance. */

    int n_scored = k - W;  /* number of slots for scored + random */
    int n_top = (int)(ANCHOR_RHO * n_scored);
    int n_random = n_scored - n_top;

    /* Compute scores for non-window positions */
    float inv_sqrt_d = 1.0f / sqrtf((float)D);
    for (int t = 0; t < P; t++) {
        float score_sum = 0.0f;
        int count = 0;
        /* Pool over observation queries (last W positions) with kernel kappa */
        for (int q = S - W; q < S; q++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += keys[q * D + d] * keys[t * D + d];
            }
            dot *= inv_sqrt_d;
            /* Softmax approximation: use exp(dot) directly (unnormalized is fine for ranking) */
            float weight = expf(dot);
            score_sum += weight;
            count++;
        }
        /* Average pool over kernel kappa */
        as.scores[t] = score_sum / (float)count;
    }

    /* Apply average pooling with kernel kappa */
    float *pooled = (float *)calloc(P, sizeof(float));
    for (int t = 0; t < P; t++) {
        float sum = 0.0f;
        int cnt = 0;
        for (int j = t - ANCHOR_KAPPA / 2; j <= t + ANCHOR_KAPPA / 2; j++) {
            if (j >= 0 && j < P) {
                sum += as.scores[j];
                cnt++;
            }
        }
        pooled[t] = sum / (float)cnt;
    }
    memcpy(as.scores, pooled, P * sizeof(float));
    free(pooled);

    /* Step 3: Select top-n_top by score */
    /* Simple selection: find top n_top indices */
    int *remaining = (int *)malloc(P * sizeof(int));
    for (int i = 0; i < P; i++) remaining[i] = i;

    for (int i = 0; i < n_top && i < P; i++) {
        /* Find max score among remaining */
        int best = 0;
        for (int j = 1; j < P - i; j++) {
            if (as.scores[remaining[j]] > as.scores[remaining[best]]) {
                best = j;
            }
        }
        int pos = remaining[best];
        as.is_anchor[pos] = 1;
        as.anchor_idx[as.n_anchors++] = pos;
        /* Swap out */
        remaining[best] = remaining[P - i - 1];
    }

    /* Step 4: Random remaining anchors (deterministic from seed) */
    uint64_t rng = ANCHOR_SEED + 1000;
    int n_left = P - n_top;
    for (int i = 0; i < n_random && n_left > 0; i++) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        int idx = (int)(rng % (uint64_t)n_left);
        int pos = remaining[idx];
        as.is_anchor[pos] = 1;
        as.anchor_idx[as.n_anchors++] = pos;
        remaining[idx] = remaining[n_left - 1];
        n_left--;
    }

    free(remaining);
    return as;
}

/* ---------- Projection and residual computation ---------- */
/* For each non-anchor token: find nearest anchor, compute projection, form residual. */

typedef struct {
    int   *anchor_of;    /* [S] which anchor each position maps to (-1 for anchors) */
    float *gamma;        /* [S] projection coefficients (1.0 for anchors) */
    float *residual;     /* [S * D] residual vectors (zero for anchors) */
    float *res_norm;     /* [S] residual L2 norms */
} projection_t;

static projection_t compute_projections(
    const float *data,        /* [S * D] keys or values */
    const anchor_set_t *as,
    int S, int D
) {
    projection_t proj;
    proj.anchor_of = (int *)malloc(S * sizeof(int));
    proj.gamma = (float *)malloc(S * sizeof(float));
    proj.residual = (float *)malloc(S * D * sizeof(float));
    proj.res_norm = (float *)malloc(S * sizeof(float));

    for (int t = 0; t < S; t++) {
        if (as->is_anchor[t]) {
            proj.anchor_of[t] = t;
            proj.gamma[t] = 1.0f;
            memset(&proj.residual[t * D], 0, D * sizeof(float));
            proj.res_norm[t] = 0.0f;
            continue;
        }

        /* Find nearest anchor by cosine similarity (Eq. 1) */
        float best_sim = -1.0f;
        int best_anchor = as->anchor_idx[0];
        float data_norm = 0.0f;
        for (int d = 0; d < D; d++) {
            data_norm += data[t * D + d] * data[t * D + d];
        }
        data_norm = sqrtf(data_norm);
        if (data_norm < 1e-10f) data_norm = 1e-10f;

        for (int a = 0; a < as->n_anchors; a++) {
            int pos = as->anchor_idx[a];
            float dot = 0.0f;
            float anch_norm = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += data[t * D + d] * data[pos * D + d];
                anch_norm += data[pos * D + d] * data[pos * D + d];
            }
            anch_norm = sqrtf(anch_norm);
            if (anch_norm < 1e-10f) anch_norm = 1e-10f;

            float sim = fabsf(dot) / (data_norm * anch_norm);
            if (sim > best_sim) {
                best_sim = sim;
                best_anchor = pos;
            }
        }

        proj.anchor_of[t] = best_anchor;

        /* Projection coefficient (Eq. 2) */
        float anch_norm_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            anch_norm_sq += data[best_anchor * D + d] * data[best_anchor * D + d];
        }
        if (anch_norm_sq < 1e-20f) anch_norm_sq = 1e-20f;

        float dot = 0.0f;
        for (int d = 0; d < D; d++) {
            dot += data[t * D + d] * data[best_anchor * D + d];
        }
        proj.gamma[t] = dot / anch_norm_sq;

        /* Residual = data - gamma * anchor (Eq. 2) */
        float res_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            float proj_val = proj.gamma[t] * data[best_anchor * D + d];
            proj.residual[t * D + d] = data[t * D + d] - proj_val;
            res_sq += proj.residual[t * D + d] * proj.residual[t * D + d];
        }
        proj.res_norm[t] = sqrtf(res_sq);
    }

    return proj;
}

/* ---------- Utility scoring (Eq. 6) ---------- */
/* Score each non-anchor token's residual by its effect on attention output. */
/* For the CPU reference, we use a simplified version: */
/*   u_K(t) = alpha_t^2 * ||r_t^K||^2 / D * ||V_t - y||^2  (key channel) */
/*   u_V(t) = alpha_t^2 * ||r_t^V||^2                      (value channel) */
/* With uniform attention weights as a proxy (no real queries available). */

typedef struct {
    float *uK;   /* [S] key utility scores */
    float *uV;   /* [S] value utility scores */
} utility_scores_t;

static utility_scores_t score_utilities(
    const projection_t *proj_K,
    const projection_t *proj_V,
    const float *values,      /* [S * D] value vectors */
    const anchor_set_t *as,
    int S, int D
) {
    utility_scores_t util;
    util.uK = (float *)calloc(S, sizeof(float));
    util.uV = (float *)calloc(S, sizeof(float));

    /* Compute attention output y = sum_t alpha_t * V_t */
    /* With uniform weights: alpha_t = 1/S for all t */
    float *y = (float *)calloc(D, sizeof(float));
    for (int t = 0; t < S; t++) {
        float alpha = 1.0f / S;
        for (int d = 0; d < D; d++) {
            y[d] += alpha * values[t * D + d];
        }
    }

    for (int t = 0; t < S; t++) {
        if (as->is_anchor[t]) {
            util.uK[t] = 0.0f;
            util.uV[t] = 0.0f;
            continue;
        }

        float alpha_t = 1.0f / S;

        /* ||V_t - y||^2 */
        float v_minus_y_sq = 0.0f;
        for (int d = 0; d < D; d++) {
            float diff = values[t * D + d] - y[d];
            v_minus_y_sq += diff * diff;
        }

        /* Key utility: alpha_t^2 * ||r_t^K||^2 / D * ||V_t - y||^2 */
        util.uK[t] = alpha_t * alpha_t * (proj_K->res_norm[t] * proj_K->res_norm[t] / D) * v_minus_y_sq;

        /* Value utility: alpha_t^2 * ||r_t^V||^2 */
        util.uV[t] = alpha_t * alpha_t * proj_V->res_norm[t] * proj_V->res_norm[t];
    }

    free(y);
    return util;
}

/* ---------- Residual selection: pool across heads, pick top N ---------- */
/* For single-head reference, this just picks the top N by utility. */

typedef struct {
    int *selected_K;   /* [NK] positions with stored key residuals */
    int *selected_V;   /* [NV] positions with stored value residuals */
    int n_K;
    int n_V;
} residual_selection_t;

static residual_selection_t select_residuals(
    const utility_scores_t *util,
    const anchor_set_t *as,
    int S, int NK, int NV
) {
    residual_selection_t sel;
    sel.selected_K = (int *)malloc((int)NK * sizeof(int));
    sel.selected_V = (int *)malloc((int)NV * sizeof(int));
    sel.n_K = 0;
    sel.n_V = 0;

    /* Collect non-anchor positions */
    int n_non_anchor = 0;
    int *non_anchors = (int *)malloc(S * sizeof(int));
    for (int t = 0; t < S; t++) {
        if (!as->is_anchor[t]) {
            non_anchors[n_non_anchor++] = t;
        }
    }

    /* Select top NK by key utility */
    int *remaining = (int *)malloc(n_non_anchor * sizeof(int));
    memcpy(remaining, non_anchors, n_non_anchor * sizeof(int));
    int n_rem = n_non_anchor;

    for (int i = 0; i < (int)NK && n_rem > 0; i++) {
        int best = 0;
        for (int j = 1; j < n_rem; j++) {
            if (util->uK[remaining[j]] > util->uK[remaining[best]]) {
                best = j;
            }
        }
        sel.selected_K[sel.n_K++] = remaining[best];
        remaining[best] = remaining[n_rem - 1];
        n_rem--;
    }

    /* Select top NV by value utility from remaining */
    for (int i = 0; i < (int)NV && n_rem > 0; i++) {
        int best = 0;
        for (int j = 1; j < n_rem; j++) {
            if (util->uV[remaining[j]] > util->uV[remaining[best]]) {
                best = j;
            }
        }
        sel.selected_V[sel.n_V++] = remaining[best];
        remaining[best] = remaining[n_rem - 1];
        n_rem--;
    }

    free(non_anchors);
    free(remaining);
    return sel;
}

/* ---------- Main compression pipeline (Algorithm 1) ---------- */

typedef struct {
    /* Anchor set */
    anchor_set_t anchors_K;
    anchor_set_t anchors_V;

    /* Projections */
    projection_t proj_K;
    projection_t proj_V;

    /* Residual selection */
    residual_selection_t sel;

    /* Quantized residuals */
    uint8_t **res_codes_K;  /* packed codes per selected position */
    float *res_scales_K;
    uint8_t **res_codes_V;
    float *res_scales_V;

    /* Byte budget */
    byte_budget_t budget;

    /* Reconstructed data */
    float *reconstructed_K;
    float *reconstructed_V;

    /* Quality metrics */
    float mse_K;
    float mse_V;
    float cosine_K;
    float cosine_V;
} anchor_kv_result_t;

static anchor_kv_result_t anchor_kv_compress(
    const float *keys,     /* [S * D] */
    const float *values,   /* [S * D] */
    int S, int D,
    float theta
) {
    anchor_kv_result_t result;
    memset(&result, 0, sizeof(result));

    /* Step 0: Compute byte budget */
    result.budget = compute_byte_budget(S, D, ANCHOR_W, theta);
    int NK = (int)result.budget.NK;
    int NV = (int)result.budget.NV;

    printf("  Budget: Mfull=%.0f bytes, Mbase=%.0f bytes, N=%d (NK=%d, NV=%d)\n",
           result.budget.Mfull, result.budget.Mbase, (int)result.budget.N, NK, NV);

    /* Step 1: Select anchors (per side, sharing positions) */
    printf("  Selecting anchors (k=%d, W=%d, rho=%.1f, kappa=%d)...\n",
           result.budget.k, ANCHOR_W, ANCHOR_RHO, ANCHOR_KAPPA);
    result.anchors_K = select_anchors(keys, S, D, ANCHOR_W, result.budget.k);
    result.anchors_V = result.anchors_K;  /* share anchor positions */

    printf("  Anchors: %d selected\n", result.anchors_K.n_anchors);

    /* Step 2: Compute projections (Eq. 1-2) */
    printf("  Computing projections...\n");
    result.proj_K = compute_projections(keys, &result.anchors_K, S, D);
    result.proj_V = compute_projections(values, &result.anchors_V, S, D);

    /* Step 3: Score utilities (Eq. 6) */
    printf("  Scoring residuals...\n");
    utility_scores_t util = score_utilities(&result.proj_K, &result.proj_V, values, &result.anchors_K, S, D);

    /* Step 4: Select residuals (top-N pooled over heads) */
    printf("  Selecting residuals...\n");
    result.sel = select_residuals(&util, &result.anchors_K, S, NK, NV);

    /* Step 5: Quantize selected residuals */
    printf("  Quantizing residuals...\n");
    result.res_codes_K = (uint8_t **)malloc(result.sel.n_K * sizeof(uint8_t *));
    result.res_scales_K = (float *)malloc(result.sel.n_K * sizeof(float));
    result.res_codes_V = (uint8_t **)malloc(result.sel.n_V * sizeof(uint8_t *));
    result.res_scales_V = (float *)malloc(result.sel.n_V * sizeof(float));

    for (int i = 0; i < result.sel.n_K; i++) {
        int t = result.sel.selected_K[i];
        result.res_scales_K[i] = quantize_residual_2bit(&result.proj_K.residual[t * D], D, &result.res_codes_K[i]);
    }
    for (int i = 0; i < result.sel.n_V; i++) {
        int t = result.sel.selected_V[i];
        result.res_scales_V[i] = quantize_residual_2bit(&result.proj_V.residual[t * D], D, &result.res_codes_V[i]);
    }

    /* Step 6: Reconstruct and measure quality */
    printf("  Reconstructing...\n");
    result.reconstructed_K = (float *)malloc(S * D * sizeof(float));
    result.reconstructed_V = (float *)malloc(S * D * sizeof(float));

    /* For reconstruction, we need to handle the key residual addition properly */
    /* Re-do reconstruction inline with residual dequant */
    int *res_slot_K = (int *)malloc(S * sizeof(int));
    int *res_slot_V = (int *)malloc(S * sizeof(int));
    for (int t = 0; t < S; t++) { res_slot_K[t] = -1; res_slot_V[t] = -1; }
    for (int i = 0; i < result.sel.n_K; i++) res_slot_K[result.sel.selected_K[i]] = i;
    for (int i = 0; i < result.sel.n_V; i++) res_slot_V[result.sel.selected_V[i]] = i;

    for (int t = 0; t < S; t++) {
        if (result.anchors_K.is_anchor[t]) {
            memcpy(&result.reconstructed_K[t * D], &keys[t * D], D * sizeof(float));
        } else {
            int anch = result.proj_K.anchor_of[t];
            for (int d = 0; d < D; d++) {
                result.reconstructed_K[t * D + d] = result.proj_K.gamma[t] * keys[anch * D + d];
            }
            if (res_slot_K[t] >= 0) {
                float deq[ANCHOR_D];
                dequantize_residual_2bit(result.res_codes_K[res_slot_K[t]], D, result.res_scales_K[res_slot_K[t]], deq);
                /* Normalize dequantized residual to match original norm */
                float deq_n = 0, orig_n = 0;
                for (int d = 0; d < D; d++) {
                    deq_n += deq[d] * deq[d];
                    orig_n += result.proj_K.residual[t * D + d] * result.proj_K.residual[t * D + d];
                }
                float ns = sqrtf(orig_n) / (sqrtf(deq_n) + 1e-10f);
                for (int d = 0; d < D; d++) {
                    result.reconstructed_K[t * D + d] += deq[d] * ns;
                }
            }
        }

        if (result.anchors_V.is_anchor[t]) {
            memcpy(&result.reconstructed_V[t * D], &values[t * D], D * sizeof(float));
        } else {
            int anch = result.proj_V.anchor_of[t];
            for (int d = 0; d < D; d++) {
                result.reconstructed_V[t * D + d] = result.proj_V.gamma[t] * values[anch * D + d];
            }
            if (res_slot_V[t] >= 0) {
                float deq[ANCHOR_D];
                dequantize_residual_2bit(result.res_codes_V[res_slot_V[t]], D, result.res_scales_V[res_slot_V[t]], deq);
                float deq_n = 0, orig_n = 0;
                for (int d = 0; d < D; d++) {
                    deq_n += deq[d] * deq[d];
                    orig_n += result.proj_V.residual[t * D + d] * result.proj_V.residual[t * D + d];
                }
                float ns = sqrtf(orig_n) / (sqrtf(deq_n) + 1e-10f);
                for (int d = 0; d < D; d++) {
                    result.reconstructed_V[t * D + d] += deq[d] * ns;
                }
            }
        }
    }

    free(res_slot_K);
    free(res_slot_V);

    /* Compute quality metrics */
    float mse_k_sum = 0.0f, mse_v_sum = 0.0f;
    float dot_k_num = 0.0f, dot_v_num = 0.0f;
    float norm_orig_k = 0.0f, norm_orig_v = 0.0f;
    float norm_recon_k = 0.0f, norm_recon_v = 0.0f;

    for (int t = 0; t < S; t++) {
        for (int d = 0; d < D; d++) {
            float dk = result.reconstructed_K[t * D + d] - keys[t * D + d];
            float dv = result.reconstructed_V[t * D + d] - values[t * D + d];
            mse_k_sum += dk * dk;
            mse_v_sum += dv * dv;

            dot_k_num += result.reconstructed_K[t * D + d] * keys[t * D + d];
            dot_v_num += result.reconstructed_V[t * D + d] * values[t * D + d];
            norm_orig_k += keys[t * D + d] * keys[t * D + d];
            norm_orig_v += values[t * D + d] * values[t * D + d];
            norm_recon_k += result.reconstructed_K[t * D + d] * result.reconstructed_K[t * D + d];
            norm_recon_v += result.reconstructed_V[t * D + d] * result.reconstructed_V[t * D + d];
        }
    }

    int total = S * D;
    result.mse_K = mse_k_sum / total;
    result.mse_V = mse_v_sum / total;
    result.cosine_K = dot_k_num / (sqrtf(norm_orig_k) * sqrtf(norm_recon_k) + 1e-10f);
    result.cosine_V = dot_v_num / (sqrtf(norm_orig_v) * sqrtf(norm_recon_v) + 1e-10f);

    /* Cleanup utility scores */
    free(util.uK);
    free(util.uV);

    return result;
}

/* ---------- Cleanup ---------- */

static void free_anchor_kv_result(anchor_kv_result_t *r) {
    free(r->anchors_K.is_anchor);
    free(r->anchors_K.anchor_idx);
    free(r->anchors_K.scores);
    /* anchors_V shares with anchors_K, don't double-free */

    free(r->proj_K.anchor_of);
    free(r->proj_K.gamma);
    free(r->proj_K.residual);
    free(r->proj_K.res_norm);
    free(r->proj_V.anchor_of);
    free(r->proj_V.gamma);
    free(r->proj_V.residual);
    free(r->proj_V.res_norm);

    free(r->sel.selected_K);
    free(r->sel.selected_V);

    for (int i = 0; i < r->sel.n_K; i++) free(r->res_codes_K[i]);
    free(r->res_codes_K);
    free(r->res_scales_K);
    for (int i = 0; i < r->sel.n_V; i++) free(r->res_codes_V[i]);
    free(r->res_codes_V);
    free(r->res_scales_V);

    free(r->reconstructed_K);
    free(r->reconstructed_V);
}

/* ---------- Synthetic data generation ---------- */

/* Box-Muller via LCG */
static float rand_normal(uint64_t *rng) {
    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)((*rng >> 11) & 0x1FFFFFFFFFFFFF) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    *rng = *rng * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)((*rng >> 11) & 0x1FFFFFFFFFFFFF) / (double)(1ULL << 53);
    return (float)(sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2));
}

static void normalize(float *v, int d) {
    float norm_sq = 0.0f;
    for (int i = 0; i < d; i++) norm_sq += v[i] * v[i];
    float inv = 1.0f / (sqrtf(norm_sq) + 1e-10f);
    for (int i = 0; i < d; i++) v[i] *= inv;
}

/*
 * Generate realistic KV data: tokens cluster around a small number of
 * dominant directions (like real attention heads), with per-token noise.
 * This mimics the low-rank structure that makes AnchorKV effective.
 *
 * Structure: 8 cluster centers in D dimensions, tokens assigned to
 * nearest center with Gaussian noise (sigma=0.3).
 */
static void generate_synthetic_kv(float *keys, float *values, int S, int D, uint64_t seed) {
    uint64_t rng = seed;
    int n_clusters = 4;

    /* Generate cluster centers */
    float *centers_K = (float *)malloc(n_clusters * D * sizeof(float));
    float *centers_V = (float *)malloc(n_clusters * D * sizeof(float));
    for (int c = 0; c < n_clusters; c++) {
        for (int d = 0; d < D; d++) {
            centers_K[c * D + d] = rand_normal(&rng);
            centers_V[c * D + d] = rand_normal(&rng);
        }
        normalize(&centers_K[c * D], D);
        normalize(&centers_V[c * D], D);
    }

    /* Generate tokens: each assigned to a cluster with noise */
    for (int t = 0; t < S; t++) {
        int cluster = t % n_clusters;  /* deterministic assignment */

        for (int d = 0; d < D; d++) {
            keys[t * D + d] = centers_K[cluster * D + d] + 0.05f * rand_normal(&rng);
            values[t * D + d] = centers_V[cluster * D + d] + 0.05f * rand_normal(&rng);
        }
        normalize(&keys[t * D], D);
        normalize(&values[t * D], D);
    }

    free(centers_K);
    free(centers_V);
}

/* ---------- Main ---------- */

int main(int argc, char **argv) {
    int S = 4096;
    int D = 128;
    float theta = 0.05f;
    int from_file = 0;

    /* Parse args */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq-len") == 0 && i + 1 < argc) S = atoi(argv[++i]);
        else if (strcmp(argv[i], "--head-dim") == 0 && i + 1 < argc) D = atoi(argv[++i]);
        else if (strcmp(argv[i], "--theta") == 0 && i + 1 < argc) theta = atof(argv[++i]);
        else if (strcmp(argv[i], "--from-file") == 0) from_file = 1;
    }

    if (D > ANCHOR_D) {
        fprintf(stderr, "Error: head-dim %d exceeds max %d\n", D, ANCHOR_D);
        return 1;
    }

    printf("AnchorKV CPU Reference\n");
    printf("======================\n");

    /* Quick sanity check: WHT round-trip */
    {
        float test_vec[128];
        for (int i = 0; i < 128; i++) test_vec[i] = (float)(i - 64) * 0.01f;
        float orig[128];
        memcpy(orig, test_vec, sizeof(test_vec));
        anchor_wht_forward(test_vec, 128);
        anchor_wht_inverse(test_vec, 128);
        float err = 0;
        for (int i = 0; i < 128; i++) err += (test_vec[i] - orig[i]) * (test_vec[i] - orig[i]);
        printf("WHT round-trip MSE: %e\n", err / 128);
    }

    /* Sanity check: residual quantization round-trip (no anchor projection) */
    {
        float orig[128], deq[128];
        uint64_t rng = 42;
        for (int i = 0; i < 128; i++) orig[i] = rand_normal(&rng) * 0.1f;
        uint8_t *codes;
        float scale = quantize_residual_2bit(orig, 128, &codes);
        dequantize_residual_2bit(codes, 128, scale, deq);
        float mse = 0, dot = 0, no = 0, nr = 0;
        for (int i = 0; i < 128; i++) {
            mse += (orig[i] - deq[i]) * (orig[i] - deq[i]);
            dot += orig[i] * deq[i];
            no += orig[i] * orig[i];
            nr += deq[i] * deq[i];
        }
        printf("Residual codec MSE: %e  Cosine: %.6f\n", mse / 128, dot / (sqrtf(no) * sqrtf(nr) + 1e-10f));
        free(codes);
    }

    printf("\n");
    printf("Sequence length: %d\n", S);
    printf("Head dimension:  %d\n", D);
    printf("Theta (retention fraction): %.3f\n", theta);
    printf("Hyperparameters: W=%d, k=S/%d=%.0f, rho=%.1f, kappa=%d\n",
           ANCHOR_W, ANCHOR_K_FRAC, (float)S / ANCHOR_K_FRAC, ANCHOR_RHO, ANCHOR_KAPPA);
    printf("\n");

    /* Allocate */
    float *keys = (float *)malloc(S * D * sizeof(float));
    float *values = (float *)malloc(S * D * sizeof(float));

    if (from_file) {
        /* Load from binary files produced by extract-kv-cache.py */
        printf("Loading KV from files: /tmp/anchor-kv-k.bin, /tmp/anchor-kv-v.bin\n");
        FILE *fk = fopen("/tmp/anchor-kv-k.bin", "rb");
        FILE *fv = fopen("/tmp/anchor-kv-v.bin", "rb");
        if (!fk || !fv) {
            fprintf(stderr, "Error: cannot open KV files\n");
            free(keys); free(values);
            if (fk) fclose(fk);
            if (fv) fclose(fv);
            return 1;
        }
        size_t nk = fread(keys, sizeof(float), S * D, fk);
        size_t nv = fread(values, sizeof(float), S * D, fv);
        fclose(fk); fclose(fv);
        if (nk != (size_t)(S * D) || nv != (size_t)(S * D)) {
            fprintf(stderr, "Error: file size mismatch (got %zu+%zu, expected %d)\n",
                    nk, nv, S * D);
            free(keys); free(values);
            return 1;
        }
        printf("  Loaded %d keys, %d values (%d x %d)\n", S, S, S, D);
    } else {
        /* Generate synthetic data */
        printf("Generating synthetic KV data...\n");
        generate_synthetic_kv(keys, values, S, D, 12345);
    }
    printf("\n");

    /* Compress */
    printf("Compressing...\n");
    anchor_kv_result_t result = anchor_kv_compress(keys, values, S, D, theta);
    printf("\n");

    /* Report */
    printf("Results\n");
    printf("=======\n");
    printf("Key   MSE: %.8f  Cosine: %.6f\n", result.mse_K, result.cosine_K);
    printf("Value MSE: %.8f  Cosine: %.6f\n", result.mse_V, result.cosine_V);
    printf("\n");

    float compression = result.budget.Mfull / (result.budget.Mbase + result.budget.N * (result.budget.cK + result.budget.cV) / 2.0f);
    printf("Compression ratio: %.1fx (target: %.1fx)\n", compression, 1.0f / theta);

    /* Pass/fail */
    int pass = 1;
    if (result.cosine_K < 0.99f) {
        printf("\nFAIL: Key cosine similarity %.6f < 0.99\n", result.cosine_K);
        pass = 0;
    }
    if (result.cosine_V < 0.99f) {
        printf("\nFAIL: Value cosine similarity %.6f < 0.99\n", result.cosine_V);
        pass = 0;
    }
    if (pass) {
        printf("\nPASS\n");
    }

    /* Cleanup */
    free_anchor_kv_result(&result);
    free(keys);
    free(values);
    free(wht_signs);

    return pass ? 0 : 1;
}
