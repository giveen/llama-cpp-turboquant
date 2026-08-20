/*
 * AnchorKV: Anchor-Residual KV Cache Compression
 *
 * Storage format and CPU compression pass for the AnchorKV algorithm
 * (arXiv:2608.02901v1, Khalaf et al., 2026).
 *
 * This file defines the per-layer compressed representation and the
 * compression function that runs at the end of prefill. The decode
 * kernel (GPU) is a separate piece not yet implemented.
 *
 * The compressed representation stores:
 *   - Dense K/V for the last W positions (recency window, stored exactly)
 *   - Per-head anchor positions, projection coefficients, residual codes
 *   - A bitmask indicating which positions have stored residuals
 *
 * All tensors are runtime-only, never stored in GGUF.
 */

#ifndef ANCHOR_KV_H
#define ANCHOR_KV_H

#include "ggml.h"
#include <cstdint>
#include <cstddef>
#include <vector>

/* ---------- Paper hyperparameters (fixed, Section 4.1) ---------- */

#define ANCHOR_KV_W         32     /* recency window */
#define ANCHOR_KV_K_FRAC    128    /* anchor budget = seq_len / 128 */
#define ANCHOR_KV_RHO       0.7f   /* fraction of anchor slots for scored positions */
#define ANCHOR_KV_KAPPA     7      /* pooling kernel width */
#define ANCHOR_KV_SEED      42     /* deterministic seed for random anchors and WHT signs */

/* ---------- Per-head compressed representation ---------- */

/*
 * Storage layout per KV head (all fields are runtime-only):
 *
 *   Anchor keys:    [k, D] in bf16           -- exact anchor K vectors
 *   Anchor values:  [k, D] in bf16           -- exact anchor V vectors
 *   Anchor positions: [k] in int32           -- position indices of anchors
 *   K coefficients: [S] in fp16              -- projection coefficient per position (K side)
 *   V coefficients: [S] in fp16              -- projection coefficient per position (V side)
 *   K anchor-of:    [S] in uint16            -- which anchor each position maps to (K side)
 *   V anchor-of:    [S] in uint16            -- which anchor each position maps to (V side)
 *   K residual mask: [ceil(S/64)] in uint64  -- bitmask: which positions have K residuals
 *   V residual mask: [ceil(S/64)] in uint64  -- bitmask: which positions have V residuals
 *   K residual codes: [N_K * D/4] in uint8   -- packed 2-bit codes
 *   K residual scales: [N_K] in fp32         -- per-token absmax scale
 *   V residual codes: [N_V * D/4] in uint8   -- packed 2-bit codes
 *   V residual scales: [N_V] in fp32         -- per-token absmax scale
 *   V slot positions: [N_V] in uint16        -- position index for each V residual slot
 *
 *   k = anchor count = max(W, S/128)
 *   S = sequence length (number of tokens in the cache)
 *   D = head dimension
 *   N_K, N_V = number of key/value residuals (from byte budget)
 */

struct anchor_kv_head {
    int k;              /* anchor count */
    int S;              /* sequence length */
    int D;              /* head dimension */
    int n_kv_heads;     /* number of KV heads (for cross-head pooling) */
    bool compress_k = true;  /* true when the K side is stored compressed */
    bool compress_v = true;  /* true when the V side is stored compressed */

    /* Anchor data */
    std::vector<uint32_t> anchor_positions;  /* [k] position indices */
    std::vector<float>    anchor_keys;       /* [k * D] exact anchor K vectors */
    std::vector<float>    anchor_values;     /* [k * D] exact anchor V vectors */

    /* Per-token projection data */
    std::vector<int>      k_anchor_of;       /* [S] which anchor each position maps to (K) */
    std::vector<int>      v_anchor_of;       /* [S] which anchor each position maps to (V) */
    std::vector<float>    k_gamma;           /* [S] projection coefficient (K) */
    std::vector<float>    v_gamma;           /* [S] projection coefficient (V) */

    /* Residual bitmasks */
    std::vector<uint64_t> k_residual_mask;   /* [ceil(S/64)] */
    std::vector<uint64_t> v_residual_mask;   /* [ceil(S/64)] */

    /* Quantized residuals */
    std::vector<uint8_t>  k_res_codes;       /* packed 2-bit codes */
    std::vector<float>    k_res_scales;      /* per-token absmax scale */
    std::vector<uint8_t>  v_res_codes;       /* packed 2-bit codes */
    std::vector<float>    v_res_scales;      /* per-token absmax scale */
    std::vector<uint16_t> v_slot_positions;  /* position index per V residual slot */

    /* Precomputed slot indices for GPU decompression */
    std::vector<int>      k_slot_of;         /* [S] slot index (-1 if no residual) */
    std::vector<int>      v_slot_of;         /* [S] slot index (-1 if no residual) */

    /* Temporary CPU state used for layer-wide residual allocation */
    std::vector<float>    k_residual_raw;
    std::vector<float>    v_residual_raw;
    std::vector<float>    k_utility;
    std::vector<float>    v_utility;
    std::vector<uint8_t>  is_anchor;
    std::vector<int>      k_selected_positions;
    std::vector<int>      v_selected_positions;

    /* Budget info */
    int n_K;            /* number of key residuals stored */
    int n_V;            /* number of value residuals stored */
    float theta;        /* compression ratio used */
};

/* ---------- Per-layer compressed representation ---------- */

struct anchor_kv_layer {
    int n_heads;
    std::vector<anchor_kv_head> heads;
};

/* ---------- Compression parameters ---------- */

struct anchor_kv_params {
    float theta;        /* combined retained fraction (both sides share one) */
    float theta_k;      /* K-side retained fraction (0 = not set) */
    float theta_v;      /* V-side retained fraction (0 = not set) */
    int   W;            /* recency window (default ANCHOR_KV_W = 32) */
    int   k_frac;       /* anchor budget fraction (default 128) */
    float rho;          /* scored anchor fraction (default 0.7) */
    int   kappa;        /* pooling kernel width (default 7) */
    bool  compress_k;   /* store the K side compressed (default true) */
    bool  compress_v;   /* store the V side compressed (default true) */

    anchor_kv_params()
        : theta(0.05f)
        , theta_k(0.0f)
        , theta_v(0.0f)
        , W(ANCHOR_KV_W)
        , k_frac(ANCHOR_KV_K_FRAC)
        , rho(ANCHOR_KV_RHO)
        , kappa(ANCHOR_KV_KAPPA)
        , compress_k(true)
        , compress_v(true)
    {}
};

/* ---------- CPU compression pass ---------- */

/*
 * Compress a dense KV cache into AnchorKV format.
 *
 * Arguments:
 *   keys    - dense K tensor data, row-major [S * D] per head
 *   values  - dense V tensor data, row-major [S * D] per head
 *   S       - sequence length (number of tokens)
 *   D       - head dimension
 *   n_heads - number of KV heads
 *   params  - compression parameters
 *
 * Returns:
 *   One anchor_kv_layer containing the compressed representation for all heads.
 *
 * The compression runs entirely on CPU. It selects anchors, computes
 * projections, scores residual utility, quantizes residuals, and packs
 * the storage format.
 */
anchor_kv_layer anchor_kv_compress(
    const float * keys,          /* [n_heads * S * D] pre-RoPE keys */
    const float * values,        /* [n_heads * S * D] values */
    int S, int D, int n_heads,
    const anchor_kv_params & params,
    const float * keys_score = nullptr /* post-RoPE keys for anchor scoring */
);

/*
 * Decompress a single head back to dense form (for validation only).
 *
 * Output is [S * D] in float32.
 */
void anchor_kv_decompress_head(
    const anchor_kv_head & head,
    float * out_k,  /* [S * D] output */
    float * out_v   /* [S * D] output */
);

/*
 * Compute byte budget for a given configuration.
 * Returns the number of residuals that fit.
 */
int anchor_kv_max_residuals(int S, int D, int W, float theta, int k_frac = ANCHOR_KV_K_FRAC);

/*
 * Per-side byte budget for K-only/V-only compression (is_v selects the side).
 * Returns the residual count whose storage fits the single side's theta budget.
 */
int anchor_kv_max_residuals_side(int S, int D, int W, float theta, bool is_v, int k_frac = ANCHOR_KV_K_FRAC);

/* Layer-wide budgets pooled across all KV heads. */
int anchor_kv_max_residuals_layer(int S, int D, int W, float theta, int n_heads, int k_frac = ANCHOR_KV_K_FRAC);
int anchor_kv_max_residuals_side_layer(int S, int D, int W, float theta, bool is_v, int n_heads, int k_frac = ANCHOR_KV_K_FRAC);

/*
 * Return true when compression is exact at the window or has residual capacity
 * for every configured compressed side.
 */
bool anchor_kv_budget_ready(int S, int D, const anchor_kv_params & params, int n_heads = 1);

/*
 * WHT sign array generation (deterministic from seed).
 * Used by both compression and (future) decode kernel.
 */
void anchor_kv_get_wht_signs(float * signs, int d, uint32_t seed = ANCHOR_KV_SEED);

/*
 * Dequantize a single 2-bit residual from packed codes.
 * Used by the decode kernel to reconstruct KV on-the-fly.
 */
void anchor_kv_dequantize_residual(
    const uint8_t * codes,  /* [D/4] packed 2-bit codes */
    int D,
    float scale,            /* absmax scale */
    float * out             /* [D] reconstructed residual */
);

#endif /* ANCHOR_KV_H */
