/*
 * KVarN: variance-normalized KV-cache quantization.
 *
 * The Hadamard rotation, Sinkhorn-style row/column variance balancing, and
 * tile bit-packing below are adapted from Anbeeld/beellama.cpp
 * (src/llama-kvarn.cpp), MIT licensed:
 *
 *   MIT License
 *   Copyright (c) 2023-2026 The ggml authors, Anbeeld
 *
 * Adapted from C++ to this file's C conventions and restricted to the
 * bit-widths this fork ships (2/3/4/5/6); the surrounding cache/op
 * integration is this fork's own and does not exist in the source above.
 *
 * This file implements only the per-tile math (rotate -> balance -> quantize
 * on a full 128-token x 128-channel tile). It is deliberately NOT wired into
 * the ggml_type trait table: unlike TurboQuant's quantize_row_* functions,
 * this codec needs all 128 rows of a group at once to compute the
 * cross-token column statistics, so it cannot be expressed as a per-row
 * ggml_type. Cache/op-level integration is separate follow-up work.
 */

#include "ggml-impl.h"
#include "ggml-kvarn-quant.h"

#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define KVARN_GROUP 128

static int kvarn_valid_bits(int bits) {
    return bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6;
}

/* ---------- bit packing ---------- */

size_t kvarn_packed_bytes(int n_values, int bits) {
    assert(n_values >= 0);
    assert(kvarn_valid_bits(bits));
    return ((size_t) n_values * (size_t) bits + 7) / 8;
}

void kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst) {
    assert(values != NULL);
    assert(dst != NULL);
    assert(kvarn_valid_bits(bits));

    memset(dst, 0, kvarn_packed_bytes(n_values, bits));

    const uint8_t mask = (uint8_t) ((1u << bits) - 1u);
    for (int i = 0; i < n_values; i++) {
        const uint8_t value = values[i] & mask;
        const size_t bit_offset = (size_t) i * (size_t) bits;

        for (int bit = 0; bit < bits; bit++) {
            const size_t dst_bit = bit_offset + (size_t) bit;
            dst[dst_bit / 8] |= (uint8_t) (((value >> bit) & 1u) << (dst_bit % 8));
        }
    }
}

uint8_t kvarn_unpack_bits_value(const uint8_t * src, int index, int bits) {
    assert(src != NULL);
    assert(kvarn_valid_bits(bits));

    uint8_t value = 0;
    const size_t bit_offset = (size_t) index * (size_t) bits;
    for (int bit = 0; bit < bits; bit++) {
        const size_t src_bit = bit_offset + (size_t) bit;
        value = (uint8_t) (value | (((src[src_bit / 8] >> (src_bit % 8)) & 1u) << bit));
    }

    return value;
}

/* ---------- per-tile layout ----------
 *
 * One record holds a K sub-record and a V sub-record for a single
 * (128-token group x 128-channel head slice), each with its own bit-width.
 * Metadata is three float16 arrays per side: a per-row combined scale, a
 * per-row combined zero-point, and a per-column scale (see the comment in
 * kvarn_quantize_tile for how these combine at dequant time).
 */

static size_t kvarn_align_up(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

struct kvarn_tile_layout kvarn_make_layout(int key_bits, int value_bits) {
    assert(kvarn_valid_bits(key_bits));
    assert(kvarn_valid_bits(value_bits));

    struct kvarn_tile_layout layout;
    memset(&layout, 0, sizeof(layout));
    size_t off = 0;

    layout.k_payload_off   = off;
    layout.k_payload_bytes = kvarn_packed_bytes(KVARN_GROUP * KVARN_GROUP, key_bits);
    off += layout.k_payload_bytes;

    layout.k_s_row_off   = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.k_zp_off      = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.k_s_col_off   = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.k_row_amp_off = off; off += KVARN_GROUP * sizeof(uint16_t);

    layout.v_payload_off   = off;
    layout.v_payload_bytes = kvarn_packed_bytes(KVARN_GROUP * KVARN_GROUP, value_bits);
    off += layout.v_payload_bytes;

    layout.v_s_row_off   = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.v_zp_off      = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.v_s_col_off   = off; off += KVARN_GROUP * sizeof(uint16_t);
    layout.v_row_amp_off = off; off += KVARN_GROUP * sizeof(uint16_t);

    layout.tile_bytes = kvarn_align_up(off, 8);
    return layout;
}

/* ---------- Hadamard rotation ----------
 *
 * Plain (non-randomized) 128-point Walsh-Hadamard transform, orthonormal so
 * it is its own inverse: calling this twice returns the input. This is a
 * different transform from this fork's turbo_cpu_fwht (which additionally
 * applies fixed pseudo-random sign diagonals before/after the butterfly);
 * KVarN's dual-axis Sinkhorn balancing is verified against this exact
 * (sign-free) rotation, so the two are not interchangeable.
 */
void kvarn_hadamard_128(float * values) {
    assert(values != NULL);

    for (int stride = 1; stride < KVARN_GROUP; stride *= 2) {
        for (int base = 0; base < KVARN_GROUP; base += 2 * stride) {
            for (int i = 0; i < stride; i++) {
                const float a = values[base + i];
                const float b = values[base + stride + i];
                values[base + i]          = a + b;
                values[base + stride + i] = a - b;
            }
        }
    }

    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < KVARN_GROUP; i++) {
        values[i] *= inv_sqrt_128;
    }
}

/* ---------- dual-axis variance normalization (Sinkhorn-style balancing) ---------- */

static float kvarn_sample_std(const float * values, int n, int stride) {
    double sum = 0.0;
    double sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        const double value = values[i * stride];
        sum += value;
        sum_sq += value * value;
    }

    const double mean = sum / n;
    const double variance = (sum_sq - n * mean * mean) / (n - 1);
    return (float) sqrt(variance > 0.0 ? variance : 0.0);
}

static float kvarn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Max/min ratio of per-column and per-row standard deviations; the
 * convergence proxy the balancing loop minimizes (lower = more balanced). */
static float kvarn_imbalance(const float * tile) {
    float col_min = INFINITY, col_max = 0.0f;
    float row_min = INFINITY, row_max = 0.0f;

    for (int c = 0; c < KVARN_GROUP; c++) {
        const float value = kvarn_sample_std(tile + c, KVARN_GROUP, KVARN_GROUP);
        if (value < col_min) col_min = value;
        if (value > col_max) col_max = value;
    }
    for (int r = 0; r < KVARN_GROUP; r++) {
        const float value = kvarn_sample_std(tile + r * KVARN_GROUP, KVARN_GROUP, 1);
        if (value < row_min) row_min = value;
        if (value > row_max) row_max = value;
    }

    const float col_min_safe = col_min > 1e-8f ? col_min : 1e-8f;
    const float row_min_safe = row_min > 1e-8f ? row_min : 1e-8f;
    return col_max / col_min_safe + row_max / row_min_safe;
}

/* Alternately rescales columns then rows by their standard deviation (in log
 * space) for `sinkhorn_iters` rounds, keeping the best (lowest-imbalance)
 * scale pair seen. `balanced` and `s_row`/`s_col` must each hold
 * KVARN_GROUP*KVARN_GROUP / KVARN_GROUP floats respectively. */
static void kvarn_variance_normalize(
        const float * tile,
        int sinkhorn_iters,
        float * balanced,
        float * s_col_best,
        float * s_row_best) {
    assert(tile != NULL);
    assert(sinkhorn_iters > 0);

    float log_s_col[KVARN_GROUP];
    float log_s_row[KVARN_GROUP];
    memset(log_s_col, 0, sizeof(log_s_col));
    memset(log_s_row, 0, sizeof(log_s_row));

    float * cur = (float *) malloc((size_t) KVARN_GROUP * KVARN_GROUP * sizeof(float));
    memcpy(cur, tile, (size_t) KVARN_GROUP * KVARN_GROUP * sizeof(float));

    for (int i = 0; i < KVARN_GROUP; i++) {
        s_col_best[i] = 1.0f;
        s_row_best[i] = 1.0f;
    }
    float imbalance_best = kvarn_imbalance(cur);

    for (int iter = 0; iter < sinkhorn_iters; iter++) {
        for (int c = 0; c < KVARN_GROUP; c++) {
            const float std = kvarn_clampf(kvarn_sample_std(cur + c, KVARN_GROUP, KVARN_GROUP), 1e-3f, 1e3f);
            log_s_col[c] = kvarn_clampf(log_s_col[c] + logf(std), -0.3f, 10.0f);
        }
        for (int r = 0; r < KVARN_GROUP; r++) {
            const float s_row = expf(log_s_row[r]);
            for (int c = 0; c < KVARN_GROUP; c++) {
                cur[r * KVARN_GROUP + c] = tile[r * KVARN_GROUP + c] / (expf(log_s_col[c]) * s_row);
            }
        }

        for (int r = 0; r < KVARN_GROUP; r++) {
            const float std = kvarn_clampf(kvarn_sample_std(cur + r * KVARN_GROUP, KVARN_GROUP, 1), 1e-3f, 1e3f);
            log_s_row[r] = kvarn_clampf(log_s_row[r] + logf(std), -0.3f, 10.0f);
        }
        for (int r = 0; r < KVARN_GROUP; r++) {
            const float s_row = expf(log_s_row[r]);
            for (int c = 0; c < KVARN_GROUP; c++) {
                cur[r * KVARN_GROUP + c] = tile[r * KVARN_GROUP + c] / (expf(log_s_col[c]) * s_row);
            }
        }

        const float imbalance = kvarn_imbalance(cur);
        if (imbalance <= imbalance_best) {
            imbalance_best = imbalance;
            for (int i = 0; i < KVARN_GROUP; i++) {
                s_col_best[i] = expf(log_s_col[i]);
                s_row_best[i] = expf(log_s_row[i]);
            }
        }
    }

    for (int r = 0; r < KVARN_GROUP; r++) {
        for (int c = 0; c < KVARN_GROUP; c++) {
            balanced[r * KVARN_GROUP + c] = tile[r * KVARN_GROUP + c] / (s_col_best[c] * s_row_best[r]);
        }
    }

    free(cur);
}

/* ---------- fp16 metadata helpers ---------- */

static void kvarn_store_fp16(uint8_t * record, size_t offset, int index, float value) {
    const ggml_fp16_t fp16 = ggml_fp32_to_fp16(value);
    memcpy(record + offset + (size_t) index * sizeof(fp16), &fp16, sizeof(fp16));
}

static float kvarn_load_fp16(const uint8_t * record, size_t offset, int index) {
    ggml_fp16_t fp16;
    memcpy(&fp16, record + offset + (size_t) index * sizeof(fp16), sizeof(fp16));
    return ggml_fp16_to_fp32(fp16);
}

/* ---------- tile quantize / dequantize ----------
 *
 * After balancing, each row of the balanced tile is uniformly quantized to
 * `bits` unsigned levels via its own min/max. The per-row scale/zero-point
 * (of the BALANCED tile) and the per-row Sinkhorn factor s_row are stored as
 * THREE SEPARATE fp16 values (not pre-multiplied together):
 *   tile[r,c] ~= s_row[r] * (q[r,c] * scale[r] + zp[r]) * s_col[c]
 * which reconstructs balanced[r,c] * s_row[r] * s_col[c] == tile[r,c].
 *
 * s_row/s_col come from kvarn_variance_normalize's log-space clamp of
 * [-0.3, 10.0], i.e. up to exp(10) ~= 22026 - safely representable in fp16
 * (max ~65504) on their own, but NOT once multiplied into scale/zp: real
 * model activations (unlike the bounded synthetic test tiles) can produce a
 * `scale` large enough that s_row[r]*scale overflows fp16 and silently
 * becomes +/-inf, corrupting the entire row (and, through attention, most of
 * the sequence). Keeping all three factors separate until dequant multiplies
 * them in fp32 avoids that overflow entirely, at the cost of one extra fp16
 * array per side (see kvarn_make_layout's k_row_amp_off/v_row_amp_off).
 */
static void kvarn_quantize_tile(
        const float * tile,
        int sinkhorn_iters,
        int bits,
        uint8_t * payload,
        size_t payload_bytes,
        uint8_t * record,
        size_t s_row_off,
        size_t zp_off,
        size_t s_col_off,
        size_t row_amp_off) {
    assert(tile != NULL);
    assert(kvarn_valid_bits(bits));

    float * balanced = (float *) malloc((size_t) KVARN_GROUP * KVARN_GROUP * sizeof(float));
    float s_col[KVARN_GROUP];
    float s_row[KVARN_GROUP];
    kvarn_variance_normalize(tile, sinkhorn_iters, balanced, s_col, s_row);

    uint8_t * q = (uint8_t *) malloc((size_t) KVARN_GROUP * KVARN_GROUP);
    const int qmax = (1 << bits) - 1;

    for (int r = 0; r < KVARN_GROUP; r++) {
        const float * row = balanced + (size_t) r * KVARN_GROUP;
        float lo = row[0], hi = row[0];
        for (int c = 1; c < KVARN_GROUP; c++) {
            if (row[c] < lo) lo = row[c];
            if (row[c] > hi) hi = row[c];
        }
        const float range = (hi - lo) / qmax;
        const float scale = range > 1e-10f ? range : 1e-10f;

        for (int c = 0; c < KVARN_GROUP; c++) {
            const float value = roundf((row[c] - lo) / scale);
            q[(size_t) r * KVARN_GROUP + c] = (uint8_t) kvarn_clampf(value, 0.0f, (float) qmax);
        }

        kvarn_store_fp16(record, s_row_off,   r, scale);
        kvarn_store_fp16(record, zp_off,      r, lo);
        kvarn_store_fp16(record, row_amp_off, r, s_row[r]);
    }

    for (int c = 0; c < KVARN_GROUP; c++) {
        kvarn_store_fp16(record, s_col_off, c, s_col[c]);
    }

    GGML_ASSERT(payload_bytes == kvarn_packed_bytes(KVARN_GROUP * KVARN_GROUP, bits));
    kvarn_pack_bits(q, KVARN_GROUP * KVARN_GROUP, bits, payload);

    free(q);
    free(balanced);
}

static void kvarn_dequantize_tile(
        const uint8_t * record,
        int bits,
        size_t payload_off,
        size_t s_row_off,
        size_t zp_off,
        size_t s_col_off,
        size_t row_amp_off,
        float * tile) {
    assert(record != NULL);
    assert(kvarn_valid_bits(bits));
    assert(tile != NULL);

    float s_col[KVARN_GROUP];
    for (int c = 0; c < KVARN_GROUP; c++) {
        s_col[c] = kvarn_load_fp16(record, s_col_off, c);
    }

    for (int r = 0; r < KVARN_GROUP; r++) {
        const float scale = kvarn_load_fp16(record, s_row_off, r);
        const float zp    = kvarn_load_fp16(record, zp_off, r);
        const float s_row = kvarn_load_fp16(record, row_amp_off, r);
        for (int c = 0; c < KVARN_GROUP; c++) {
            const uint8_t q = kvarn_unpack_bits_value(record + payload_off, r * KVARN_GROUP + c, bits);
            tile[r * KVARN_GROUP + c] = s_row * ((float) q * scale + zp) * s_col[c];
        }
    }
}

void kvarn_quantize_k_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record) {
    kvarn_quantize_tile(tile, sinkhorn_iters, bits,
            record + layout->k_payload_off, layout->k_payload_bytes, record,
            layout->k_s_row_off, layout->k_zp_off, layout->k_s_col_off, layout->k_row_amp_off);
}

void kvarn_quantize_v_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record) {
    kvarn_quantize_tile(tile, sinkhorn_iters, bits,
            record + layout->v_payload_off, layout->v_payload_bytes, record,
            layout->v_s_row_off, layout->v_zp_off, layout->v_s_col_off, layout->v_row_amp_off);
}

void kvarn_dequantize_k_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile) {
    kvarn_dequantize_tile(record, bits,
            layout->k_payload_off, layout->k_s_row_off, layout->k_zp_off, layout->k_s_col_off,
            layout->k_row_amp_off, tile);
}

void kvarn_dequantize_v_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile) {
    kvarn_dequantize_tile(record, bits,
            layout->v_payload_off, layout->v_s_row_off, layout->v_zp_off, layout->v_s_col_off,
            layout->v_row_amp_off, tile);
}
