/* KVarN CPU codec round-trip test: Hadamard self-inverse property and
 * per-tile quantize/dequantize quality at the three shipped bit-widths.
 * Mirrors the shape of test-turbo-quant.c. */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* Mirrors ggml/src/ggml-kvarn-quant.c; kept in sync manually like
 * test-turbo-quant.c's extern declarations for the turbo codec. */
struct kvarn_tile_layout {
    size_t k_payload_off;
    size_t v_payload_off;
    size_t k_s_row_off;
    size_t k_zp_off;
    size_t k_s_col_off;
    size_t k_row_amp_off;
    size_t v_s_row_off;
    size_t v_zp_off;
    size_t v_s_col_off;
    size_t v_row_amp_off;

    size_t k_payload_bytes;
    size_t v_payload_bytes;
    size_t tile_bytes;
};

extern void kvarn_hadamard_128(float * values);
extern struct kvarn_tile_layout kvarn_make_layout(int key_bits, int value_bits);
extern void kvarn_quantize_k_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record);
extern void kvarn_dequantize_k_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile);

#define GROUP 128
#define N (GROUP * GROUP)

/* Synthetic tile with per-row and per-column scale outliers, the exact shape
 * the dual-axis balancing step exists to correct. */
static void make_synthetic_tile(float * tile, unsigned seed) {
    float row_scale[GROUP], col_scale[GROUP];
    unsigned s = seed;
    for (int i = 0; i < GROUP; i++) {
        s = s * 1103515245u + 12345u;
        row_scale[i] = 1.0f + 20.0f * ((s >> 16) & 0xff) / 255.0f;
        s = s * 1103515245u + 12345u;
        col_scale[i] = 1.0f + 20.0f * ((s >> 16) & 0xff) / 255.0f;
    }
    for (int r = 0; r < GROUP; r++) {
        for (int c = 0; c < GROUP; c++) {
            const float base = sinf(r * 0.11f + c * 0.037f) + 0.3f * cosf(r * 0.05f - c * 0.19f);
            tile[r * GROUP + c] = base * row_scale[r] * col_scale[c];
        }
    }
}

static void metrics(const float * a, const float * b, int n, double * mse, double * cosv) {
    double se = 0.0, dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n; i++) {
        const double d = (double) a[i] - (double) b[i];
        se  += d * d;
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    *mse  = se / n;
    *cosv = (na > 0.0 && nb > 0.0) ? dot / (sqrt(na) * sqrt(nb)) : 0.0;
}

static int test_hadamard_self_inverse(void) {
    float x[GROUP], y[GROUP];
    unsigned s = 7;
    for (int i = 0; i < GROUP; i++) {
        s = s * 1103515245u + 12345u;
        x[i] = ((float) ((s >> 8) & 0xffff) / 65535.0f - 0.5f) * 10.0f;
    }
    memcpy(y, x, sizeof(x));
    kvarn_hadamard_128(y);
    kvarn_hadamard_128(y);

    double mse, cosv;
    metrics(x, y, GROUP, &mse, &cosv);
    printf("Hadamard self-inverse: MSE=%.10g Cosine=%.8f\n", mse, cosv);
    return mse < 1e-6 ? 0 : 1;
}

static int test_tile_round_trip(int bits) {
    struct kvarn_tile_layout layout = kvarn_make_layout(bits, bits);

    float * tile   = malloc(N * sizeof(float));
    float * out    = malloc(N * sizeof(float));
    uint8_t * record = malloc(layout.tile_bytes);

    make_synthetic_tile(tile, 12345u + (unsigned) bits);

    /* Per-token Hadamard rotation, as applied at cache-write time before the
     * group is assembled into a tile (rows = tokens, 128 channels each). */
    float * rotated = malloc(N * sizeof(float));
    memcpy(rotated, tile, N * sizeof(float));
    for (int r = 0; r < GROUP; r++) {
        kvarn_hadamard_128(rotated + (size_t) r * GROUP);
    }

    kvarn_quantize_k_tile(rotated, 16, bits, &layout, record);
    kvarn_dequantize_k_tile(record, bits, &layout, out);

    /* Undo the per-token rotation (self-inverse) before comparing against
     * the original, pre-rotation tile. */
    for (int r = 0; r < GROUP; r++) {
        kvarn_hadamard_128(out + (size_t) r * GROUP);
    }

    double mse, cosv;
    metrics(tile, out, N, &mse, &cosv);
    printf("kvarn%d tile round-trip: layout=%zu bytes (payload k=%zu v=%zu) MSE=%.6g Cosine=%.6f\n",
            bits, layout.tile_bytes, layout.k_payload_bytes, layout.v_payload_bytes, mse, cosv);

    free(tile);
    free(out);
    free(rotated);
    free(record);

    /* Loose sanity floor -- catches a broken codec (near-zero cosine), not a
     * precision regression. Lower bits has less headroom; thresholds set with
     * margin below this synthetic tile's measured cosine at each width
     * (2=0.940, 3=0.989, 4=0.998, 5=0.999, 6=0.9999). */
    const double min_cosine =
        bits >= 6 ? 0.99 :
        bits >= 5 ? 0.97 :
        bits >= 4 ? 0.9  :
        bits >= 3 ? 0.95 : 0.85;
    return cosv >= min_cosine ? 0 : 1;
}

int main(void) {
    printf("=== KVarN CPU Codec Round-Trip Test ===\n\n");

    int failures = 0;
    failures += test_hadamard_self_inverse();
    printf("\n");

    const int bit_widths[] = { 2, 3, 4, 5, 6 };
    for (size_t i = 0; i < sizeof(bit_widths) / sizeof(bit_widths[0]); i++) {
        failures += test_tile_round_trip(bit_widths[i]);
    }

    printf("\n");
    if (failures) {
        printf("=== FAILED: %d check(s) ===\n", failures);
        return 1;
    }

    printf("=== Done ===\n");
    return 0;
}
