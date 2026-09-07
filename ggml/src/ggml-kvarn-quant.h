#pragma once

// Public API of ggml-kvarn-quant.c: the per-tile KVarN codec (Hadamard
// rotation, dual-axis variance balancing, bit-packed quantize/dequantize).
// See ggml-kvarn-quant.c for the algorithm's origin and adaptation notes.

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct kvarn_tile_layout {
    size_t k_payload_off;
    size_t v_payload_off;
    size_t k_s_row_off;   // per-row quantization scale (of the BALANCED tile - not combined with k_row_amp)
    size_t k_zp_off;      // per-row zero point (of the BALANCED tile - not combined with k_row_amp)
    size_t k_s_col_off;   // per-column Sinkhorn factor
    size_t k_row_amp_off; // per-row Sinkhorn factor, stored separately (see ggml-kvarn-quant.c)
    size_t v_s_row_off;
    size_t v_zp_off;
    size_t v_s_col_off;
    size_t v_row_amp_off;

    size_t k_payload_bytes;
    size_t v_payload_bytes;
    size_t tile_bytes;
};

struct kvarn_tile_layout kvarn_make_layout(int key_bits, int value_bits);

size_t  kvarn_packed_bytes(int n_values, int bits);
void    kvarn_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst);
uint8_t kvarn_unpack_bits_value(const uint8_t * src, int index, int bits);

void kvarn_hadamard_128(float * values);

void kvarn_quantize_k_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record);
void kvarn_quantize_v_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record);
void kvarn_dequantize_k_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile);
void kvarn_dequantize_v_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile);

#ifdef __cplusplus
}
#endif
