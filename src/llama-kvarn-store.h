#pragma once

// Standalone (single-sequence, head_dim=128) KVarN group-staging store: the
// "recent tokens stay exact F16 until a 128-token group completes, then it
// seals into a compressed tile" state machine that the KVarN cache-type
// design needs on top of the plain per-row TurboQuant model. Deliberately
// decoupled from llama_kv_cache's cell/stream/defrag bookkeeping so the core
// mechanism can be built and tested in isolation before it is wired into the
// real cache (see AGENTS.md: prefer proving a mechanism standalone before
// integrating it into code that must stay stable).
//
// Multi-sequence support, arbitrary head_dim (128/256/512 via multiple
// slices), and the permanent attention sink are follow-up work once this
// core mechanism is validated.

#include <cstdint>
#include <vector>

extern "C" {

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

struct kvarn_tile_layout kvarn_make_layout(int key_bits, int value_bits);

void kvarn_hadamard_128(float * values);

void kvarn_quantize_k_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record);
void kvarn_quantize_v_tile(const float * tile, int sinkhorn_iters, int bits,
        const struct kvarn_tile_layout * layout, uint8_t * record);
void kvarn_dequantize_k_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile);
void kvarn_dequantize_v_tile(const uint8_t * record, int bits,
        const struct kvarn_tile_layout * layout, float * tile);

} // extern "C"

class llama_kvarn_layer_store {
public:
    static constexpr int HEAD_DIM = 128;
    static constexpr int GROUP    = 128;

    llama_kvarn_layer_store(int key_bits, int value_bits, int sinkhorn_iters = 16);

    // Appends one token's post-RoPE, post-Hadamard-rotated K and V vectors
    // (HEAD_DIM floats each, already rotated by the caller). Returns the
    // token's 0-based index. May seal a group (quantize the just-completed
    // 128 tokens) as a side effect.
    uint32_t append(const float * k_rot, const float * v_rot);

    uint32_t size() const { return n_total_; }

    // Materializes tokens [0, n) into `out_k`/`out_v` (n * HEAD_DIM floats
    // each), still in the rotated domain (caller applies the inverse
    // Hadamard rotation, matching this fork's existing turbo convention).
    // Reads sealed tiles where available, the live tail otherwise.
    void read(uint32_t n, float * out_k, float * out_v) const;

private:
    void seal_group();

    int key_bits_;
    int value_bits_;
    int sinkhorn_iters_;
    kvarn_tile_layout layout_;

    std::vector<float> tail_k_; // up to GROUP*HEAD_DIM floats, exact
    std::vector<float> tail_v_;
    uint32_t tail_count_ = 0;

    std::vector<uint8_t> sealed_; // n_sealed_ * layout_.tile_bytes
    uint32_t n_sealed_ = 0;

    uint32_t n_total_ = 0;
};
