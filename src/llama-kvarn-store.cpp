#include "llama-kvarn-store.h"

#include <algorithm>
#include <cassert>
#include <cstring>

llama_kvarn_layer_store::llama_kvarn_layer_store(int key_bits, int value_bits, int sinkhorn_iters)
    : key_bits_(key_bits), value_bits_(value_bits), sinkhorn_iters_(sinkhorn_iters) {
    layout_ = kvarn_make_layout(key_bits, value_bits);
    tail_k_.reserve((size_t) GROUP * HEAD_DIM);
    tail_v_.reserve((size_t) GROUP * HEAD_DIM);
}

uint32_t llama_kvarn_layer_store::append(const float * k_rot, const float * v_rot) {
    const uint32_t idx = n_total_;

    tail_k_.resize((size_t) (tail_count_ + 1) * HEAD_DIM);
    tail_v_.resize((size_t) (tail_count_ + 1) * HEAD_DIM);
    memcpy(tail_k_.data() + (size_t) tail_count_ * HEAD_DIM, k_rot, HEAD_DIM * sizeof(float));
    memcpy(tail_v_.data() + (size_t) tail_count_ * HEAD_DIM, v_rot, HEAD_DIM * sizeof(float));

    tail_count_++;
    n_total_++;

    if (tail_count_ == GROUP) {
        seal_group();
    }

    return idx;
}

void llama_kvarn_layer_store::seal_group() {
    assert(tail_count_ == GROUP);

    const size_t off = (size_t) n_sealed_ * layout_.tile_bytes;
    sealed_.resize(off + layout_.tile_bytes);
    uint8_t * record = sealed_.data() + off;

    kvarn_quantize_k_tile(tail_k_.data(), sinkhorn_iters_, key_bits_,   &layout_, record);
    kvarn_quantize_v_tile(tail_v_.data(), sinkhorn_iters_, value_bits_, &layout_, record);

    n_sealed_++;
    tail_count_ = 0;
    tail_k_.clear();
    tail_v_.clear();
}

void llama_kvarn_layer_store::read(uint32_t n, float * out_k, float * out_v) const {
    assert(n <= n_total_);

    const uint32_t sealed_tokens = n_sealed_ * GROUP;

    float tile_k[GROUP * HEAD_DIM];
    float tile_v[GROUP * HEAD_DIM];

    uint32_t written = 0;
    for (uint32_t g = 0; written < n && g < n_sealed_; g++) {
        const uint8_t * record = sealed_.data() + (size_t) g * layout_.tile_bytes;
        kvarn_dequantize_k_tile(record, key_bits_,   &layout_, tile_k);
        kvarn_dequantize_v_tile(record, value_bits_, &layout_, tile_v);

        const uint32_t base = g * GROUP;
        const uint32_t take = std::min<uint32_t>(GROUP, n - base);
        memcpy(out_k + (size_t) base * HEAD_DIM, tile_k, (size_t) take * HEAD_DIM * sizeof(float));
        memcpy(out_v + (size_t) base * HEAD_DIM, tile_v, (size_t) take * HEAD_DIM * sizeof(float));
        written = base + take;
    }

    if (n > sealed_tokens) {
        const uint32_t from_tail = n - sealed_tokens;
        memcpy(out_k + (size_t) sealed_tokens * HEAD_DIM, tail_k_.data(), (size_t) from_tail * HEAD_DIM * sizeof(float));
        memcpy(out_v + (size_t) sealed_tokens * HEAD_DIM, tail_v_.data(), (size_t) from_tail * HEAD_DIM * sizeof(float));
    }
}
