#pragma once

#include <cstdint>
#include <string>

// Feature gate for experimental block-granular KV cache streaming: the
// authoritative KV cache lives in a pinned host staging area while a bounded
// device pool holds a runtime-adjusted split of resident pages plus a small
// async transfer ring. Streaming is opt-in (stage_bytes == 0 disables it) and
// only valid on cache shapes the region/plan math below actually supports.
struct llama_kv_stream_config {
    uint64_t stage_bytes         = 0;
    uint64_t minimum_stage_bytes = 0;

    // True when the active memory is a plain unified llama_kv_cache (or the
    // attention component of a hybrid recurrent+attention cache) rather than
    // a specialized attention cache (DSA/DSV4 MLA-style, MSA indexer, SWA
    // dual-cache) whose region layout this planner does not yet model.
    bool unified_kv_cache = false;
    bool context_default  = false;
    bool single_sequence  = false;
    bool flash_attention  = false;
    bool kv_offload       = false;

    // True when the active (K, V) cache type pair has a backend-reported
    // streamed FlashAttention implementation, for whichever types those are -
    // not restricted to one hardcoded pair.
    bool type_pair_supported = false;
};

struct llama_kv_stream_config_result {
    bool valid   = false;
    bool enabled = false;
    std::string error;
};

llama_kv_stream_config_result llama_kv_stream_config_validate(const llama_kv_stream_config & config);

struct llama_kv_stream_pool_layout_params {
    uint64_t pool_bytes = 0;
    uint64_t page_bytes = 0;
    uint32_t layer_count = 0;
    uint32_t scratch_pages = 0;
};

struct llama_kv_stream_pool_layout {
    bool valid = false;
    std::string error;

    uint32_t resident_pages_per_layer  = 0;
    uint32_t resident_tokens_per_layer = 0;
    uint64_t scratch_bytes  = 0;
    uint64_t resident_bytes = 0;
    uint64_t unused_bytes   = 0;
};

llama_kv_stream_pool_layout llama_kv_stream_pool_layout_make(
    const llama_kv_stream_pool_layout_params & params);
