// Shared AnchorKV FA state (header-only inline functions).
//
// The KV cache sets the current layer's compressed GPU data pointers via
// these inline functions. The FA dispatch reads them via anchor_kv_fa_enabled().
//
// This is shared by anchor-kv-decompress.cu (writer) and anchor-kv-fa.cu (reader).

#pragma once

#include "anchor-kv-common.cuh"

// Static state (one instance across the whole library)
inline anchor_kv_data_t & anchor_kv_fa_state_ref() {
    static anchor_kv_data_t s_state = {};
    return s_state;
}

inline int & anchor_kv_fa_enabled_ref() {
    static int s_flag = 0;
    return s_flag;
}

// Writer API (called by KV cache)
inline void anchor_kv_fa_set_state(const anchor_kv_data_t & data) {
    anchor_kv_fa_state_ref() = data;
    anchor_kv_fa_enabled_ref() = 1;
}

inline void anchor_kv_fa_clear_state() {
    anchor_kv_fa_enabled_ref() = 0;
}

// Reader API (called by FA dispatch)
inline bool anchor_kv_fa_is_enabled() {
    return anchor_kv_fa_enabled_ref() != 0;
}

inline const anchor_kv_data_t & anchor_kv_fa_read_state() {
    return anchor_kv_fa_state_ref();
}
