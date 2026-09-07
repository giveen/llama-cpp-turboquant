#pragma once

// Fast, word-aligned bit-pack/unpack helpers for KVarN's arbitrary (2-6)
// bit-width payloads, replacing per-BIT extraction/insertion (looping
// `bits` times, one single-bit read/write each) with a single 32-bit-word
// read/write covering the whole value.
//
// Numerically identical to the bit-at-a-time version in every kernel this
// replaces - this only changes how many load/store instructions the
// extraction takes, never the value extracted/stored. Confirmed safe for
// KVarN's exact layout by direct computation, not just by analogy: each
// side's payload is exactly 16*bits*128 bytes (kvarn_make_layout), i.e.
// 128*bits words - always a WHOLE number of 4-byte words with zero slack,
// and 16*bits (each row's byte span) is itself always a multiple of 4. The
// last valid index's bit range therefore always ends exactly on the final
// word's last bit, so a value that straddles two words never reads past
// the buffer - verified for bits in {2,3,4,5,6} (the only values
// kvarn_valid_bits allows) by direct arithmetic, not assumed.
//
// Anbeeld/beellama.cpp's own KVarN fork (a different, from-scratch CUDA
// implementation of the same algorithm) independently arrived at this
// exact optimization for its own bit-unpacking hot loop, measuring ~1.13
// vs ~2 load instructions per element and ~40% less L1 traffic - cited
// here as the reason this was worth trying, not as something ported
// (the code itself is original, written for this fork's own layout).

__device__ __forceinline__ uint32_t kvarn_unpack_bits_fast(const uint8_t * payload, int64_t index, int bits) {
    const uint32_t * words = (const uint32_t *) payload;
    const int64_t bit_offset  = index * bits;
    const int64_t word_offset = bit_offset >> 5;
    const int      shift      = (int) (bit_offset & 31);
    uint64_t packed = (uint64_t) words[word_offset];
    if (shift + bits > 32) {
        packed |= (uint64_t) words[word_offset + 1] << 32;
    }
    return (uint32_t) ((packed >> shift) & (uint32_t) ((1u << bits) - 1u));
}

// Writes `value`'s low `bits` bits at `index` into `row_bytes` via
// OR-accumulation - safe WITHOUT atomics only because every call site
// using this has exactly one thread owning the entire `row_bytes` range
// (one thread packs one full 128-value row), so there is no cross-thread
// write to the same word. Caller must zero the row's bytes first (same
// requirement the bit-at-a-time version already had).
__device__ __forceinline__ void kvarn_pack_bits_fast(uint8_t * row_bytes, int index, int bits, uint32_t value) {
    uint32_t * words = (uint32_t *) row_bytes;
    const int bit_offset  = index * bits;
    const int word_offset = bit_offset >> 5;
    const int shift       = bit_offset & 31;
    const uint32_t v = value & (uint32_t) ((1u << bits) - 1u);
    words[word_offset] |= v << shift;
    if (shift + bits > 32) {
        words[word_offset + 1] |= v >> (32 - shift);
    }
}
