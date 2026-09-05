#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-kv-cells.h"
#include "llama-memory.h"
#include "anchor-kv.h"

#include <unordered_map>
#include <vector>

struct llama_cparams;
struct llama_hparams;
struct llama_model;
struct llama_context;

//
// llama_kv_cache
//

class llama_kv_cache : public llama_memory_i {
public:
    struct stream_copy_info {
        bool empty() const {
            assert(ssrc.size() == sdst.size());
            return ssrc.empty();
        }

        std::vector<uint32_t> ssrc;
        std::vector<uint32_t> sdst;
    };

    // for each ubatch, create a slot_info that contains information about where the ubatch should be inserted in the
    //   KV cells. for example, cell indices for each token, such that: token[i] -> goes to cells[idxs[i]]
    struct slot_info {
        // data for ggml_set_rows
        using idx_vec_t = std::vector<uint32_t>;

        // number of streams: ns = s1 - s0 + 1
        uint32_t s0;
        uint32_t s1;

        std::vector<llama_seq_id> strm; // [ns]
        std::vector<idx_vec_t>    idxs; // [ns]

        uint32_t head() const {
            GGML_ASSERT(idxs.size() == 1);
            GGML_ASSERT(!idxs[0].empty());

            return idxs[0][0];
        }

        void resize(size_t n) {
            strm.resize(n);
            idxs.resize(n);
        }

        size_t size() const {
            GGML_ASSERT(idxs.size() == strm.size());
            GGML_ASSERT(!idxs.empty());

            return idxs[0].size();
        }

        size_t n_stream() const {
            return strm.size();
        }

        bool empty() const {
            return idxs.empty();
        }

        void clear() {
            idxs.clear();
        }

        // check if indices are contiguous starting from head()
        bool is_contiguous() const {
            if (idxs.empty() || idxs[0].empty()) {
                return true;
            }
            if (idxs.size() > 1) {
                return false;
            }
            const uint32_t h = idxs[0][0];
            for (size_t i = 0; i < idxs[0].size(); ++i) {
                if (idxs[0][i] != h + i) {
                    return false;
                }
            }
            return true;
        }
    };

    using slot_info_vec_t = std::vector<slot_info>;

    // TODO: refactor the memory instances to not depend on `llama_model`
    //       instead pass all necessary info (e.g. hparams, dev layers, arch, etc.) directly
    //       likely through `struct llama_memory_params`
    llama_kv_cache(
            const llama_model & model,
          const llama_hparams & hparams,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   v_trans,
                         bool   offload,
                         bool   unified,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
               llama_memory_t   mem_other,
        const layer_filter_cb & filter,
        const  layer_reuse_cb & reuse,
        const  layer_share_cb & share,
                 const char *   name_tag = "");

    ~llama_kv_cache() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    //
    // llama_kv_cache specific API
    //

    uint32_t get_size()     const;
    uint32_t get_n_stream() const;
    std::vector<uint32_t> get_layer_ids() const;
    ggml_tensor * get_k_storage(int32_t il) const;
    ggml_tensor * get_v_storage(int32_t il) const;
    bool get_v_transposed() const;

    bool get_has_shift() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // AnchorKV: RoPE needs the live cparams (freq_base/freq_scale overrides,
    // yarn settings) for both the pre-compression inverse rotation and the
    // in-graph reconstruction rotation. llama_kv_cache is constructed without
    // a cparams reference (see the ctor above), so llama_context sets this
    // once right after creating the memory object (see llama-context.cpp,
    // just after model.create_memory()). cparams lives for the lifetime of
    // the owning llama_context, which always outlives this cache, so storing
    // the pointer is safe.
    void anchor_kv_set_cparams(const llama_cparams & cparams) { anchor_cparams = &cparams; }

    // AnchorKV: parse ANCHOR_KV_THETA(_K/_V) env vars into anchor_kv_enabled and
    // akv_params.compress_k/compress_v. Must run before the dense cache tensors
    // are allocated so K and V can be placed in separate buffers when only one
    // side is compressed.
    void anchor_kv_parse_env();

    // AnchorKV: compress the dense cache after prefill
    void anchor_kv_compress_all();
    bool get_anchor_kv_enabled() const { return anchor_kv_enabled; }
    bool get_anchor_kv_compressed() const { return !anchor_kv_data.empty(); }
    // ANCHOR_KV_DENSE_TEST diagnostic: compression still happened (anchor_kv_data
    // is populated, so these two stay true - re-trigger guards / state I/O /
    // shift-blocking are unaffected), but decode must NOT route through the
    // shared scratch buffer or the GGML_OP_ANCHOR_DECOMPRESS graph op - the
    // reconstruction was already written back into the dense tensors once.
    // See anchor_kv_parse_env / anchor_kv_compress_all for where this is set.
    bool get_anchor_kv_compressed_k() const { return !anchor_kv_data.empty() && akv_params.compress_k && !anchor_kv_dense_test; }
    bool get_anchor_kv_compressed_v() const { return !anchor_kv_data.empty() && akv_params.compress_v && !anchor_kv_dense_test; }
    const anchor_kv_layer * get_anchor_kv_layer(int32_t il) const;

    // AnchorKV: build the in-graph decompress node for one layer. The op writes
    // the reconstructed dense K (is_k=true) or V (is_k=false) into the shared
    // scratch buffer; the result is a view of that scratch.
    ggml_tensor * anchor_kv_build_decompress(ggml_context * ctx, int32_t il, bool is_k) const;

    // AnchorKV: shared dense scratch buffers (one layer of dense K/V, reused
    // across all layers so only a single layer's worth of memory is resident)
    ggml_tensor * get_anchor_scratch_k() const { return anchor_scratch_k; }
    ggml_tensor * get_anchor_scratch_v() const { return anchor_scratch_v; }

    // AnchorKV: reset the per-graph-build dependency chain (see anchor_chain_k /
    // anchor_chain_v below). Must be called exactly once before each new graph
    // is built, since the cached tensors belong to the previous graph's
    // ggml_context and must not be reused as src[] ancestors in a new one.
    // Called from llama_kv_cache_context::apply(), which always runs before
    // the graph for a ubatch is constructed.
    void anchor_kv_reset_chain() const { anchor_chain_k = nullptr; anchor_chain_v = nullptr; }

    const llama_kv_cells & get_cells(llama_seq_id seq_id) const;

    // state_read, plus the cells the restored tokens were placed in.
    // a cache that mirrors another one cell for cell (the qwen4exp indexer) cannot search for
    // its own cells here: a second independent search only happens to agree with the first.
    //   sinfos_out: if set, resized to n_stream and filled with the layout used; a stream that
    //               carried no cells leaves an empty entry
    //   sinfos_in : if set, the layout to use instead of searching for one. it must have one
    //               entry per stream and the entry must match the cell count in the blob,
    //               otherwise the read fails as it would on any other corrupt input
    void state_read_sinfo(
            llama_io_read_i & io,
               llama_seq_id   seq_id,
      llama_state_seq_flags   flags,
          slot_info_vec_t *   sinfos_out,
     const slot_info_vec_t *   sinfos_in);

    //
    // graph_build API
    //

    uint32_t get_n_kv(const slot_info & sinfo) const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const;

    // TurboQuant: get rotation matrices (stored as row-major C arrays)
    // turbo_rotation = R (forward rotation, for Q pre-rotate-queries)
    // turbo_rotation_inv = R^T = R^{-1} (inverse rotation, for V output un-rotation)
    ggml_tensor * get_turbo_rotation() const { return turbo_rotation; }
    ggml_tensor * get_turbo_rotation_inv() const { return turbo_rotation_inv; }

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const { return turbo_innerq_scale_inv; }

    // store k_cur and v_cur in the cache based on the provided head location
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const;

    //
    // preparation API
    //

    // find places for the provided ubatches in the cache, returns the slot infos
    // return empty vector on failure
    slot_info_vec_t prepare(const std::vector<llama_ubatch> & ubatches);

    bool update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info);

    // find a slot of kv cells that can hold the ubatch
    // if cont == true, then the slot must be continuous
    // return empty slot_info on failure
    slot_info find_slot(const llama_ubatch & ubatch, bool cont) const;

    // emplace the ubatch context into slot: [sinfo.idxs[0...ubatch.n_tokens - 1]]
    void apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch);

    //
    // input API
    //

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const;

    void set_input_k_shift(ggml_tensor * dst) const;

    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    const llama_model & model;
    const llama_hparams & hparams;

    // AnchorKV: set post-construction via anchor_kv_set_cparams() - see the
    // doc comment on that method for why this can't be a constructor param
    // or a plain reference member.
    const llama_cparams * anchor_cparams = nullptr;

    struct kv_layer {
        // layer index in the model
        // note: can be different from the layer index in the KV cache
        uint32_t il;

        ggml_tensor * k;
        ggml_tensor * v;

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;
    };

    bool v_trans = true;  // the value tensor is transposed

    const uint32_t n_seq_max = 1;
    const uint32_t n_stream  = 1;

    // required padding
    const uint32_t n_pad = 1;

    // SWA
    const uint32_t n_swa = 0;

    // env: LLAMA_ATTN_ROT_DISABLE
    bool attn_rot_k = false;
    bool attn_rot_v = false;

    // if all layers participating in the cache have constant head size, the value is stored here
    // otherwise the value is -1
    int32_t n_embd_head_k_all = 0;
    int32_t n_embd_head_v_all = 0;

    // pre-computed hadamard martrices
    std::unordered_map<int64_t, std::vector<float>> attn_rot_hadamard;

    // env: LLAMA_KV_CACHE_DEBUG
    int debug = 0;

    // this is the SWA type of the cache - not to be confused with the model SWA type
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    // ggml contexts for the KV cache along with the allocated backend buffers.
    // side is 0 for K (or the shared K+V buffer when AnchorKV is off) and 1 for
    // a separate V buffer (AnchorKV on), so a single side can be freed after
    // compression while the other stays dense.
    struct kv_ctx_buf {
        ggml_context_ptr      ctx;
        ggml_backend_buffer_ptr buf;
        int side = 0;
    };
    std::vector<kv_ctx_buf> ctxs_bufs;

    // the current index from where we start searching for a free slot in the ring buffer of KV cells (see find_slot())
    // note: this is not part of the KV state and it's only used to speed-up the find_slot() method
    std::vector<uint32_t> v_heads;

    // TODO: temporary until we refactor to be able to share the same cells between 2 kv caches [TAG_KV_CACHE_SHARE_CELLS]
    llama_kv_cache * other;

    std::shared_ptr<llama_kv_cells_vec> v_cells_impl;

    llama_kv_cells_vec & v_cells;

    // maps from a sequence id to a stream id
    std::vector<uint32_t> seq_to_stream;

    // pending stream copies that will be applied during the next update
    stream_copy_info sc_info;

    std::vector<kv_layer> layers;

    // TurboQuant rotation matrices (128x128, row-major stored)
    ggml_tensor * turbo_rotation = nullptr;      // R (forward rotation)
    ggml_tensor * turbo_rotation_inv = nullptr;   // R^T = R^{-1} (inverse rotation)

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization (128 floats)
    ggml_tensor * turbo_innerq_scale_inv = nullptr;

    // AnchorKV: per-layer compressed representation (populated after prefill)
    bool anchor_kv_enabled = false;
    // ANCHOR_KV_DENSE_TEST diagnostic flag - see get_anchor_kv_compressed_k/v
    // and anchor_kv_compress_all for what this changes.
    bool anchor_kv_dense_test = false;
    struct anchor_kv_params akv_params;
    std::vector<anchor_kv_layer> anchor_kv_data;  // one per cache layer

    // AnchorKV GPU decompression buffers (per layer)
    struct anchor_kv_gpu {
        ggml_context_ptr ctx;                    // context owning the tensors
        ggml_backend_buffer_ptr buf;             // backing buffer (device or CPU)
        ggml_tensor * anchors      = nullptr;    // [n_heads, 2, k, D] bf16
        ggml_tensor * anchor_of    = nullptr;    // [2, n_heads, S] i32
        ggml_tensor * gamma        = nullptr;    // [2, n_heads, S] f16
        ggml_tensor * slot_of      = nullptr;    // [2, n_heads, S] i32
        ggml_tensor * k_res_codes  = nullptr;    // [n_heads, n_K, D/4] u8
        ggml_tensor * k_res_scales = nullptr;    // [n_heads, n_K] f32
        ggml_tensor * v_res_codes  = nullptr;    // [n_heads, n_V, D/4] u8
        ggml_tensor * v_res_scales = nullptr;    // [n_heads, n_V] f32
    };
    std::vector<anchor_kv_gpu> anchor_kv_gpu_data;  // one per layer

    // AnchorKV: shared dense scratch buffers + their context/buffer
    ggml_context_ptr anchor_ctx;
    ggml_backend_buffer_ptr anchor_buf;
    ggml_tensor * anchor_scratch_k = nullptr;    // [n_embd_k_gqa, kv_size] f16
    ggml_tensor * anchor_scratch_v = nullptr;    // [n_embd_v_gqa, kv_size] f16

    // AnchorKV: per-graph-build chain state. Holds the most recently built
    // tensor that touched anchor_scratch_k / anchor_scratch_v (decompress result,
    // then cpy_k/cpy_v result, then get_k/get_v result, then next layer's
    // decompress result, ...). anchor_kv_build_decompress/get_k/get_v/cpy_k/cpy_v
    // read this as the base for their view/op instead of the raw scratch tensor,
    // and write their own result back into it. This threads a REAL src[]/view_src
    // ancestry chain through every reader and writer of the shared scratch buffer,
    // across all layers, so ggml's topological order (not incidental call order)
    // enforces layer-il decompress -> cpy -> get -> layer-(il+1) decompress -> ...
    // Reset once per graph build via anchor_kv_reset_chain() (see above); must not
    // be dereferenced or reused after the ggml_context it was built in is gone.
    mutable ggml_tensor * anchor_chain_k = nullptr;
    mutable ggml_tensor * anchor_chain_v = nullptr;

    // AnchorKV: build the per-layer GPU tensors for the in-graph decompress op
    void anchor_kv_upload_layer(int32_t ikv);

    // AnchorKV: free the dense per-layer KV buffers after compression
    void anchor_kv_free_dense();

    // AnchorKV diagnostic (ANCHOR_KV_GRAPH_DIFF=<layer> env var): runs the real
    // GGML_OP_ANCHOR_DECOMPRESS kernel standalone for one layer, on a throwaway
    // destination tensor, and diffs its output value-by-value against the
    // independent float32 CPU reference (anchor_kv_decompress_head + forward
    // RoPE - the same math ANCHOR_KV_DENSE_TEST writes into the dense cache).
    // Since the compressed representation and the shared-scratch decompress
    // output never change across decode steps for positions [0, S), this is
    // equivalent to diffing at the first decode step without needing to hook
    // into live graph execution. CPU-backend only (requires -ngl 0) - see
    // AnchorKV-status memory for why the graph-integration layer, not the
    // compression math, is the current suspect.
    void anchor_kv_debug_graph_diff(int32_t il);

    // AnchorKV live graph-diff capture, called by anchor_kv_debug_graph_diff:
    // installs a temporary ggml_backend_sched eval callback (via the live
    // llama_cparams this cache was given - see anchor_kv_set_cparams) that
    // fires the instant this layer's REAL decompress node computes during
    // the next graph build/compute - i.e. the actual first decode step after
    // compression - and diffs it against `ref_k`/`ref_v` before the next
    // layer's decompress overwrites the shared scratch buffer. No-ops (with
    // a warning) if a real cb_eval is already installed, so this never
    // clobbers a caller's own eval-callback tooling. Resets cb_eval to
    // nullptr once both sides have been captured.
    void anchor_kv_debug_install_live_capture(int32_t il, int n_heads, int D, std::vector<float> ref_k, std::vector<float> ref_v);
    static bool anchor_kv_debug_eval_cb(struct ggml_tensor * t, bool ask, void * user_data);

    // Fires once per real decode step for a given target tensor name -
    // ANCHOR_KV_GRAPH_DIFF_STEP (default 1) selects which occurrence actually
    // gets captured/logged (hits reaching that count), so step 2, 3, ... can
    // be checked for drift instead of only the first decode step. Earlier
    // occurrences are silently skipped (hits just increments).
    int                anchor_kv_diff_target_step = 1;
    std::string        anchor_kv_diff_name_k;
    std::string        anchor_kv_diff_name_v;
    std::vector<float> anchor_kv_diff_ref_k;
    std::vector<float> anchor_kv_diff_ref_v;
    int                anchor_kv_diff_hits_k = 0;
    int                anchor_kv_diff_hits_v = 0;

    // Same capture, but targeting get_k()/get_v()'s attention-read view AFTER
    // cpy_k()/cpy_v() has appended the newly decoded token past S - checks
    // that the append doesn't corrupt the already-verified [0, S) range, and
    // logs the new token's own row for a sanity look (no independent
    // reference exists for it, unlike the compressed range).
    std::string        anchor_kv_diff_name_get_k;
    std::string        anchor_kv_diff_name_get_v;
    int                anchor_kv_diff_hits_get_k = 0;
    int                anchor_kv_diff_hits_get_v = 0;

    // Reference-attention diagnostic, extending ANCHOR_KV_GRAPH_DIFF: once
    // decompress+append are already verified correct, also capture Q, the
    // real KQ mask, and the real FLASH_ATTN_EXT node/output for this layer,
    // then compute attention from scratch (softmax(QK^T*scale + mask)V with
    // GQA head-group broadcasting) using the SAME already-verified
    // get_k()/get_v() values, and diff the result against the real op's
    // output. The only way left to localize a bug in attention math itself,
    // since every value feeding into it has already been measured correct.
    std::string anchor_kv_diff_name_q;
    std::string anchor_kv_diff_name_fattn;
    int         anchor_kv_diff_hits_q     = 0;
    int         anchor_kv_diff_hits_fattn = 0;

    int anchor_kv_diff_n_heads_kv = 0;
    int anchor_kv_diff_head_dim   = 0;

    std::vector<float> anchor_kv_diff_captured_k; // [n_kv][n_heads_kv][D], from anchor_get_l<il>_k
    std::vector<float> anchor_kv_diff_captured_v; // [n_kv][n_heads_kv][D], from anchor_get_l<il>_v
    int64_t            anchor_kv_diff_captured_n_kv = 0;

    std::vector<float> anchor_kv_diff_captured_q; // [D_q][n_head_q][n_tokens], from attn_q_l<il>
    int64_t anchor_kv_diff_q_head_dim = 0;
    int64_t anchor_kv_diff_q_n_heads  = 0;
    int64_t anchor_kv_diff_q_n_tokens = 0;

    void anchor_kv_debug_reference_attn(const struct ggml_tensor * fattn_node);

    // AnchorKV: fetch this layer's RoPE params (freq_base/scale, n_rot, YaRN,
    // rope_factors) and rotate `keys` (dense [S_used, n_embd_k_gqa] float, in
    // place) - forward=false inverts (pre-compression), forward=true re-applies
    // (ANCHOR_KV_DENSE_TEST reconstruction writeback). Single source of truth
    // for both directions so they can't drift apart - see anchor_kv_invert_rope_k.
    void anchor_kv_apply_rope_to_dense_k(
            std::vector<float> & keys, uint32_t S_used, int n_embd_k_gqa,
            uint32_t head_k, int n_head_kv_k, uint32_t layer_il, bool forward);

    // model layer id -> KV cache layer id
    std::unordered_map<int32_t, int32_t> map_layer_ids;

    size_t total_size() const;

    size_t size_k_bytes() const;
    size_t size_v_bytes() const;

    ggml_tensor * build_rope_shift(
            const llama_cparams & cparams,
                   ggml_context * ctx,
                    ggml_tensor * cur,
                    ggml_tensor * shift,
                    ggml_tensor * rot,
                    ggml_tensor * factors,
                          float   freq_base,
                          float   freq_scale,
                       uint32_t   il) const;

    ggml_cgraph * build_graph_shift(
               llm_graph_result * res,
                  llama_context * lctx) const;

    struct cell_ranges_t {
        uint32_t strm;

        std::vector<std::pair<uint32_t, uint32_t>> data; // ranges, from inclusive, to exclusive
    };

    void state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const;

    // sinfo_in, when set, replaces the find_slot call: the cells are given by the caller
    bool state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count,       slot_info & sinfo, llama_seq_id dest_seq_id = -1, const slot_info * sinfo_in = nullptr);
    bool state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo);
};

class llama_kv_cache_context : public llama_memory_context_i {
public:
    // some shorthands
    using slot_info_vec_t  = llama_kv_cache::slot_info_vec_t;
    using stream_copy_info = llama_kv_cache::stream_copy_info;

    // used for errors
    llama_kv_cache_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_context(
            llama_kv_cache * kv);

    // used to create an update context
    llama_kv_cache_context(
            llama_kv_cache * kv,
            llama_context * lctx,
            bool do_shift,
            stream_copy_info sc_info);

    // used to create a batch processing context from a batch
    llama_kv_cache_context(
            llama_kv_cache * kv,
            slot_info_vec_t sinfos,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_context specific API
    //

    uint32_t get_n_kv() const;

    ggml_type type_k() const;
    ggml_type type_v() const;

    // get views of the current state of the cache
    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * get_v(ggml_context * ctx, int32_t il) const;

    // AnchorKV: true when decode graphs must read from the compressed
    // representation (compression happened) instead of the dense cache
    bool get_anchor_active() const override;

    // AnchorKV: per-side variants - true only when that specific side is
    // stored compressed (e.g. V-only compression keeps K dense)
    bool get_anchor_active_k() const;
    bool get_anchor_active_v() const;

    // AnchorKV: build the in-graph decompress node for one layer (expanded by
    // the caller before cpy_k/cpy_v so it executes first)
    ggml_tensor * build_anchor_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * build_anchor_v(ggml_context * ctx, int32_t il) const;

    // AnchorKV: shared dense scratch buffers
    ggml_tensor * get_anchor_scratch_k() const { return kv->get_anchor_scratch_k(); }
    ggml_tensor * get_anchor_scratch_v() const { return kv->get_anchor_scratch_v(); }

    // TurboQuant rotation accessors
    ggml_tensor * get_turbo_rotation() const;
    ggml_tensor * get_turbo_rotation_inv() const;

    // Override virtual methods from llama_memory_context_i
    ggml_tensor * get_turbo_rot_forward() const override;
    ggml_tensor * get_turbo_rot_inverse() const override;

    // TurboQuant InnerQ: per-channel scale_inv for Q/V equalization
    ggml_tensor * get_turbo_innerq_scale_inv() const override;

    // store k_cur and v_cur in the cache based on the provided head location
    // note: the heads in k_cur and v_cur should be laid out contiguously in memory
    //   - k_cur  [n_embd_head_k, n_head_k, n_tokens]
    //   - k_idxs [n_tokens]
    //   - v_cur  [n_embd_head_v, n_head_v, n_tokens]
    //   - v_idxs [n_tokens] or [n_tokens*n_embd_v_gqa] depending if V cache is transposed
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;
    ggml_tensor * cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const;

    // create destination indices for each head of the current batch for where it would be written in the KV cache
    // the indices address the global KV cache (not per stream) - this is not relevant for the user of this API, but
    //   helps understand the implementation logic of cpy_k and cpy_v
    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    ggml_tensor * build_input_v_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;
    void set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_shift   (ggml_tensor * dst) const;
    void set_input_kq_mask   (ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const;
    void set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const;

    void set_input_k_rot(ggml_tensor * dst) const;
    void set_input_v_rot(ggml_tensor * dst) const;

private:
    llama_memory_status status;

    llama_kv_cache * kv;
    llama_context * lctx;

    //
    // update context
    //

    bool do_shift = false;

    stream_copy_info sc_info;

    //
    // batch processing context
    //

    // the index of the cur ubatch to process
    size_t i_cur = 0;

    slot_info_vec_t sinfos;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    //

    // a heuristic, to avoid attending the full cache if it is not yet utilized
    // as the cache gets filled, the benefit from this heuristic disappears
    int32_t n_kv;
};
