// Manual regression check for the KVarN graph-reuse fix (see the project's
// KVarN graph-reuse plan doc for full context). NOT wired into CMakeLists/
// ctest: it needs a real, multi-layer, multi-head GGUF model on disk to be
// meaningful - the bug this guards against (llama_context::process_ubatch's
// graph-reuse optimization replaying a stale destination offset from an
// earlier position against new input data) was invisible in every
// synthetic-model test this project used elsewhere, and only reproduced at
// real (36-layer, n_head_kv=8) scale. Adding proper model-loading test
// infrastructure to the automated suite is valuable future work, out of
// scope for this fix.
//
// What it checks: decodes the SAME prompt into the SAME growing KVarN cache
// via two DIFFERENT chunkings (e.g. one big 4096-token ubatch vs. 32
// separate 128-token ubatches). Since KVarN's group-sealing boundaries are
// always at absolute multiples of 128 regardless of how the caller chunked
// the prompt, both chunkings should produce numerically equivalent final
// logits - a real, measurable divergence (cosine well below ~0.999, scaled
// for the bit-width being tested) means the reused graph replayed a stale
// destination offset or stale content-length value from an earlier call.
//
// Build (from the project's build/ directory, after a normal cmake build):
//   g++ -O2 -std=c++17 -I../include -I../ggml/include \
//       ../tests/manual-kvarn-graph-reuse-check.cpp \
//       -Lbin -lllama -lggml -lggml-base -o /tmp/kvarn-graph-reuse-check \
//       -Wl,-rpath,$PWD/bin
//
// Run:
//   /tmp/kvarn-graph-reuse-check /path/to/real-multilayer-model.gguf \
//       <kvarn_bits> <n_prompt> <chunkA> [chunkB]
//
// Example (matches the numbers recorded in the graph-reuse plan doc):
//   /tmp/kvarn-graph-reuse-check ~/models/Qwen3-8B.Q5_K_M.gguf 4 512 512 128
//   -> expect cosine > 0.999 (PASS)
//   /tmp/kvarn-graph-reuse-check ~/models/Qwen3-8B.Q5_K_M.gguf 4 2048 2048 128
//   -> expect cosine ~0.9963 - this is a known, pre-existing, scale-
//      dependent floating-point characteristic (verified via `git stash` to
//      be bit-identical to the pre-graph-reuse-fix code at this exact
//      scale/config - see the plan doc's Phase 1 Session Log entry), NOT a
//      sign of the graph-reuse bug re-appearing. Only investigate a NEW
//      divergence if a config that used to pass at cosine > 0.999 starts
//      failing, or if a "clean" config (small n, no depth stress) suddenly
//      drops to near-zero or negative cosine - that shape (not a gradual
//      float-noise dip) is what the original bug looked like
//      (-0.41 to -0.51 cosine, not 0.996).

#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

static llama_context * make_ctx(llama_model * model, int32_t kvarn_bits, uint32_t n_ctx) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx    = n_ctx;
    params.n_batch  = n_ctx;
    params.n_ubatch = n_ctx;
    params.n_seq_max = 1;
    params.kv_unified = true;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    if (kvarn_bits > 0) {
        params.kvarn_key_bits   = kvarn_bits;
        params.kvarn_value_bits = kvarn_bits;
    }
    return llama_init_from_model(model, params);
}

static bool decode_batch(llama_context * ctx, const std::vector<llama_token> & toks, llama_pos pos0) {
    llama_batch batch = llama_batch_init((int) toks.size(), 0, 1);
    for (size_t i = 0; i < toks.size(); i++) {
        batch.token[batch.n_tokens] = toks[i];
        batch.pos[batch.n_tokens] = pos0 + (llama_pos) i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i + 1 == toks.size());
        batch.n_tokens++;
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static bool run_chunked(llama_model * model, int32_t kvarn_bits, uint32_t n_ctx,
        const std::vector<llama_token> & prompt, int chunk, std::vector<float> & out_logits) {
    llama_context * ctx = make_ctx(model, kvarn_bits, n_ctx);
    if (!ctx) return false;
    const int n_prompt = (int) prompt.size();
    for (int off = 0; off < n_prompt; off += chunk) {
        const int n = std::min(chunk, n_prompt - off);
        std::vector<llama_token> c(prompt.begin() + off, prompt.begin() + off + n);
        if (!decode_batch(ctx, c, off)) { fprintf(stderr, "decode failed at offset %d chunk %d\n", off, chunk); llama_free(ctx); return false; }
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * logits = llama_get_logits_ith(ctx, -1);
    out_logits.assign(logits, logits + n_vocab);
    llama_free(ctx);
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.gguf kvarn_bits n_prompt chunkA [chunkB]\n", argv[0]); return 1; }
    const int32_t kvarn_bits = atoi(argv[2]);
    const int n_prompt = atoi(argv[3]);
    const int chunkA = atoi(argv[4]);
    const int chunkB = argc > 5 ? atoi(argv[5]) : 128;

    ggml_backend_load_all();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::vector<llama_token> prompt(n_prompt);
    for (int i = 0; i < n_prompt; i++) prompt[i] = 1 + (i * 7919) % (n_vocab - 1);

    const uint32_t n_ctx = (((uint32_t) n_prompt + 128) / 128) * 128;

    std::vector<float> logitsA, logitsB;
    if (!run_chunked(model, kvarn_bits, n_ctx, prompt, chunkA, logitsA)) return 1;
    if (!run_chunked(model, kvarn_bits, n_ctx, prompt, chunkB, logitsB)) return 1;

    double dot = 0.0, na = 0.0, nb = 0.0, max_abs_diff = 0.0;
    for (int i = 0; i < n_vocab; i++) {
        const double a = logitsA[i], b = logitsB[i];
        dot += a * b; na += a * a; nb += b * b;
        max_abs_diff = std::max(max_abs_diff, std::fabs(a - b));
    }
    const double cosv = (na > 0.0 && nb > 0.0) ? dot / (std::sqrt(na) * std::sqrt(nb)) : 0.0;

    printf("kvarn_bits=%d n_prompt=%d chunkA=%d chunkB=%d: cosine=%.6f max_abs_diff=%.6f\n",
            kvarn_bits, n_prompt, chunkA, chunkB, cosv, max_abs_diff);
    printf(cosv > 0.999 ? "=== PASS ===\n" : "=== FAIL (diverges - see the doc comment above for how to tell a real regression from known float noise) ===\n");

    llama_model_free(model);
    return cosv > 0.999 ? 0 : 1;
}
