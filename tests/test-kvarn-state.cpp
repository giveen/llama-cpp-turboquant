// KVarN state serialization round-trip: decode a prompt long enough to seal
// a group (128 tokens) plus a partial tail, save the session file, generate
// reference tokens, restore, and require byte-identical greedy continuation.
// Exercises state_write_kvarn_data/state_read_kvarn_data (tensor bytes +
// per-tile counters) through the public session-file API.

#include "llama.h"

#include <cstdio>
#include <string>
#include <vector>

static bool decode_at(llama_context * ctx, llama_token * toks, int n, int pos0) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; i++) {
        batch.token[i]    = toks[i];
        batch.pos[i]      = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = true;
    }
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

static llama_token greedy_next(llama_context * ctx, llama_sampler * smpl) {
    return llama_sampler_sample(smpl, ctx, -1);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        // No requiring-model fixture exists for 128/256-dim heads in CI, so
        // without a model argument this is a no-op SKIP, not a failure.
        printf("SKIP: usage: %s MODEL [--bits N] [--prompt-tokens N]\n", argv[0]);
        return 0;
    }
    int bits = 4;
    int max_prompt = 160;
    for (int i = 2; i + 1 < argc; i++) {
        if (std::string(argv[i]) == "--bits") {
            bits = atoi(argv[i + 1]);
        }
        if (std::string(argv[i]) == "--prompt-tokens") {
            max_prompt = atoi(argv[i + 1]);
        }
    }

    printf("=== KVarN State Round-Trip Test (kvarn%d) ===\n\n", bits);

    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers       = 99;
    llama_model * model        = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        printf("FAIL: model load\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx                = 1024;
    cparams.n_batch              = 512;
    cparams.n_ubatch             = 512;
    cparams.n_seq_max            = 1;
    cparams.kv_unified           = true;
    cparams.kvarn_key_bits       = bits;
    cparams.kvarn_value_bits     = bits;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        printf("FAIL: context init\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // ~150-token prompt: 1 sealed group (128) + partial tail
    std::string prompt;
    for (int i = 0; i < 12; i++) {
        prompt += "The capital of France is Paris. The Seine flows through it. ";
    }
    std::vector<llama_token> toks(256);
    int                      n_prompt =
        llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), toks.data(), (int) toks.size(), true, true);
    if (n_prompt < 130) {
        printf("FAIL: prompt too short (%d tokens)\n", n_prompt);
        return 1;
    }
    if (n_prompt > max_prompt) {
        n_prompt = max_prompt;
    }
    printf("prompt tokens: %d\n", n_prompt);

    if (!decode_at(ctx, toks.data(), n_prompt, 0)) {
        printf("FAIL: prompt decode\n");
        return 1;
    }

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    int pos = n_prompt;
    // sample one token without decoding it yet (save-before-last protocol:
    // the file holds one more token than the cache has decoded)
    auto sample_one = [&](llama_token & out) {
        out = greedy_next(ctx, smpl);
    };
    auto gen = [&](int n, std::vector<llama_token> & out) {
        for (int i = 0; i < n; i++) {
            llama_token t = greedy_next(ctx, smpl);
            out.push_back(t);
            if (!decode_at(ctx, &t, 1, pos++)) {
                return false;
            }
        }
        return true;
    };

    const char *             path = "/tmp/test-kvarn-state.bin";
    std::vector<llama_token> back;
    int                      failures = 0;

    // decode prompt + 7, hold the 8th sampled token back: file will hold
    // 108 tokens with 107 decoded, mirroring common_prompt_batch_decode
    std::vector<llama_token> pre;
    if (!gen(7, pre)) {
        printf("FAIL: pre-save generation\n");
        return 1;
    }
    llama_token held = 0;
    sample_one(held);

    std::vector<llama_token> history(toks.begin(), toks.begin() + n_prompt);
    history.insert(history.end(), pre.begin(), pre.end());
    history.push_back(held);
    if (!llama_state_save_file(ctx, path, history.data(), history.size())) {
        printf("FAIL: state save\n");
        return 1;
    }
    printf("saved state: %zu tokens stored, %d decoded\n", history.size(), pos);

    // reference run: decode the held token, then 8 more
    std::vector<llama_token> ref;
    ref.push_back(held);
    if (!decode_at(ctx, &held, 1, pos++)) {
        printf("FAIL: reference held-token decode\n");
        return 1;
    }
    std::vector<llama_token> tail;
    if (!gen(8, tail)) {
        printf("FAIL: post-save generation\n");
        return 1;
    }

    std::vector<llama_token> restored_toks((size_t) n_prompt + 32);
    size_t n_restored = 0;
    if (!llama_state_load_file(ctx, path, restored_toks.data(), restored_toks.size(), &n_restored)) {
        printf("FAIL: state load\n");
        return 1;
    }
    // byte-level round-trip check: save again right after load, files must match
    const char * path2 = "/tmp/test-kvarn-state-2.bin";
    if (!llama_state_save_file(ctx, path2, history.data(), history.size())) {
        printf("FAIL: second state save\n");
        return 1;
    }
    {
        FILE * f1 = fopen(path, "rb");
        FILE * f2 = fopen(path2, "rb");
        fseek(f1, 0, SEEK_END);
        fseek(f2, 0, SEEK_END);
        const long s1 = ftell(f1), s2 = ftell(f2);
        bool same = s1 == s2;
        if (same) {
            rewind(f1);
            rewind(f2);
            for (long i = 0; same && i < s1; i++) {
                same = fgetc(f1) == fgetc(f2);
            }
        }
        fclose(f1);
        fclose(f2);
        printf("state bytes: %ld vs %ld -> %s\n", s1, s2, same ? "IDENTICAL" : "DIFFER");
        if (!same) {
            printf("FAIL: state bytes differ across save/load/save\n");
            return 1;
        }
    }

    // replay the held token (pos = stored - 1, consecutive with cache head)
    // to refresh logits, then continue - the standard session protocol
    if (!decode_at(ctx, &history.back(), 1, (int) history.size() - 1)) {
        printf("FAIL: replay last token\n");
        return 1;
    }
    pos = (int) history.size();
    if (!gen(8, back)) {
        printf("FAIL: post-restore generation\n");
        return 1;
    }

    // NOTE: llama_state_save_file with n_token_count>0 also stores the prompt
    // prefix; the load path re-decodes it, so the sampler must be reset to
    // match. Greedy has no state; still accept sampler explicitly.
    llama_sampler_reset(smpl);

    // reference continuation is held + tail; restored run must reproduce all 9
    std::vector<llama_token> ref_full = ref;
    ref_full.insert(ref_full.end(), tail.begin(), tail.end());
    std::vector<llama_token> back_full;
    // back[0] corresponds to ref_full[1]: the replayed held token's own
    // sample was consumed by the replay decode, so compare from there
    bool same = back.size() == ref_full.size() - 1;
    for (size_t i = 0; same && i < back.size(); i++) {
        same = back[i] == ref_full[i + 1];
    }
    printf("reference :");
    for (auto t : ref_full) {
        printf(" %d", t);
    }
    printf("\nrestored  : %d", ref_full[0]);
    for (auto t : back) {
        printf(" %d", t);
    }
    printf("\n%s\n", same ? "MATCH" : "MISMATCH");
    if (!same) {
        failures++;
    }

    remove(path);
    remove("/tmp/test-kvarn-state-2.bin");
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    if (failures) {
        printf("=== FAILED ===\n");
        return 1;
    }
    printf("=== Done ===\n");
    return 0;
}
