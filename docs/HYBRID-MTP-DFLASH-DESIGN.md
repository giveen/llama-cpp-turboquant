# MTP + DFlash Hybrid Speculative Decoding — Design Sketch

**Branch:** `feature/speculative-mtp-dflash-hybrid`  
**Repo:** `/mnt/storage/llama-cpp-turboquant-hybrid` (clone of `giveen/llama-cpp-turboquant`, tracking `origin/feature/turboquant-kv-cache`)  
**Target model for prototyping:** Qwen3.5-2B (small, dense, has real MTP head in GGUF, fits on consumer hardware)  
**Draft model for DFlash side:** Qwen3-4B-DFlash or any Qwen3-size DFlash drafter (separate GGUF)

---

## 1. What "hybrid" means here

A single speculative decode step where **DFlash drafts a branching tree**, the **highest-confidence branch is promoted to a flat prefix**, and **MTP extends that prefix** with an additional flat sequence. The main model verifier sees one flat sequence with per-token provenance.

Ordering is **DFlash first, MTP second**:
- DFlash gives structure (a tree of alternatives).
- MTP needs a flat conditioning prefix; it has no mechanism to choose among branches.
- Reversing the order would require MTP to emit branching structure, which it does not.

---

## 2. Data flow (one speculative step)

```
Accepted prefix (from main model)
         │
         ▼
   ┌───────────┐
   │  DFlash   │  tree depth = D, block size = B
   │  draft    │  emits N candidate paths + per-node confidence
   └─────┬─────┘
         │  select highest-confidence surviving branch → flat prefix P tokens
         ▼
   ┌───────────┐
   │    MTP    │  extension length = T
   │  draft    │  conditions on DFlash-accepted prefix
   └─────┬─────┘
         │  appends T tokens
         ▼
   ┌─────────────────────────┐
   │  Flat draft sequence    │  = [P DFlash tokens] + [T MTP tokens]
   │  Provenance mask        │  per-token: 0 = DFlash, 1 = MTP
   └──────────┬──────────────┘
              │
              ▼
         Main model verifier (one forward pass)
              │
              ▼
         Accept longest matching prefix
              │
      ┌───────┴────────┐
      │  Mixed outcome │  update KV + stats per backend
      │  DFlash tree   │  kept alive for next step
      │  MTP KV reset  │  rebuilt on next hybrid pass
      └────────────────┘
```

---

## 3. Concrete struct changes

### 3.1 New type enum

**File:** `common/speculative.cpp`, line ~39 (inside `common_speculative_type_from_name_map`)

```c
{"draft-hybrid-mtp-dflash", COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID_MTP_DFLASH},
```

**File:** `include/llama.h` (or wherever `enum common_speculative_type` lives)

```c
COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID_MTP_DFLASH,
```

### 3.2 New struct: `draft_hybrid_state`

**File:** `common/speculative.h`, after the existing `common_speculative_draft_params` struct (~line 51)

```c
struct draft_hybrid_state {
    // DFlash sub-state
    dflash_state * df;
    int            df_max_depth;
    float          df_confidence_threshold;

    // MTP sub-state
    llama_kv_cache * mtp_kv;
    int              mtp_extension_length;

    // Composed draft output
    llama_token  * draft_tokens;      // capacity = P + T
    uint8_t      * provenance;        // capacity = P + T, 0=DFlash, 1=MTP
    int            n_draft;           // actual length this step

    // Stats
    size_t n_gen_df_tokens;
    size_t n_acc_df_tokens;
    size_t n_gen_mtp_tokens;
    size_t n_acc_mtp_tokens;
    int64_t t_draft_df_us;
    int64_t t_draft_mtp_us;
};
```

### 3.3 New impl class: `common_speculative_impl_hybrid_mtp_dflash`

**File:** `common/speculative.cpp`, after `common_speculative_impl_dflash` (~line 1200+)

```c
struct common_speculative_impl_hybrid_mtp_dflash : public common_speculative_impl {
    common_params_speculative params;

    // Two models
    llama_model   * model_df;
    llama_context * ctx_df;
    llama_model   * model_mtp;
    llama_context * ctx_mtp;

    // Per-sequence state
    std::vector<draft_hybrid_state> states;

    common_speculative_impl_hybrid_mtp_dflash(
        const common_params_speculative & p, uint32_t n_seq)
        : common_speculative_impl(
            COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID_MTP_DFLASH, n_seq)
        , params(p)
        , model_df(nullptr), ctx_df(nullptr)
        , model_mtp(nullptr), ctx_mtp(nullptr)
    {
        states.resize(n_seq);
    }

    ~common_speculative_impl_hybrid_mtp_dflash() override {
        if (ctx_df) llama_free(ctx_df);
        if (model_df) llama_free_model(model_df);
        if (ctx_mtp) llama_free(ctx_mtp);
        if (model_mtp) llama_free_model(model_mtp);
    }

    bool init(const std::string & df_path, const std::string & mtp_path) {
        // Load DFlash model
        llama_model_params mparams = llama_model_default_params();
        model_df = llama_load_model_from_file(df_path.c_str(), mparams);
        if (!model_df) return false;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 8192;  // draft context, tune later
        ctx_df = llama_new_context_with_model(model_df, cparams);
        if (!ctx_df) return false;

        // Load MTP model (must share target vocab)
        model_mtp = llama_load_model_from_file(mtp_path.c_str(), mparams);
        if (!model_mtp) return false;

        ctx_mtp = llama_new_context_with_model(model_mtp, cparams);
        if (!ctx_mtp) return false;

        return true;
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        draft_hybrid_state & s = states[seq_id];

        // Reset DFlash state — reuse existing tree if warm, else rebuild
        dflash_reset(s.df, prompt);

        // Reset MTP KV — rebuilt every step from DFlash branch prefix
        llama_kv_cache_clear(s.mtp_kv);

        s.n_draft = 0;
    }

    void draft(std::vector<common_speculative_draft_params_vec> & dparams) override {
        for (size_t seq_idx = 0; seq_idx < dparams.size(); ++seq_idx) {
            auto & dp = dparams[seq_idx];
            if (!dp.drafting) continue;

            draft_hybrid_state & s = states[seq_idx];
            auto t0 = ggml_time_us();

            // ---- Phase 1: DFlash tree draft ----
            dflash_tree * tree = dflash_propose_tree(s.df,
                dp.prompt->data(), dp.prompt->size() - dp.n_past);

            // Select highest-confidence branch
            int p = 0;
            llama_token branch[MAX_DFLASH_DEPTH];
            float branch_conf[MAX_DFLASH_DEPTH];
            dflash_select_best_branch(tree,
                s.df_confidence_threshold, s.df_max_depth,
                branch, branch_conf, &p);

            t0 = ggml_time_us() - t0;
            s.t_draft_df_us += t0;

            // ---- Phase 2: MTP extension from branch prefix ----
            auto t1 = ggml_time_us();
            llama_kv_cache_clear(s.mtp_kv);

            // Prime MTP KV with the branch tokens
            if (p > 0) {
                llama_batch batch = llama_batch_init(p, 0, 1);
                for (int i = 0; i < p; i++) {
                    batch.token[i] = branch[i];
                    batch.pos[i]   = dp.n_past + i;
                    batch.n_tokens++;
                }
                llama_decode(s.mtp, batch);
            }

            // MTP head proposes T more tokens
            int t = 0;
            llama_token mtp_tokens[MAX_MTP_DRAFT];
            mtp_propose(s.mtp_kv, dp.n_past + p, mtp_tokens, s.mtp_extension_length, &t);

            t1 = ggml_time_us() - t1;
            s.t_draft_mtp_us += t1;

            // ---- Phase 3: Concatenate + provenance ----
            for (int i = 0; i < p; i++) {
                s.draft_tokens[i] = branch[i];
                s.provenance[i]  = 0;  // DFlash
            }
            for (int i = 0; i < t; i++) {
                s.draft_tokens[p + i] = mtp_tokens[i];
                s.provenance[p + i]  = 1;  // MTP
            }
            s.n_draft = p + t;

            // Update stats
            s.n_gen_df_tokens  += p;
            s.n_gen_mtp_tokens += t;

            // Write result back
            dp.result->resize(s.n_draft);
            for (int i = 0; i < s.n_draft; i++)
                (*dp.result)[i] = s.draft_tokens[i];

            // Reset drafting flag so chain implementations don't re-fire
            dp.drafting = false;
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        draft_hybrid_state & s = states[seq_id];
        if (n_accepted == 0) return;

        // Walk provenance to find per-backend split
        int df_accepted = 0;
        int mtp_accepted = 0;
        for (int i = 0; i < n_accepted && i < s.n_draft; i++) {
            if (s.provenance[i] == 0)
                df_accepted++;
            else if (s.provenance[i] == 1)
                mtp_accepted++;
        }

        s.n_acc_df_tokens  += df_accepted;
        s.n_acc_mtp_tokens += mtp_accepted;

        // DFlash: prune tree to df_accepted, keep alive for next step
        dflash_prune_tree(s.df, df_accepted);

        // MTP: KV is invalid past the accepted point — tear down
        llama_kv_cache_clear(s.mtp_kv);
        s.n_draft = 0;
    }

    bool need_embd() const override {
        // DFlash needs target embeddings; MTP does not
        return true;
    }

    bool need_embd_nextn() const override {
        // MTP needs pre-norm embeddings from the target
        return true;
    }

    bool get_state(llama_seq_id seq_id, std::vector<uint8_t> & data) const override {
        // Serialize both DFlash tree + MTP KV
        // ...
        return true;
    }

    void set_state(llama_seq_id seq_id, const std::vector<uint8_t> & data) override {
        // Deserialize both
        // ...
    }
};
```

### 3.4 Integration into `common_speculative_init`

**File:** `common/speculative.cpp`, inside `common_speculative_init`

Add a new branch in the switch/if-chain that dispatches on `params.type`:

```c
case COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID_MTP_DFLASH: {
    auto * impl = new common_speculative_impl_hybrid_mtp_dflash(params, n_seq);
    const char * df_path  = params.draft_dflash_model.c_str();
    const char * mtp_path = params.draft_mtp_model.c_str();
    if (!impl->init(df_path, mtp_path)) {
        delete impl;
        return nullptr;
    }
    return impl;
}
```

---

## 4. CLI parameter surface

**File:** `common/params.h` (or equivalent `common_params_speculative` definition)

New fields in `common_params_speculative`:

```c
std::string draft_dflash_model;   // path to DFlash drafter GGUF
std::string draft_mtp_model;      // path to MTP model GGUF (or empty = use target's own MTP head)
int         hybrid_df_max_depth   = 4;
float       hybrid_df_confidence  = 0.5f;
int         hybrid_mtp_extension  = 3;
```

**New CLI args** (in `common/arg.cpp`):

```
  --spec-dflash-model PATH       DFlash draft model for hybrid mode (default: "")
  --spec-mtp-model PATH          MTP draft model for hybrid mode (default: "")
  --hybrid-df-depth N            Max DFlash tree depth (default: 4)
  --hybrid-df-confidence F       Min node confidence to keep branch (default: 0.5)
  --hybrid-mtp-length N          MTP extension tokens (default: 3)
```

**Usage example:**

```bash
llama-server \
  -m Qwen3.5-9B.gguf \
  -md Qwen3-4B-DFlash.gguf \
  --spec-type draft-hybrid-mtp-dflash \
  --spec-dflash-model Qwen3-4B-DFlash.gguf \
  --spec-mtp-model Qwen3.5-2B-MTP.gguf \
  --hybrid-df-depth 4 \
  --hybrid-df-confidence 0.5 \
  --hybrid-mtp-length 3 \
  --spec-draft-n-max 8 \
  -fa on
```

When `--spec-mtp-model` is empty, the hybrid uses the target model's **own MTP head** (`draft-mtp` path) instead of a separate model. This is the cheaper option and matches how `draft-mtp` already works today.

---

## 5. MTP propose helper (new function)

**File:** `common/speculative.cpp`, new static function

```c
static void mtp_propose(llama_kv_cache * kv, llama_pos pos,
                        llama_token * out, int n_max, int * n_out) {
    // Extract MTP head logits from the target context
    // This reuses the existing llama_get_embeddings_nextn_ith() /
    // llama_set_embeddings_nextn() API already used by draft-mtp

    int n = 0;
    for (int i = 0; i < n_max; i++) {
        float * logits = llama_get_embeddings_nextn_ith(target_ctx, i);
        if (!logits) break;

        // Greedy or sampled — follow existing MTP policy
        llama_token id = sample_greedy(logits);
        out[n++] = id;

        // Advance MTP KV with this token for next iteration
        llama_batch step = llama_batch_init(1, 0, 1);
        step.token[0] = id;
        step.pos[0]   = pos + i;
        step.n_tokens = 1;
        llama_decode(mtp_ctx, step);
    }
    *n_out = n;
}
```

This is a simplification — the real MTP path already exists in `common/speculative.cpp` under `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`. The hybrid just calls into it, rather than reimplementing.

---

## 6. DFlash tree selection helper (new function)

**File:** `src/models/dflash.cpp` or `common/speculative.cpp`

```c
// Walk the DFlash tree, pick the highest-confidence branch that stays
// within max_depth. Returns flat token array + per-token confidence.
void dflash_select_best_branch(dflash_tree * tree,
                                float threshold,
                                int max_depth,
                                llama_token * branch_out,
                                float * conf_out,
                                int * n_out) {
    *n_out = 0;

    // Start from root's top-k children sorted by confidence
    dflash_node * candidates[MAX_DFLASH_BEAM];
    int n_candidates = dflash_top_k(tree->root, threshold, MAX_DFLASH_BEAM, candidates);

    for (int c = 0; c < n_candidates && *n_out < max_depth; c++) {
        dflash_node * node = candidates[c];
        branch_out[*n_out] = node->token;
        conf_out[*n_out]   = node->confidence;
        (*n_out)++;

        // Walk down preferring highest-confidence child at each step
        while (*n_out < max_depth && node->n_children > 0) {
            node = dflash_best_child(node, threshold);
            if (!node) break;
            branch_out[*n_out] = node->token;
            conf_out[*n_out]   = node->confidence;
            (*n_out)++;
        }
    }
}
```

---

## 7. Acceptance and provenance accounting

The current `common_speculative::accept` path already walks the verified sequence and finds the mismatch point. The hybrid impl overrides `accept()` to additionally:

1. Walk the `provenance` mask to count accepted tokens per backend.
2. Update per-backend stats (`n_acc_df_tokens`, `n_acc_mtp_tokens`).
3. Call `dflash_prune_tree()` to keep the DFlash tree alive for the surviving prefix.
4. Clear MTP KV unconditionally — the MTP extension is ephemeral and must be rebuilt each step.

---

## 8. Verification gates before claiming a win

Before merging or calling this "done":

1. **Exactness gate:** Run the target model with `--spec-type none` and with `--spec-type draft-hybrid-mtp-dflash` on the same 200 prompts (e.g., AlpacaEval + HumanEval + GSM8K). Compare token-for-token with greedy sampling. Zero divergence required at temp=0. At temp>0, divergence is expected (speculative decode reorders float reductions); document the delta rather than treating it as a bug.

2. **Acceptance rate gate:** Instrument per-backend acceptance rate:
   - `df_accept_rate = n_acc_df_tokens / n_gen_df_tokens`
   - `mtp_accept_rate = n_acc_mtp_tokens / n_gen_mtp_tokens`
   - `combined = (n_acc_df + n_acc_mtp) / (n_gen_df + n_gen_mtp)`
   The hybrid must beat the better of DFlash-alone or MTP-alone by **at least 10% relative** acceptance rate to justify its complexity.

3. **Throughput gate:** Measure tokens/sec at `bs=1`, 64-prompt / 128-output, same hardware, same quant. The hybrid must exceed the better single-backend throughput. If it does not, the hybrid is slower despite higher acceptance because of the two sequential draft passes.

4. **Memory gate:** Combined VRAM + RAM for target + DFlash + MTP must fit within the tested hardware budget. Two draft KV caches plus main KV is a strict increase over either single-backend path.

5. **Regression gate:** DFlash-alone and MTP-alone must still pass their existing tests. The hybrid code must not touch those paths.

---

## 9. Test plan

**Unit level:**
- `tests/unit/test_hybrid_dflash_mtp.cpp` — standalone harness that:
  - Constructs a hybrid impl with two tiny test models.
  - Calls `begin()` / `draft()` / `accept()` 100 times.
  - Asserts `n_draft > 0`, provenance mask length matches `n_draft`, stats accumulate correctly.
  - Simulates partial rejection: feed mismatched verification sequence, assert DFlash tree survives for surviving prefix and MTP KV is cleared.

**Integration level:**
- Build the full `llama-server` binary from this branch.
- Run with `--spec-type draft-hybrid-mtp-dflash` against Qwen3.5-2B target + Qwen3-4B-DFlash draft + Qwen3.5-1B MTP draft.
- Measure acceptance rate and t/s.
- Compare against:
  - `--spec-type draft-dflash` alone
  - `--spec-type draft-mtp` alone

**Minimum viable signal:** If the hybrid does not beat the better single backend on acceptance rate AND t/s on a 2B target, it is unlikely to win on a 9B or 27B target where the verification cost ratio is worse.

---

## 10. Engineering cost estimate

| Task | Estimated effort |
|---|---|
| Struct + vtable additions | 1 day |
| `hybrid_mtp_dflash` impl class | 1–2 days |
| CLI plumbing + parameter validation | 0.5 day |
| MTP propose helper (thin wrapper over existing code) | 0.5 day |
| DFlash tree selection helper | 0.5 day |
| Acceptance / provenance accounting | 0.5 day |
| Unit tests | 1 day |
| Build fixups + CI | 0.5 day |
| Benchmark + acceptance gate evaluation | 1–2 days |
| **Total** | **5–7 days** |

The hard parts are the acceptance-loop invariants and provenance-correct stats, not the forward pass. The risk is subtle bugs under partial rejection, which this architecture maximizes.

---

## 11. Opinion

**Does it work structurally?** Yes. DFlash-first ordering respects both backends' conditioning semantics. The tree-to-flat handoff is a clean seam.

**Does it win in practice?** Conditionally. It helps when:
- DFlash alone tops out at depth D because its tree memory scales with branching factor.
- MTP alone tops out at length T because chain-conditioning degrades.
- The combination beats either at their respective limits on your hardware.

It does **not** help when:
- Single-backend acceptance rate is already above ~70%.
- You are memory-bound — two draft KV caches plus main KV is strictly more pressure.
- The sequential draft passes (DFlash then MTP) take longer than one or the other alone, even with higher combined acceptance.

**My recommendation:** Prototype behind the new `COMMON_SPECULATIVE_TYPE_DRAFT_HYBRID_MTP_DFLASH` enum, instrument per-backend stats from day one, and benchmark against DFlash-alone and MTP-alone on Qwen3.5-2B. If the hybrid does not beat the better single backend by 10%+ relative acceptance rate AND throughput, abandon it. The engineering cost is real; the expected marginal gain is small.

---

*Sketch written against `giveen/llama-cpp-turboquant` branch `feature/turboquant-kv-cache`, commit `b7fef7186`, 2026-08-25.*
