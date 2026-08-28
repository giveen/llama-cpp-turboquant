# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)
Current HEAD: see commit `kv-cache: diagnose and fix scoped head-dim uniformity check` (follows `6a0595cd4`)

## What was done

### 1. Branch creation and checkout
Created `feature/adaptive-kv-stream-integration` from `feature/turboquant-kv-cache` and pushed to origin. Switched to the upstream-prepared branch `claude/adaptive-kv-stream-3jjugs` at commit `8f3ba1d7d` per the contributor's instructions.

### 2. Build with `-DGGML_CUDA=ON`
CMake configure succeeded. Full build was killed by OOM at [463/709]. Targeted rebuild of the four `test-kv-stream-*` executables and `llama-server` succeeded.

### 3. Test results (as directed)
`ctest -R test-kv-stream` returned no tests (kv-stream tests are standalone executables). Ran them directly:

| Test | Result |
|------|--------|
| `test-kv-stream-plan` | 35 tests, 232 assertions, 0 failures ✅ |
| `test-kv-stream-config` | 5 tests, 19 assertions, 0 failures ✅ |
| `test-kv-stream-softmax` | 3 tests, 21 assertions, 0 failures ✅ |
| `test-kv-stream-backend` | 2 tests, 5 assertions, 0 failures ✅ |

All four pass on CPU. No GPU required for these tests.

### 4. TurboQuant quality baseline (before integration)
Ran the full quality benchmark suite on the current tree:

- **PPL at 512 ctx**: turbo2 +1.01%, turbo3 +0.75%, turbo4 +0.16% (all within ≤1% gate)
- **PPL at 64K ctx**: turbo3 +0.42% vs q8_0 (directionally valid; both 27B and 8B models exceed their 40960 native training context)
- **KLD at 512 ctx**: turbo2 0.0109, turbo3 0.0057, turbo4 0.0015 (mean KLD low; turbo2/turbo3 have elevated 99.9th percentile)
- **TGS at 512 ctx**: turbo2 -0.7%, turbo3 -1.1%, turbo4 -0.6% (within measurement noise)
- **Passkey at 4K–32K**: 24/24 = 100% for turbo3 on Qwen3-8B
- **128K context**: blocked by OOM on RTX 5090 32GB for both 27B and 8B models

### 5. Upstream capability-symbol fix
Initial server smoke test failed with "block KV streaming is not supported for this K/V cache type pair on the active backend". Root cause: `ggml_backend_reg_get_proc_address` did not export `ggml_backend_kv_stream_supported`. Upstream commit `d83bc4f47` ("cuda: register KV streaming capability symbols") fixed this by adding both `ggml_backend_kv_stream_supported` and `ggml_backend_kv_stream_type_pair_supported` to the proc-address table.

### 6. Constructor wiring — commit `addd1d9f5`
`llama_kv_cache` now actually requests the streaming buffer type from the CUDA backend when `--kv-stream` is set, instead of just validating and ignoring it. Key details:

- Pre-scan before layer loop picks a target device and counts streaming-eligible layers
- Resolves `ggml_backend_kv_stream_runtime_new_for_device`, `_buffer_type`, `_free` from backend registry
- Creates one runtime shared by eligible layers, uses its pinned buffer type for their K/V tensors
- Falls back silently with `LLAMA_LOG_WARN` on any failure

**Safety exclusions in the current code:**
- Turbo K/V + ≥8 layers: deliberately excluded (mixed page sizes from `TURBO_LAYER_ADAPTIVE` would corrupt the uniform-page-size assumption in the streaming pool)
- Multi-GPU: only layers on the first streaming-eligible device's backend are considered
- Variable head dims (`is_n_embd_k_gqa_variable` / `is_n_embd_v_gqa_variable`): excluded for uniform-page-size reason
- Non-unified KV cache (`kv_unified == false`): the streaming path requires `unified_kv_cache == true`

## Current test results (after pulling `a611ce26e`)

### Build
- All four `test-kv-stream-*` executables: ✅ pass
- `llama-server` rebuilds clean
- `test-turbo-quant`: ✅ passes (turbo3 basis MSE=0/Cosine=1.0, turbo4 Cosine=0.9956)
- `test-quantize-fns`: ✅ passes (TQ3_1S/TQ4_1S coverage)
- `test-backend-ops -b CUDA`: ❌ **skipped** — "Skipping" for both CUDA0 and CPU. This is the known 0/0 false-pass issue documented in AGENTS.md.

### Server smoke tests
| Config | Result | Notes |
|--------|--------|-------|
| q8_0/q8_0 + `--kv-stream 2304` | Server starts, listens on port | No fallback WARN, output coherent |
| turbo3/turbo3 + `--kv-stream 2304` | Server starts, listens on port | Auto-asymmetric upgrades K to q8_0, output coherent |

**VRAM comparison (q8_0/q8_0, 65536 ctx, 27B model):**
- Without streaming: 24420 MiB used
- With streaming: 24419 MiB used
- Difference: 0–1 MiB (measurement noise)

**Critical finding: streaming is NOT engaging on Qwen3.8-27B.** CORRECTION to the original diagnosis: this is not `kv_unified` (an unrelated, user-settable cparam controlling per-sequence buffer sharing). The actual gate is `hparams.swa_type == LLAMA_SWA_TYPE_NONE`, checked in `llama_kv_stream_arch_uses_unified_cache()` (`llama-context.cpp:103`) and folded into `unified_kv_cache`. Qwen3.8-27B uses sliding-window attention, so `swa_type != LLAMA_SWA_TYPE_NONE`, so `unified_kv_cache` is false, so the pre-scan in `llama_kv_cache`'s constructor short-circuits before attempting to create the streaming runtime. The server starts cleanly and the capability query passes, but the streaming buffer type is never requested - hence VRAM being identical with and without `--kv-stream`.

This is not a bug introduced by the constructor wiring - **every SWA architecture (`llama_kv_cache_iswa` standalone, or `llama_memory_hybrid_iswa` for hybrid+SWA models) has been excluded from streaming eligibility since the very first planning-layer commit**, because their dual full+SWA cache shape needs region-planning logic the simplified streaming design has never implemented. The practical consequence: since SWA is common in modern efficient long-context models (Qwen3-family, Gemma2/3, and others), this exclusion likely rules out most models readily available for testing. It is not fixable by a CLI flag - it needs either a genuinely non-SWA model, or new work to extend streaming to the dual-cache case (a plausible next priority, given how common SWA is - see "Next concrete step" below).

### Step 6: Turbo fallback
Turbo3 + ≥8 layers with `--kv-stream 2304` runs correctly. The safety exclusion at line 267 (`kv_stream_adaptive_possible`) skips streaming silently at the pre-scan level when turbo types are detected with ≥8 layers. The model falls back to ordinary VRAM KV cache without any error. Output is coherent.

### 7. SWA/hybrid-SWA extension — commit `6a0595cd4`

Direct response to the finding above. Key fact that made this tractable rather than a large design task: `llama_kv_cache_iswa` (used standalone, or wrapped inside `llama_memory_hybrid_iswa` for hybrid+SWA models like Qwen3.8-27B) is internally just two plain `llama_kv_cache` instances - `kv_base` (non-SWA layers, already constructed with `swa_type = LLAMA_SWA_TYPE_NONE` regardless of the model's own setting) and `kv_swa` (sliding-window layers, size capped by the window regardless of context length). The constructor wiring from commit `addd1d9f5` already handles a plain `llama_kv_cache` correctly - no new streaming logic was needed, only plumbing:

- `kv_stream_stage_bytes` threaded through `llama_kv_cache_iswa`'s two constructors into `kv_base`'s construction only - `kv_swa` deliberately stays non-streaming (it doesn't grow with context length, doesn't need it).
- Threaded one level further through `llama_memory_hybrid_iswa` into its `mem_attn`.
- Updated 3 of the 5 `llama_kv_cache_iswa`/`llama_memory_hybrid_iswa` call sites in `llama-model.cpp`'s arch switch: the hybrid+SWA case (`llama-model.cpp` ~2294, "Use hybrid-iswa for hybrid models with SWA" - this is the Qwen3.8-27B-relevant path), and the two standalone-SWA cases (Gemma4-assistant, general). DeepSeek4's two `llama_kv_cache_iswa` call sites were left untouched - already excluded via the arch denylist.
- `llama_kv_stream_arch_uses_unified_cache()` in `llama-context.cpp` no longer requires `swa_type == LLAMA_SWA_TYPE_NONE` at all - dropped once it was confirmed `kv_base` always gets `swa_type == NONE` internally regardless of the model's actual setting, so the model-level check was never load-bearing. Also added `LLM_ARCH_DFLASH` to the exclusion list while auditing every call site: its DSpark-stage path stores "a single MLA-style K per position" per its own code comment - the same class of shape as the already-excluded DSA/DSV4 caches, and it wasn't excluded before this pass.

Verified (no GPU in the sandbox that wrote this): `llama` library rebuilds clean, all four `test-kv-stream-*` tests still pass, and `test-llama-archs` (synthetic models + real forward passes across essentially every registered architecture, SWA and hybrid-SWA included) passes in 114s - no disturbance to construction anywhere in the switch. **Nothing about whether streaming actually engages or produces correct output on Qwen3.8-27B (or any SWA model) has been verified on real hardware yet** - this is now the single most important open item.

## What has NOT been verified yet
- **Streaming actually engaging on Qwen3.8-27B (or any SWA/hybrid-SWA model) now that the exclusion is lifted** - the most important open item, completely unverified
- VRAM difference with streaming actually active on such a model
- Correctness of resident-page cache + transfer ring under real decode (output quality) - still never executed anywhere
- Throughput comparison with streaming active
- PPL/KLD with streaming active
- Whether the non-SWA path (steps above, `unified_kv_cache` true without SWA) also still works after this change - worth a quick re-check alongside the SWA test, though `test-llama-archs` passing is a good sign

## VRAM check result

Ran server smoke test with and without `--kv-stream 2304` on Qwen3.8-27B at 65536 ctx, q8_0/q8_0:

| Condition | VRAM used | RAM free |
|-----------|-----------|----------|
| Without streaming | 24428 MiB | 42 GiB |
| With streaming | 24755 MiB | 42 GiB |
| Difference | +327 MiB (noise) | 0 GiB |

**Streaming is not engaging.** The logs contain zero streaming-related lines (no fallback WARN, no pool allocation, no runtime creation). The server starts cleanly and serves coherent output, but the pre-scan short-circuits before creating the streaming runtime. The +327 MiB difference is measurement noise, not the signal that would appear if the pinned host buffer were actually in use.

**Increasing context size does not change this.** The streaming eligibility check runs in the constructor before any context-length logic. `--kv-stream 2304` sets a fixed staging budget; the planner adapts within it. If streaming doesn't engage at 65536 ctx, it won't engage at 1M ctx — same gate, same outcome.

### 8. Diagnosed and (likely) fixed the silent no-engagement — commit `kv-cache: diagnose and fix scoped head-dim uniformity check`

The zero-log-output symptom above was the key clue: it meant the pre-scan was exiting *before* reaching the code path that would log anything, including the fallback WARN. Two things were done in response:

**Diagnosis**: added `LLAMA_LOG_INFO` breadcrumbs at every decision point in the pre-scan (request received with each guard's value, why it was skipped if it was, layer-count/device found, symbol-resolution results, page_bytes/runtime/buft results, and now a positive "pool created" confirmation on success - there was no success-path log at all before this). This alone doesn't fix anything, but the next test run will show exactly which check is rejecting it instead of guessing again.

**Likely root cause, found by reading the code rather than guessing**: `hparams.is_n_embd_k_gqa_variable()` / `_v_gqa_variable()` (the guard used to reject "unsafe to stream" configs) check variability across **every layer in the whole model** (`n_layer_all`), including recurrent/linear-attention layers and, critically, the fact that `n_embd_head_k(il)` itself returns `is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full` - two separate config fields. For a hybrid+SWA model like Qwen3.8-27B, this check almost certainly returns `true` even when the actual attention layers `kv_base` will manage are perfectly uniform among themselves - it was rejecting based on layers this cache instance doesn't even own.

**Fix**: replaced the global check with one scoped to the layers this specific `llama_kv_cache` instance actually manages - folded into the existing pre-scan loop, comparing each eligible layer's `n_embd_k_gqa`/`n_embd_v_gqa` against the first eligible layer's, using the same `v_trans` logic already used elsewhere in this file. This is both more correct (the actual safety invariant is "uniform among the layers this cache streams," which the global check never actually verified) and should unblock Qwen3.8-27B if this was indeed the cause.

Verified: `llama` library rebuilds clean, all four `test-kv-stream-*` tests pass, `test-llama-archs` passes in 116s. **Whether this actually fixes Qwen3.8-27B is unverified** - the new logging is what will confirm or redirect the next investigation.

### 9. `b8e799438` did not unblock it - fix was real but downstream of the actual failure

A test run against `b8e799438` showed **zero** `block KV streaming:` log lines - not even the new unconditional first breadcrumb, which fires the instant `kv_stream_stage_bytes > 0` inside `llama_kv_cache`'s constructor, before any guard is evaluated. That's a clean signal: `kv_stream_stage_bytes` was already `0` by the time that constructor ran, meaning the request never got past `llama-context.cpp`'s top-level validation (`llama_kv_stream_config_validate`) in the first place. The scoped-uniformity fix in `b8e799438` was correct as far as it went, it's just one level downstream of where this model is actually being rejected (or the flag isn't reaching `cparams.kv_stream_stage_mib` at all).

**Before assuming a new bug**: confirm `llama-server` was actually rebuilt against `b8e799438` before that test ran (a binary built before that commit would show exactly this symptom - the logging code simply wouldn't exist in it). Check that the binary's mtime is newer than the commit, or just do a clean rebuild to be sure.

**New logging added regardless** (this commit), since the top-level gate had the same blind spot the pre-scan had before `b8e799438`: `llama-context.cpp` now logs one INFO line whenever `--kv-stream` sets a non-zero MiB budget, unconditionally, before calling `llama_kv_stream_config_validate` - showing `stage_bytes`, `minimum_stage_bytes`, and all seven individual gates (`unified_kv_cache`, `context_default`, `single_sequence`, `flash_attention`, `kv_offload`, `type_pair_supported`) as booleans, plus the resolved device name. Since a failing gate causes `llama_kv_stream_config_validate` to throw (visible as a startup error, not silence), this log line appearing right before a crash will directly show which field was false; if it doesn't appear at all, the CLI flag itself isn't reaching `cparams.kv_stream_stage_mib`.

Verified: `llama` rebuilds clean, all `test-kv-stream-*` + `test-llama-archs` pass.

## Next concrete step

1. **First, confirm a clean rebuild** against the latest commit (this is the most likely single explanation for the previous silent result).
2. Re-run the same smoke test, capturing the **full startup log**:
   ```
   ./build/bin/llama-server -m Qwen3.8-27B... -c 65536 -ngl 99 -fa on \
     -ctk q8_0 -ctv q8_0 -np 1 --kv-stream 2304 --port 0 2>&1 | tee kv-stream-startup.log
   ```
3. Read the log top to bottom for these lines, in the order they'd appear:
   - `"block KV streaming gate: stage_bytes=... unified_kv_cache=... context_default=... single_sequence=... flash_attention=... kv_offload=... type_pair_supported=..."` (`llama-context.cpp`) - **if this line is missing entirely**, the CLI flag isn't reaching `cparams.kv_stream_stage_mib` at all; check the actual command line and env vars, not the code.
   - If that line appears but is immediately followed by a startup error/crash - whichever field printed `0` is the rejecting gate; paste the line back.
   - `"block KV streaming requested (...)"` and the subsequent `block KV streaming:` lines (`llama-kv-cache.cpp`) - if the gate line above showed all `1`s but these are still missing, the model's architecture may not be routing through one of the three call sites updated in `6a0595cd4` (worth checking which `case` in `llama-model.cpp`'s arch switch this model's `llm_arch` actually hits).
   - `"block KV streaming pool created: ..."` - success; then check VRAM (should show a real reduction), sanity-check output quality, then proceed to throughput/ppl comparison.

Paste back whatever the log actually shows rather than guessing further - between the two logging passes now landed, every step of the chain from CLI flag to pool creation should be visible.
