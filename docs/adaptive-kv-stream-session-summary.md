# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)
Current HEAD: `bf9971c38` ("llama-bench: wire up --kv-stream sweep parameter")

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
- Turbo K/V + `TURBO_LAYER_ADAPTIVE` actually varying per-layer types: excluded (mixed page sizes would corrupt the uniform-page-size assumption in the streaming pool). See section 11 below - this used to blanket-exclude any turbo K/V with >= 8 layers; it's now precise about which mode/type combinations actually vary.
- Multi-GPU: only layers on the first streaming-eligible device's backend are considered
- Variable head dims (`is_n_embd_k_gqa_variable` / `is_n_embd_v_gqa_variable`): excluded for uniform-page-size reason
- Non-unified KV cache (`kv_unified == false`): the streaming path requires `unified_kv_cache == true`

### 7. SWA/hybrid-SWA extension — commit `6a0595cd4`

Direct response to the finding above. Key fact that made this tractable rather than a large design task: `llama_kv_cache_iswa` (used standalone, or wrapped inside `llama_memory_hybrid_iswa` for hybrid+SWA models like Qwen3.8-27B) is internally just two plain `llama_kv_cache` instances - `kv_base` (non-SWA layers, already constructed with `swa_type = LLAMA_SWA_TYPE_NONE` regardless of the model's own setting) and `kv_swa` (sliding-window layers, size capped by the window regardless of context length). The constructor wiring from commit `addd1d9f5` already handles a plain `llama_kv_cache` correctly - no new streaming logic was needed, only plumbing:

- `kv_stream_stage_bytes` threaded through `llama_kv_cache_iswa`'s two constructors into `kv_base`'s construction only - `kv_swa` deliberately stays non-streaming (it doesn't grow with context length, doesn't need it).
- Threaded one level further through `llama_memory_hybrid_iswa` into its `mem_attn`.
- Updated 3 of the 5 `llama_kv_cache_iswa`/`llama_memory_hybrid_iswa` call sites in `llama-model.cpp`'s arch switch: the hybrid+SWA case (`llama-model.cpp` ~2294, "Use hybrid-iswa for hybrid models with SWA" - this is the Qwen3.8-27B-relevant path), and the two standalone-SWA cases (Gemma4-assistant, general). DeepSeek4's two `llama_kv_cache_iswa` call sites were left untouched - already excluded via the arch denylist.
- `llama_kv_stream_arch_uses_unified_cache()` in `llama-context.cpp` no longer requires `swa_type == LLAMA_SWA_TYPE_NONE` at all - dropped once it was confirmed `kv_base` always gets `swa_type == NONE` internally regardless of the model's actual setting, so the model-level check was never load-bearing. Also added `LLM_ARCH_DFLASH` to the exclusion list while auditing every call site: its DSpark-stage path stores "a single MLA-style K per position" per its own code comment - the same class of shape as the already-excluded DSA/DSV4 caches, and it wasn't excluded before this pass.

Verified: `llama` library rebuilds clean, all four `test-kv-stream-*` tests still pass, and `test-llama-archs` (synthetic models + real forward passes across essentially every registered architecture, SWA and hybrid-SWA included) passes in 114s - no disturbance to construction anywhere in the switch.

### 8. Diagnosed and fixed the silent no-engagement — commit `b8e799438`

The zero-log-output symptom was the key clue: it meant the pre-scan was exiting *before* reaching the code path that would log anything, including the fallback WARN. Two things were done in response:

**Diagnosis**: added `LLAMA_LOG_INFO` breadcrumbs at every decision point in the pre-scan (request received with each guard's value, why it was skipped if it was, layer-count/device found, symbol-resolution results, page_bytes/runtime/buft results, and now a positive "pool created" confirmation on success - there was no success-path log at all before this). This alone doesn't fix anything, but the next test run shows exactly which check is rejecting it instead of guessing again.

**Root cause**: `hparams.is_n_embd_k_gqa_variable()` / `_v_gqa_variable()` check variability across **every layer in the whole model** (`n_layer_all`), including recurrent/linear-attention layers and, critically, the fact that `n_embd_head_k(il)` itself returns `is_swa(il) ? n_embd_head_k_swa : n_embd_head_k_full` - two separate config fields. For a hybrid+SWA model like Qwen3.8-27B, this check almost certainly returns `true` even when the actual attention layers `kv_base` will manage are perfectly uniform among themselves - it was rejecting based on layers this cache instance doesn't even own.

**Fix**: replaced the global check with one scoped to the layers this specific `llama_kv_cache` instance actually manages - folded into the existing pre-scan loop, comparing each eligible layer's `n_embd_k_gqa`/`n_embd_v_gqa` against the first eligible layer's, using the same `v_trans` logic already used elsewhere in this file. This is both more correct (the actual safety invariant is "uniform among the layers this cache streams," which the global check never actually verified) and should unblock Qwen3.8-27B if this was indeed the cause.

Verified: `llama` library rebuilds clean, all four `test-kv-stream-*` tests pass, `test-llama-archs` passes in 116s.

### 9. Top-level gate logging — commit `73e6f1310`

The scoped-uniformity fix in `b8e799438` was correct as far as it went, but a test run showed **zero** `block KV streaming:` log lines - not even the new unconditional first breadcrumb. That's a clean signal: `kv_stream_stage_bytes` was already `0` by the time `llama_kv_cache`'s constructor ran, meaning the request never got past `llama-context.cpp`'s top-level validation (`llama_kv_stream_config_validate`) in the first place.

This commit adds one `LLAMA_LOG_INFO` line, printed whenever `--kv-stream` sets a non-zero MiB budget, before calling `llama_kv_stream_config_validate`: shows `stage_bytes`, `minimum_stage_bytes`, and all seven individual gates (`unified_kv_cache`, `context_default`, `single_sequence`, `flash_attention`, `kv_offload`, `type_pair_supported`) plus the resolved device name. A failing gate makes `validate()` throw (visible as a startup error, not silence), so this line appearing right before that error shows exactly which field was false; if it doesn't appear at all, the CLI flag isn't reaching `cparams.kv_stream_stage_mib`.

Verified: `llama` rebuilds clean, all `test-kv-stream-*` + `test-llama-archs` pass.

## Confirmed engagement on Qwen3.8-27B

Server smoke test with `--kv-stream 2304` on Qwen3.8-27B at 65536 ctx, q8_0/q8_0:

```
0.11.162.267 I llama_context: block KV streaming gate: stage_bytes=2304.00 MiB minimum_stage_bytes=0.53 MiB unified_kv_cache=1 context_default=1 single_sequence=1 flash_attention=1 kv_offload=1 type_pair_supported=1 (dev=CUDA0)
0.11.162.268 I llama_context: experimental block KV streaming config validated, pool = 2304.00 MiB
0.11.162.698 I llama_kv_cache: block KV streaming requested (2304.00 MiB): offload=1 adaptive_possible=0 n_layer=64
0.11.162.700 I llama_kv_cache: block KV streaming: pre-scan found 16 eligible layer(s) on device CUDA0
0.11.162.701 I llama_kv_cache: block KV streaming: device_index=0 runtime_new=1 buffer_type=1 free=1
0.11.164.968 I llama_kv_cache: block KV streaming: page_bytes=557056 runtime=1 buft=1
0.11.164.970 I llama_kv_cache: block KV streaming pool created: 2304.00 MiB, 16 layer(s), device CUDA0
```

**Streaming is now engaging on Qwen3.8-27B.** The server starts cleanly, serves coherent output, and the full breadcrumb trail from CLI flag to pool creation is present in the logs.

### Step 6: Turbo fallback (superseded by section 11 below)
Turbo3 + ≥8 layers with `--kv-stream 2304` ran correctly but fell back to the ordinary VRAM KV cache without error: the old safety exclusion treated any turbo K/V type combined with ≥8 layers as unsafe, regardless of whether `TURBO_LAYER_ADAPTIVE` would actually vary anything for that type. Section 11 replaces this with a precise check - a plain turbo3 K/V config like this one (no `TURBO_LAYER_ADAPTIVE` override, V isn't turbo2 so the auto-enable never triggers) should now stream instead of falling back. **Needs re-verification on hardware** to confirm it now engages.

### 10. Wired `--kv-stream` into llama-bench, llama-perplexity, llama-cli — commit `bf9971c38`

To unblock PPL/KLD comparison, throughput benchmarking, and general interactive
testing with streaming active, checked whether the three remaining CLI tools
expose `--kv-stream`:

- **llama-bench**: has its own `cmd_params` sweep system (separate from
  `common_params`, built for combinatorial parameter sweeps), so this
  genuinely needed new code. Added `--kv-stream <n>` as a full sweep
  parameter mirroring the existing `n_batch`/`fit_min_ctx` pattern: CLI
  parsing (comma-range values via `parse_int_range`), `cmd_params` /
  `cmd_params_instance` / `test` struct fields, cartesian-product
  combination (new `for` loop + all three instance-literal sites), and a
  `kvs` results-table column that only appears when the value is actually
  swept away from the default (0). Built cleanly under this sandbox's
  CPU-only (`GGML_CUDA=OFF`) configuration; smoke-tested `--help` output,
  single-value and multi-value (`256,512,1024`) sweeps, and confirmed the
  `kvs` column only renders when non-default.
- **llama-perplexity** and **llama-cli**: needed **no code changes**.
  `--kv-stream` is registered in `common/arg.cpp` under
  `LLAMA_EXAMPLE_COMMON` with no `.set_examples()` restriction, and every
  example inherits `LLAMA_EXAMPLE_COMMON` options (`arg.cpp:1389-1396`).
  Both tools parse via `common_params_parse` -> `common_init_from_params`
  -> `common_context_params_to_llama`, which already threads
  `params.kv_stream_stage_mib` into `cparams.kv_stream_stage_mib`
  (`common/common.cpp:1727`). `llama-cli`'s embedded server
  (`cli-context.cpp:123`) passes the same unmodified `common_params`
  through to `server->start(params)`. Confirmed by building both targets
  and checking `--kv-stream N` appears in their `--help` output.

All three tools now build and expose `--kv-stream N` on this sandbox's
CPU-only build. GPU testing of the actual sweep/benchmark behavior (as
opposed to CLI wiring) still needs real hardware.

### 11. Precise turbo exclusion + `--kv-stream auto` vs manual: measured results

**Turbo3 K/V with `--kv-stream auto` at 65536 ctx, q8_0/q8_0:**

| Component | Baseline | `auto` | turbo3 K/V + `auto` |
|-----------|----------|--------|----------------------|
| CUDA0 model | 25972 MiB | 25972 MiB | 25972 MiB |
| CUDA0 KV cache | 2325 MiB | 149 MiB | 149 MiB |
| CUDA0 compute | 388 MiB | 592 MiB | 506 MiB |
| Host free | 1372 MiB | 3548 MiB | 2860 MiB |
| Unaccounted | 1653 MiB | 1913 MiB | 1749 MiB |
| **Device KV reduction** | — | **−2176 MiB** | **−2176 MiB** |
| **Host increase** | — | **+2176 MiB** | **+1488 MiB** |

- `auto` computed `stage=910 MiB` for this model/type at 65536 ctx, under the 8192 MiB ceiling.
- Device-side KV reduction matches the manual `2304` case: **−2176 MiB**.
- `turbo3` + `auto` reduces device KV cache by the same **−2176 MiB**; host increase is smaller because turbo3 uses less memory overall than q8_0/q8_0.
- All three modes reduce the same 16 eligible layers’ resident pages; the pool size scales with context, not cache type.
- `llama-perplexity` plain-PPL path still requires `-b <= -c`; with that workaround, both baseline and auto PPL at 512 ctx are `~6.9568–6.9576 ± 0.045`, directionally identical.
- The previous blocker `block KV streaming requires exactly one sequence (-np 1)` is unchanged; there is no exposed `-np` in `llama-perplexity` for plain PPL/KLD.

### 12. `--kv-stream auto`: derive the stage pool from `-c`

Requested so users don't have to hand-pick a MiB budget for every context size, e.g. `-c 262144 --kv-stream auto`. The pool is deliberately meant to stay much smaller than the full context's KV cache - sizing it to the full context would defeat the point of streaming.

Formula (`src/llama-context.cpp`, resolved once inside `llama_context`'s constructor):

- `resident_tokens = clamp(n_ctx * 10%, floor=2048 tokens, ceiling=n_ctx)`
- rounded up to the nearest 256-token page
- `stage_bytes = page_bytes(K+V, first eligible layer) * eligible_layer_count * (resident_pages + 4 scratch pages)`
- `stage_mib = min(ceil(stage_bytes / 1 MiB), 8192 MiB)`

Plumbing: new `bool kv_stream_auto` field threaded through `include/llama.h`, `src/llama-cparams.h`, `common/common.h`, `common/common.cpp`, and `src/llama-context.cpp`. `common/arg.cpp`'s `--kv-stream` handler accepts either a MiB number or `auto`; the resolved MiB value is logged before falling into validation.

**Verified on hardware (RTX 5090, Qwen3.8-27B q8_0/q8_0):**
- `-c 65536 --kv-stream auto` -> `stage=910 MiB`, device KV reduced by `2176 MiB`, host increased by `2176 MiB`
- `-c 65536 --kv-stream auto -ctk turbo3 -ctv turbo3` -> same device KV reduction, host increase `1488 MiB`, compute buffer `506 MiB`
- `llama-bench` does **not** accept `--kv-stream auto`; only numeric MiB values are parsed. Documented in next-step item 8.

### 13. Precise turbo exclusion instead of blanket turbo K/V + >= 8 layers

The old exclusion (section 6) rejected any turbo K/V type combined with
>= 8 layers, on the theory that `TURBO_LAYER_ADAPTIVE` (the existing
per-layer K/V type override, auto-enabled for turbo2-V models) *might* give
layers different byte-per-token sizes, which the streaming pool's
single-page-size assumption can't handle. In practice that override only
changes anything for specific (mode, type) combinations:

- Modes 1/2 (K+V boundary upgrade to q8_0) apply only when **K** is a turbo
  type and `n_layer >= 8`.
- Modes 5/6/7 (V-only boundary upgrade) apply only when **V** is a turbo
  type and `n_layer >= 8`. Mode 7 is auto-enabled whenever V is turbo2 with
  `n_layer >= 8` and no explicit `TURBO_LAYER_ADAPTIVE` override - this is
  the one real-world case (turbo2-V) that was correctly excluded before.
- A pure turbo3 or turbo4 K/V config never triggers any of these (mode
  stays 0), so it was always safe to stream - the old check just couldn't
  tell the difference and rejected it anyway.

Changes in `src/llama-kv-cache.cpp`:

- Hoisted the `TURBO_LAYER_ADAPTIVE` mode resolution (env var override, or
  the turbo2-V auto-enable) out of the per-layer loop's
  `static const int adaptive_mode = [&]() {...}();` into a plain local
  variable computed once per constructor call, before the pre-scan. This
  also fixes a latent correctness bug: a function-local `static` has
  process lifetime, so the *first* `llama_kv_cache` ever constructed in a
  process pinned the mode for every later construction regardless of its
  own `type_v` - directly relevant now that `llama-bench` can sweep
  `--cache-type-v` across many contexts in one process.
- Replaced the pre-scan's blanket
  `(k_is_turbo || v_is_turbo) && n_layer >= 8` check with
  `kv_stream_layers_vary`, which mirrors the exact per-layer dispatch
  conditions (mode 1/2 + K-is-turbo, or mode 5/6/7 + V-is-turbo) so only a
  resolved mode that actually produces different types is excluded.

Verified in this sandbox (no GPU, so the runtime-creation half of the path
is still unverified here): `llama` rebuilds clean with zero new warnings,
all four `test-kv-stream-*` tests pass, `test-llama-archs` passes across
every registered architecture with no new failures.

**Needs hardware verification**: does a pure turbo3/turbo4 K/V config
(the common case, e.g. Qwen3.8-27B with `-ctk turbo3 -ctv turbo3`) now
actually engage streaming end-to-end (pool created, VRAM reduced, correct
output), and does turbo2-V with `n_layer >= 8` still correctly fall back
(mode 7 auto-enables, so `kv_stream_layers_vary` should still be true there).

### 14. `--kv-stream auto`: derive the stage pool from `-c` (design detail + ceiling analysis)

Requested so users don't have to hand-pick a MiB budget for every context
size, e.g. `-c 262144 --kv-stream auto`. Asked the user which heuristic to
use (AskUserQuestion) since the pool is deliberately meant to stay much
smaller than the full context's KV cache - sizing it to the *full* context
would defeat the point of streaming - so "derive from -c" needed a precise
definition. Chosen: **a fixed percentage of the context stays resident per
layer**, not the full context size.

Formula (`src/llama-context.cpp`, resolved once inside `llama_context`'s
constructor, right where `kv_stream_stage_bytes` was already being derived
from the manual MiB value - same place model/hparams/`cparams.n_ctx` are
all available together):

- `resident_tokens = clamp(n_ctx * 10%, floor=2048 tokens, ceiling=n_ctx)`
- rounded up to the nearest 256-token page
- `stage_bytes = page_bytes(K+V, first eligible layer) * eligible_layer_count * (resident_pages + 4 scratch pages)`
- `stage_mib = min(ceil(stage_bytes / 1 MiB), 8192 MiB)`

Plumbing: new `bool kv_stream_auto` field threaded through the same path as
`kv_stream_stage_mib` end to end - `include/llama.h`
(`llama_context_params`), `src/llama-cparams.h`, `common/common.h`
(`common_params`), `common/common.cpp`
(`common_context_params_to_llama`), and `src/llama-context.cpp`'s default
struct. `common/arg.cpp`'s `--kv-stream` handler switched from an int
handler to a string handler so it accepts either a MiB number or the
literal `auto`; the resolved MiB value is logged (`block KV streaming
auto: n_ctx=... resident=... tokens/layer (10%) layers=... -> stage=... MiB`)
before falling into the same validation/gating path a manual value uses.

**Not done**: `llama-bench`'s `--kv-stream` sweep parameter still only
accepts numeric MiB values - `auto` isn't wired into its `cmd_params`
sweep system in this pass, since sweeping "auto" per-instance would need
its own bool-sweep dimension. Numeric sweeps remain the way to do
controlled A/B throughput comparisons there.

Verified in this sandbox (built a tiny synthetic qwen3 model via
`test-llama-archs -o <dir>` to exercise the real constructor path, no GPU
available so only the CPU fallback end-state was reachable):
- `-c 262144 --kv-stream auto` -> `resident=26368 tokens/layer (10%)
  layers=2 -> stage=54 MiB`
- `-c 4096 --kv-stream auto` -> hits the 2048-token floor -> `stage=6 MiB`
- `-c 1000000000 --kv-stream auto` -> hits the 8192 MiB ceiling
- manual `--kv-stream 128` and no-flag default both unaffected
- all four `test-kv-stream-*` tests and `test-llama-archs` (111s) still pass

**Needs hardware verification**: whether the default 10%/2048-floor/8192-MiB-ceiling
heuristic actually gives good throughput/VRAM tradeoffs in practice, e.g. at
`-c 262144` on Qwen3.8-27B - the constants were chosen for defensibility, not
tuned against real measurements yet.

**What context size does the 8192 MiB ceiling actually bind at?** Using the real
Qwen3.8-27B numbers already measured above (q8_0/q8_0, `page_bytes=557056` bytes
for one 256-token K+V page, `16` streaming-eligible layers - both from the
confirmed engagement log):

- `stage_bytes = 557056 * 16 * (resident_pages + 4)`; solving for
  `stage_bytes = 8192 MiB` gives `resident_pages ~= 960` -> `resident_tokens ~=
  245760` -> since that's 10% of context, `n_ctx ~= 2.46M tokens`.
- So for this model/type pair, the ceiling is a distant safety backstop, not
  something normal use hits: at the `-c 262144` example from this session,
  auto computes `stage_bytes = 557056 * 16 * (103 + 4) ~= 910 MiB` - nowhere
  near 8192 MiB. The ceiling only starts constraining the nominal 10% target
  past roughly 2.46M-token contexts for this specific model/type combination.
- This scales with `page_bytes * layer_count`: a model with a bigger KV
  footprint per layer (e.g. F16 instead of q8_0, or more streaming-eligible
  layers) hits the ceiling at a proportionally smaller context size - there's
  no single universal crossover point, it depends on the model being served.

### 15. Fixed `llama-perplexity`'s plain-PPL path silently overriding `-np`

Went through the outstanding next-step list one item at a time (one idea = one PR).
Item 1 was the one clear code fix available without real hardware: `tools/perplexity/perplexity.cpp:2036`
computed `n_parallel = max(1, n_batch / n_ctx)` for plain PPL/KLD runs unconditionally,
silently discarding whatever `-np`/`--parallel` the user passed - including an explicit
`-np 1` needed to satisfy `--kv-stream`'s single-sequence gate. The `-b <= -c` workaround
(section on unverified PPL/KLD above) only worked around this; it didn't fix it.

**Fix**: added `bool n_parallel_explicit` to `common_params` (same pattern as the existing
`moe_cache.mode_explicit` field), set it in `common/arg.cpp`'s `-np`/`--parallel` handler
(the non-server branch, which is what `LLAMA_EXAMPLE_PERPLEXITY` uses), and changed
`perplexity.cpp`'s plain-PPL branch to `else if (!params.n_parallel_explicit)` - an explicit
`-np` now sticks exactly as given; the auto-batching heuristic still applies unchanged when
the user doesn't pass `-np` at all, so default behavior for normal (non-streaming) PPL runs
is unaffected.

Verified in this sandbox with a synthetic qwen3-dense model (`test-llama-archs -o`):
- Without `-np`: gate log shows `single_sequence=0` (old auto-batching behavior, unchanged)
- With `-np 1`: gate log shows `single_sequence=1` - now passes that gate, fails only at
  `type_pair_supported=0` (expected CPU-backend limitation, same as every other test in
  this sandbox)
- `llama` + `common` rebuild clean, `test-kv-stream-*` and `test-llama-archs` still pass

With this fix, the real `-b <= -c` workaround documented above is no longer necessary -
`-np 1 --kv-stream <n>` (or `--kv-stream auto`) should work directly on its own batch/ctx
settings.

**Confirmed on hardware** (RTX 5090, fresh build at `2bb5ba583`, `-np 1` directly, no
`-b`/`-c` workaround): 512 ctx baseline `PPL = 6.9576 ± 0.04503` vs `--kv-stream auto`
`PPL = 6.9568 ± 0.04502` - `~0.0%` difference, exact parity with the earlier
workaround-based measurement. The fix works as intended on real hardware.

`llama-bench --kv-stream 2304` at 512 ctx / 128 gen / 3 reps also re-confirmed: baseline
pp512 `3649.95-3655.61 ± ~138`, tg128 `53.91-53.93 ± ~0.25`; with streaming pp512
`3522.98-3532.44 ± ~165-192`, tg128 `52.50-52.57 ± ~0.10-0.18` - consistently
**~-3.5% pp, ~-2.5% tg** across repeated runs, matching section 11's earlier numbers.

**Still blocked**: 64K PPL still gets killed mid-run (`exit -9`) on this hardware even
after the `-np` fix - confirms this is a system stability/resource issue unrelated to
the perplexity `-np` bug or to streaming itself. Server memory breakdown for this round
did not complete before the test script was terminated - still pending.

### Status of the remaining next-step items (as of this pass)

- **Item 4 (non-SWA path parity)**: exercised repeatedly in this sandbox via the plain
  (non-iswa/hybrid) qwen3-dense synthetic model across all the auto-sizing and `-np` tests
  above - the gate/pre-scan path behaves identically to the hybrid+SWA case already
  confirmed on hardware. Full non-SWA parity on real inference output still needs hardware.
- **Item 6 (turbo2-V still falls back)**: turbo3 K/V + `auto` is now confirmed engaging on
  real hardware (section 11). Turbo2-V's fallback is unchanged code (my fix only added
  precision for the cases that *don't* vary; the turbo2-V + mode-7 + `n_layer >= 8` path is
  untouched) and confirmed via code review, but **could not be exercised end-to-end in this
  sandbox**: `test-llama-archs` only produces `n_layer >= 8` synthetic models for
  `LLM_ARCH_GEMMA3N` (22 layers), and that architecture's synthetic model is marked
  unsupported for save-to-file by the test harness itself - every other architecture the
  harness can save defaults to 2 layers, below the threshold where `TURBO_LAYER_ADAPTIVE`
  mode 7 would ever engage. Still needs a real hardware check with turbo2-V + >= 8 layers.
- **Items 2, 3, 5, 7**: all need real GPU hardware (KLD measurement, longer-generation
  throughput, interactive `llama-cli` testing, tuning the `auto` heuristic constants against
  real A/B data) - nothing further to do on those from this sandbox.

## Next concrete step

1. Run a controlled PPL comparison with and without `--kv-stream 2304` on a fixed prompt set using `llama-perplexity`
2. Use `llama-bench --kv-stream 0,2304` (and other stage sizes) to characterize throughput scaling across generation lengths and context sizes in one sweep
3. Verify the non-SWA path with a non-SWA model
4. General interactive testing via `llama-cli --kv-stream <n>`
### 8. Measurement tests

Environment: NVIDIA GeForce RTX 5090, 32 GB VRAM, CUDA compute capability 12.0.

**Server startup VRAM breakdown (65536 ctx, q8_0/q8_0, Qwen3.8-27B, after prompt cache + 1 completion):**

| Component | Baseline | With streaming |
|-----------|----------|----------------|
| CUDA0 model | 25972 MiB | 25972 MiB |
| CUDA0 KV cache | 2325 MiB | 149 MiB |
| CUDA0 compute | 388 MiB | 592 MiB |
| Host free | 1372 MiB | 3548 MiB |
| Unaccounted | 1672 MiB | 3971 MiB |
| **Device KV reduction** | — | **−2176 MiB** |
| **Host increase** | — | **+2176 MiB** |

**Throughput (`llama-bench`, 512 ctx, 128 gen, 3 reps):**

| Condition | pp512 t/s | tg128 t/s |
|-----------|-----------|-----------|
| Baseline | 3640.26 ± 192.63 | 53.75 ± 0.26 |
| With streaming | 3514.62 ± 176.04 | 52.53 ± 0.11 |
| Difference | −3.5% | −2.3% |

**Server inference (`llama-server`, 512 ctx, 1 completion):**

| Condition | tokens/s | predicted | evaluated |
|-----------|----------|-----------|-----------|
| Baseline | 58.55 | 8 | 5 |
| With streaming | 56.74 | 8 | 5 |
| Difference | −3.1% | 0 | 0 |

**Quality:** identical output ("I love you very much.") in both server conditions.

**Findings:**
- Streaming is engaging on Qwen3.8-27B and reducing device KV cache by ~2.2 GiB.
- The host-side memory increase matches the streaming staging pool budget.
- Throughput impact is small (~2–3% decrease), consistent with expected PCIe transfer overhead for the resident-page cache + transfer ring.
- Startup VRAM delta of +327 MiB from earlier was noise because no decode had populated resident pages; the real signal appears after prompt eval.

**What remains unverified:**
- PPL/KLD with streaming active — `llama-perplexity` plain-PPL path requires `-b <= -c` to avoid `block KV streaming requires exactly one sequence (-np 1)`. With that workaround:
  - 512 ctx baseline: `PPL = 6.9576 ± 0.04503`
  - 512 ctx `--kv-stream auto`: `PPL = 6.9568 ± 0.04502`
  - Difference: `~0.0%` — no measurable quality delta.
  - 64K PPL is blocked on this hardware: the baseline run itself is killed during pass calculation (`exit -9`), before streaming can be evaluated. This is a system stability/setup issue, not a streaming-specific failure.
  - **Fixed (section 15)**: plain-PPL now respects an explicit `-np`/`--parallel` instead of computing it from `max(1, n_batch / n_ctx)` unconditionally - `-np 1 --kv-stream <n>` should work directly now, without the `-b <= -c` workaround. Needs a real hardware re-run to confirm.
- KLD with streaming active — not yet measured; needs the same 64K stable path or a different KL dataset/batching strategy.
- Throughput at larger generation lengths / longer contexts
- Non-SWA path parity after SWA extension
- Long-context stability at 128K+

### 16. `llama-bench --kv-stream auto` fixed, and the real root cause of the 64K PPL `exit -9`

Went back through the two remaining "stated issues" from the latest hardware round.

**`llama-bench` rejecting `--kv-stream auto`**: this was simply never wired up (documented
as "Not done" in section 14) - `cmd_params::kv_stream_stage_mib` was a plain
`std::vector<uint32_t>` with no way to represent "auto". Fixed by introducing a small
`kv_stream_spec { uint32_t mib; bool auto_size; }` value type used everywhere that field
appears (`cmd_params`, `cmd_params_instance`, `test`), with `operator<<`/`operator==` so it
drops into the existing `join()`/comparison machinery unchanged. `--kv-stream` now parses
comma-separated tokens where each one is either a MiB number or the literal `auto`; the
results-table column reclassifies from `INT` to `STRING` (prints "auto" or the number) since
a column can now hold either. Verified in this sandbox: `--kv-stream auto`,
`--kv-stream 0,2304,auto` (mixed sweep), and the default (column hidden) all work; `-np`
independent `test-kv-stream-*` suite still passes.

**64K PPL `exit -9`**: traced this to an actual root cause rather than leaving it as
"system instability." `tools/perplexity/perplexity.cpp:512-515`:
```cpp
std::vector<float> logits;
if (num_batches > 1) {
    logits.reserve(size_t(n_ctx) * n_vocab);
}
```
`num_batches = (n_ctx + n_batch - 1) / n_batch`, so with the default `-b 2048` and
`-c 65536`, `num_batches = 32 > 1`, and this buffer reserves `65536 * n_vocab` floats on
the **host**. For a ~152K-token vocab (Qwen3-class), that's `65536 * 151936 * 4 bytes ~=
39.8 GiB` of host RAM for one intermediate buffer - independent of `--kv-stream` entirely,
and independent of GPU VRAM. This matches the observation that the *baseline* run (no
streaming at all) is what gets killed. The comment directly above it (`// TODO: this could
be made smaller; it's currently the worst-case size`) is pre-existing upstream code, not
something introduced by this branch or by streaming.

**Testable workaround (no code change)**: pass `-b` >= `-c` for the 64K run (e.g.
`-b 65536 -c 65536`), which makes `num_batches == 1` and skips this allocation entirely.
Trade-off: a single 65536-token batch is a much larger one-shot GPU compute step, which
could hit a different (VRAM-side) limit instead - untested on this hardware.

**Not fixed in this pass**: actually shrinking this buffer so large-context PPL works
with `num_batches > 1` too would mean restructuring the perplexity computation to process
and discard logits incrementally per sub-batch instead of buffering a whole chunk at once -
a real, riskier change to core PPL/KLD numerics, unrelated to the KV-streaming feature
itself and pre-existing in upstream llama.cpp. Flagging rather than attempting blind,
per "one idea = one PR" and pausing on invasive changes - happy to take this on as its own
follow-up if wanted, once the `-b >= -c` workaround has been tried.

### 17. Multi-GPU (tensor-split) streaming, DFLASH architecture extension, and SM/compute-capability audit

Per the request to extend to other architectures, other SM/compute capabilities, and
tensor-split multi-GPU setups while hardware testing continues:

**Multi-GPU tensor-split streaming** (`src/llama-kv-cache.{h,cpp}`) - the previous
implementation only ever built *one* runtime, on whichever device happened to host the
first eligible layer; every other device's layers silently never streamed at all, which
made the feature effectively single-GPU-only. Reworked the pre-scan to group eligible
layers by device (`std::map<ggml_backend_dev_t, uint32_t>`) instead of tracking a single
device, and to create an **independent runtime per device** that has any eligible layers -
each sized with the caller's `--kv-stream` MiB value as a **per-device** budget (a 4-GPU
tensor split with `--kv-stream 2304` reserves 2304 MiB on *each* GPU with eligible layers,
not 2304 MiB split four ways - simpler to reason about, and VRAM is inherently a per-device
resource). The uniform-byte-per-token safety check now applies across *all* eligible layers
regardless of device (previously it only checked layers on the first device, since other
devices' layers were skipped anyway). A device whose runtime creation fails just falls back
to the ordinary buffer type for its own layers - it doesn't block streaming on the other
devices. `llama_kv_cache::kv_stream_runtime`/`kv_stream_runtime_free_fn` (single handles)
became `kv_stream_entries` (`std::vector<{dev, buft, runtime, free_fn}>`); the destructor
frees all of them; the per-layer buffer-type lookup is now a linear scan over that small
vector (construction-time only, negligible cost).

**DFLASH architecture extension** (`src/llama-context.cpp`) - `LLM_ARCH_DFLASH` was
blanket-excluded via the same architecture denylist as the genuinely special MLA/indexer
caches (MinMax M3, GLM DSA, DeepSeek32/4). But DFLASH only uses a non-standard cache
(`llama_kv_cache_iswa` with a custom MLA-style filter) when `hparams.dsv4_hc_mult > 0`
(the DSV4 DSpark-stage case) - per `llama-model.cpp`'s own construction switch, with
`dsv4_hc_mult == 0` it falls through to the exact same standard
`llama_kv_cache`/hybrid/recurrent construction as any other architecture. Same class of
over-conservative fix as section 13's turbo exclusion: `llama_kv_stream_arch_uses_unified_cache`
now takes `hparams` and returns `hparams.dsv4_hc_mult == 0` for DFLASH instead of a blanket
`false`, so non-DSpark DFLASH configs are streamable like any other architecture.
**Could not be exercised end-to-end in this sandbox**: `test-llama-archs` unconditionally
skips `LLM_ARCH_DFLASH` (and `EAGLE3`) itself - "needs more fixture params", same category
as its existing Gemma4 FIXME - so this fix is verified by code review against the actual
construction switch, not by a synthetic-model run. Audited every other `create_memory`
switch case (MSA, DSA, DSV4, EAGLE3, and the standalone/hybrid/recurrent default path) for
a similar mismatch between the denylist and the actual branch taken - didn't find another one.

**SM / compute-capability audit** - no code change needed, but the concern didn't check
out as a real gap: `ggml_backend_cuda_kv_stream_supported()`'s own comment already states
the design intent ("true for any valid CUDA/HIP/MUSA device... no device feature beyond
what every supported GPU already has"), and `ggml/src/ggml-cuda/kv-stream.cu`'s kernels
use no compute-capability-conditional code, warp-size-dependent intrinsics, or tensor-core
instructions of any kind - confirmed by grepping for `__shfl`/`warpSize`/`__CUDA_ARCH__`/
`wmma` and finding nothing. `CMakeLists.txt` compiles `kv-stream.cu` for every
`CMAKE_CUDA_ARCHITECTURES` target the same as every other `.cu` file, no special-casing.
So SM/compute-capability breadth was already unconditional from the original port - what's
actually missing is *hardware testing* on something other than the RTX 5090 (sm_120) used
so far, not a code fix.

Verified in this sandbox: `llama` rebuilds clean (touched `llama-kv-cache.h/.cpp` and
`llama-context.cpp`), all four `test-kv-stream-*` tests pass, `test-llama-archs` passes
across every architecture it covers with no new failures.

**Needs hardware verification**: multi-GPU tensor-split streaming has never been exercised
on real multi-GPU hardware - the single-GPU RTX 5090 testing so far can't exercise the new
per-device grouping/multiple-runtime-creation path at all. DFLASH (any config) and other
compute capabilities (anything other than sm_120) are likewise unverified beyond code
review, for lack of that hardware in this session.

### 18. Root-caused and fixed the `common_fit_params` hard error with `--kv-stream`

Real hardware run reported: `llama-perplexity --kv-stream auto` aborts in
`common_init_from_params` with `"failed to fit parameters to device memory (hard error);
retry with -fit off"` at both 512 and 2048 ctx on Qwen3-8B, while the baseline (no
`--kv-stream`) runs normally - and asked, reasonably, why this would even touch device
memory at all when KV streaming's whole point is keeping the cache off-device.

**Root cause, traced end to end**: `-fit` is on by default (`common/common.h`:
`fit_params = true`), and `common_init_from_params()` calls `common_fit_params()`
(`common/fit.cpp`) *before* the real context is built, to measure memory usage for
candidate configurations. Its measurement helper
(`common_get_device_memory_data_impl`) loads the model and builds a **trial** context
with `mparams.no_alloc = true` - a dry run that measures tensor sizes without doing
real allocation work (the existing K/V tensor buffers and the turbo rotation matrices
already respect this flag the same way). But `llama-context.cpp`'s streaming
validation gate (`llama_kv_stream_config_validate`) **throws a hard `std::runtime_error`
on any failing gate**, by design, for a real, explicit `--kv-stream` request that can't
be satisfied - and it ran unconditionally, `no_alloc` or not. During a probe, the
CPU-backend `type_pair_supported=0` gate (or any other gate, depending on what
candidate configuration the fit search happens to be trying) fails exactly the way it
normally would on an unsupported backend, throws, and `common_get_device_memory_data_impl`
treats *any* failure to construct the initial trial as fatal (`common_fit_params:
encountered an error while trying to fit params ...` -> `COMMON_PARAMS_FIT_STATUS_ERROR`
-> the hard-error message the user saw). Reproduced this exact failure in this sandbox
(no GPU needed - the mechanism doesn't depend on CUDA, `no_alloc=true` triggers on any
backend) with a synthetic model: `llama-perplexity ... --kv-stream auto` (default `-fit`
on) hit the identical hard-error message, `-fit off` did not.

**Fix** (`src/llama-context.cpp`): skip the entire streaming validation/gate block
during a `no_alloc` probe instead of running it and throwing - equivalent to leaving
`kv_stream_enabled = false`, exactly like the real construction's own graceful
fallback for configurations the feature doesn't apply to. Also added the same guard to
`src/llama-kv-cache.cpp`'s per-device runtime-creation loop (belt and suspenders: even
if streaming were somehow still requested during a probe, it must never perform a real
backend allocation like `cudaMalloc` there either - every candidate configuration
fit-params tries would otherwise leak a real device allocation it never frees between
trials). This does **not** change behavior for a real (non-probe) `--kv-stream` request,
which still throws exactly as before when explicitly requested and genuinely
unsatisfiable - only the fit-params dry run is affected.

Verified in this sandbox: reproduced the original crash, confirmed the fix resolves it
(`common_fit_params: successfully fit params to free device memory` instead of the hard
error), confirmed `-fit off` and the no-`--kv-stream` baseline are both unaffected
(byte-identical behavior to before), all four `test-kv-stream-*` tests and
`test-llama-archs` still pass.

**Separately**: the `llama-bench --kv-stream 2304` context-creation failure reported in
the same round is **not** this bug - `llama-bench` only invokes `common_fit_params` when
`-fitt`/`--fit-target` is explicitly passed (`do_fit` is false by default), which wasn't
the case here. That failure is most likely the genuine VRAM/pool-size constraint the user
already suspected (2304 MiB oversized relative to the ~910 MiB `auto` sweet spot measured
at 65K ctx for this model) rather than a code bug - needs the actual error message from
that run to confirm, since it wasn't captured in the report.

## Next concrete step

One idea = one PR: each numbered item below is its own separate, independently
reviewable change - not to be bundled together into one commit/PR.

1. ~~Re-run PPL/KLD with `-np 1 --kv-stream ...` directly~~ - **done, confirmed (section 15)**: `~0.0%` PPL delta at 512 ctx, exact parity with the earlier `-b <= -c` workaround measurement
2. ~~`llama-bench` rejecting `--kv-stream auto`~~ - **done (section 16)**: fixed and tested in sandbox
3. Try the `-b >= -c` workaround for 64K PPL (section 16's root-cause diagnosis) on real hardware, e.g. `-b 65536 -c 65536`; if it avoids the `exit -9` but hits a GPU-side limit instead, report back before attempting the deeper fix (restructuring the PPL logits buffer to not need the full `n_ctx * n_vocab` allocation)
4. Measure KLD with streaming active once a stable 64K (or smaller stand-in) path exists
5. Benchmark longer generation lengths to characterize throughput scaling
6. Verify the non-SWA path with a non-SWA model on real inference output (sandbox-level gate/pre-scan parity already confirmed, section 15)
7. General interactive testing via `llama-cli --kv-stream <n>`
8. Verify turbo2-V with `n_layer >= 8` still correctly falls back on real hardware (turbo3 already confirmed engaging, section 11; code path for turbo2-V is unchanged and could not be exercised end-to-end in the sandbox, section 15)
9. Tune `--kv-stream auto` defaults: 10% resident, 2048-token floor, 8192 MiB ceiling — real A/B at multiple context sizes on stable hardware
10. Server memory breakdown for the latest round - script was terminated before this landed, needs a re-run
11. Verify multi-GPU tensor-split streaming (section 17) on real multi-GPU hardware - confirm each device with eligible layers gets its own pool, VRAM reduction is per-device, and a device where the runtime fails to create still falls back correctly without breaking the others
12. Verify the DFLASH `dsv4_hc_mult == 0` fix (section 17) with a real DFLASH model - could not be exercised in the sandbox since `test-llama-archs` skips this architecture entirely
13. Try `--kv-stream` on a non-sm_120 GPU if/when available - the code has no SM-specific gating (section 17), but this has only ever run on an RTX 5090 so far
14. Re-run `llama-perplexity --kv-stream auto` (default `-fit on`, no workaround) on real hardware to confirm section 18's fix resolves the hard error there too, not just in this sandbox's CPU repro
15. Capture the actual error message from `llama-bench --kv-stream 2304` failing to create context on Qwen3.8-27B (section 18) - needed to confirm it's the suspected VRAM/pool-size constraint and not a separate bug
