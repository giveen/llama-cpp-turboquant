# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)
Current HEAD: `addd1d9f5 kv-cache: wire llama_kv_cache to request the streaming buffer type`

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

### 4. Server smoke test — INITIAL FAIL, THEN FIXED BY UPSTREAM
First attempt:
```
./build/bin/llama-server -m <model> -c 65536 -ngl 99 -fa on \
  -ctk turbo3 -ctv turbo3 -np 1 --kv-stream 2304
```
Result:
```
failed to initialize the context: block KV streaming is not supported
for this K/V cache type pair on the active backend
```

Root cause: the CUDA backend's `ggml_backend_reg_get_proc_address` did not export `ggml_backend_kv_stream_supported`. The capability query in `src/llama-kv-stream-backend.cpp` got `nullptr` and returned `streamable = false`.

### 5. Upstream fix landed in `d83bc4f47`
Pulled the latest from origin. Commit `d83bc4f47` ("cuda: register KV streaming capability symbols") added the missing proc-address registrations:
- `ggml_backend_kv_stream_supported` → `ggml_cuda_fattn_kv_type_supported`
- `ggml_backend_kv_stream_type_pair_supported` → `ggml_cuda_fattn_kv_type_pair_supported`

Rebuilt and re-ran the server smoke test:
```
./build/bin/llama-server -m Qwen3.8-27B-Q6_CR.gguf \
  -c 65536 -ngl 99 -fa on -ctk turbo3 -ctv turbo3 -np 1 \
  --kv-stream 2304 --port 0
```
Result: model loaded, initialized, listening on port. No streaming rejection. ✅

The `TURBO_AUTO_ASYMMETRIC=1` path upgraded K from turbo3 to q8_0 for this model's GQA ratio (6:1), which is expected behavior and does not affect the streaming gate.

### 6. `llama_kv_cache` wired to actually request the streaming buffer type — commit `addd1d9f5`

Before this commit, `--kv-stream` validated and the server started (`d83bc4f47`), but nothing downstream requested the pinned streaming buffer - the KV cache was still ordinary VRAM storage regardless of the flag. `addd1d9f5` closes that gap:

- `llama_kv_cache`'s constructor now does a pre-scan before its layer loop (when `kv_stream_stage_bytes > 0` and offloaded): picks a target device and counts streaming-eligible layers on it, resolves `ggml_backend_kv_stream_runtime_new_for_device` / `_buffer_type` / `_free` off that device's backend registry (same `ggml_backend_reg_get_proc_address` plugin pattern used elsewhere in this codebase), and if that succeeds, creates one runtime shared by those layers and uses its pinned buffer type instead of the ordinary device buffer type for their K/V tensors. Any failure at any step (backend doesn't export the symbols, allocation/layout failure) falls back silently to today's behavior with a `LLAMA_LOG_WARN`.
- Threaded `kv_stream_stage_bytes` through `llama_memory_params` → `llama_memory_hybrid` → the two real call sites in `llama-model.cpp`'s architecture switch that can actually reach `unified_kv_cache == true`: the plain default (non-hybrid, non-SWA) case, and the non-SWA hybrid case (e.g. Qwen3.5-style models). The DSA/DSV4/MSA MTP-context call site needed no change - those architectures are already excluded from `unified_kv_cache` upstream, so the new parameter's default (0) is always correct there.
- Added a generic `ggml_backend_kv_stream_runtime_new_for_device` wrapper (plain scalar args: device index, pool_bytes, page_bytes, stage_slots, layer_count) in `ggml-cuda/kv-stream.cu`/`.cuh`, since the existing constructor took a CUDA-internal params struct that generic (non-CUDA) code can't reference. Registered it plus `_free`/`_buffer_type` in `ggml-cuda.cu`'s proc-address table.
- **Important safety exclusion**: streaming is skipped whenever K or V is a turbo type (`turbo2_0`/`turbo3_0`/`turbo4_0`) *and* the model has ≥ 8 layers. Reason: the already-shipping `TURBO_LAYER_ADAPTIVE` override (auto-enabled by default whenever V is `turbo2_0` on ≥ 8 layers - see the mode-dispatch block in `llama-kv-cache.cpp`, around where `layer_type_k`/`layer_type_v` are resolved per layer) can give different layers different K/V types, and hence different bytes-per-token. The streaming pool assumes one uniform page size across every layer it manages, so mixed-size layers would corrupt that assumption. The exclusion is deliberately the coarse, *provably*-safe condition (that's the only condition under which the adaptive override changes anything) rather than replicating its ~50-line dispatch table to predict the outcome exactly - **this means streaming currently will not engage for the turbo K/V + ≥8-layer combination, which is plausibly your main intended use case.** Relaxing this safely (either by making the streaming pool support per-layer page sizes, or by precisely predicting the adaptive dispatch outcome with its own test coverage) is real follow-up work, not something to do without a compiler and a test in hand.
- Streaming currently only targets a single device: if a model's layers are split across multiple GPUs, only the layers on the *first* streaming-eligible layer's device are considered; layers on other devices keep the ordinary buffer type.
- Also requires `!hparams.is_n_embd_k_gqa_variable() && !hparams.is_n_embd_v_gqa_variable()` (uniform head dims across layers) for the same uniform-page-size reason.

Verified in the sandbox that produced this commit (no GPU there): the `llama` library rebuilds clean, all four `test-kv-stream-*` tests still pass, and `test-llama-archs` (builds tiny synthetic models and runs real forward passes across many architectures through the actual model-loading path) passes in ~103s with the default `kv_stream_stage_bytes = 0` - meaning the constructor change doesn't disturb ordinary (non-streaming) KV cache construction anywhere in the architecture switch. **Nothing about whether streaming actually engages, allocates correctly, or produces correct output has been verified on real hardware yet.**

## Current code changes committed
- `ggml/src/ggml-cuda/ggml-cuda.cu` — proc-address table exports `ggml_backend_kv_stream_supported`, `_type_pair_supported`, `_runtime_new_for_device`, `_runtime_free`, `_buffer_type`
- `ggml/src/ggml-cuda/kv-stream.cu` / `.cuh` — the pinned buffer type, device pool, resident-page cache, transfer ring, online-softmax fixup kernels, and the generic `_runtime_new_for_device` wrapper
- `ggml/src/ggml-cuda/fattn.cu` / `.cuh` — generalized (K,V) type-pair support check, reused by the existing dispatch
- `src/llama-kv-stream-{config,plan,softmax,backend}.{h,cpp}` — the backend-agnostic planning layer and capability-query plumbing
- `src/llama-kv-cache.{h,cpp}`, `src/llama-memory.h`, `src/llama-memory-hybrid.{h,cpp}`, `src/llama-model.cpp`, `src/llama-context.cpp` — the constructor wiring described above
- `src/llama-kv-stream-backend.cpp` — unchanged from earlier upstream commits; no debug logging present

## What has NOT been done yet
- **Not tested on real hardware at all**: the constructor wiring in `addd1d9f5` has never run against an actual CUDA device. This is the most important open item.
- No benchmark runs with `--kv-stream` (see "what to test" below for what would actually be meaningful now)
- No tensor-creation-through-GGML-backend adaptation beyond what's described above
- No asymmetric K/V planner extension
- No relaxation of the turbo/≥8-layer exclusion or the multi-GPU/variable-head-dim limitations
- Vulkan and Metal execution backends were never started - everything so far is CUDA/HIP/MUSA only (the three backends that share `ggml-cuda`'s source tree)

## What needs to be tested next

1. **Build with `-DGGML_CUDA=ON` and confirm it still compiles.** This is the first time the constructor-wiring commit (`addd1d9f5`) has touched a real CUDA compiler - the CUDA-side additions (`runtime_new_for_device` wrapper, proc-address registrations) are new since the last verified build.

2. **Server smoke test, same as before**, but now watch for whether streaming actually *engages* rather than just validates:
   ```
   ./build/bin/llama-server -m <model> -c 65536 -ngl 99 -fa on \
     -ctk q8_0 -ctv q8_0 -np 1 --kv-stream 2304
   ```
   Use a non-turbo type pair (e.g. `q8_0`/`q8_0`) or a model with < 8 layers for this first test, since turbo + ≥8 layers is deliberately excluded right now (see above) - a turbo-type test would currently exercise the *fallback* path, not the streaming path, which is worth testing separately but shouldn't be the first test.
   - Look for the `LLAMA_LOG_WARN` fallback message ("block KV streaming requested but not available on this backend/device") - if it appears when you expected streaming to engage, something in the pre-scan or symbol resolution is failing and needs a real backtrace/log to diagnose.
   - If it does *not* appear, streaming should be live for those layers.

3. **VRAM check**: compare `nvidia-smi` memory usage at a given `-c` with and without `--kv-stream`. This time a *difference* is the expected, correct signal (less VRAM used for the KV cache portion when streaming is active, since it's now genuinely in pinned host memory) - the opposite of the "should be identical" check from before this commit.

4. **Correctness check before any perf number is trusted**: run a short completion and sanity-check the output isn't garbage. The resident-page cache and transfer ring in `kv-stream.cu` were written and reviewed carefully but never executed - if there's a subtle bug in the page-copy/event choreography, garbled or repetitive output is the most likely symptom, not a crash.

5. **Only after 2-4 look correct**: throughput (`llama-bench` or manual token/s) and ppl/KLD comparison with and without `--kv-stream`, which is now a meaningful comparison for the first time.

6. **Test the turbo/≥8-layer fallback explicitly**: confirm a turbo K/V + ≥8-layer model still starts and runs correctly with `--kv-stream` set (falling back to the ordinary cache, not erroring) - this is the safety exclusion working as intended, and should be a silent, unremarkable no-op from the model's perspective (only the log line differs).
