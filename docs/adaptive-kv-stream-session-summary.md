# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)
Current HEAD: `d83bc4f47 cuda: register KV streaming capability symbols`

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

## Current code changes committed
- `ggml/src/ggml-cuda/ggml-cuda.cu` — proc-address table now exports both `ggml_backend_kv_stream_supported` and `ggml_backend_kv_stream_type_pair_supported` (added in upstream commit `d83bc4f47`)
- `src/llama-kv-stream-backend.cpp` — unchanged from upstream; our debug `fprintf` additions were reverted before commit

## What has NOT been done yet
- No bytes-per-token accounting fix for the planner
- No tensor-creation-through-GGML-backend adaptation
- No asymmetric K/V planner extension
- No streaming runtime benchmark with `--kv-stream`

## Next concrete step
Run `llama-bench` and a simple completion through the streaming server to verify decode throughput with `--kv-stream 2304 -ctk q8_0 -ctv turbo3`, then measure ppl and KLD at 64K context with and without streaming to establish the baseline comparison.
