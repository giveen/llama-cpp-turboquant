# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)

## What was done

### 1. Branch creation and checkout
Created `feature/adaptive-kv-stream-integration` from `feature/turboquant-kv-cache` and pushed to origin. Later switched to the upstream-prepared branch `claude/adaptive-kv-stream-3jjugs` at commit `8f3ba1d7d` per the contributor's instructions.

### 2. Build with `-DGGML_CUDA=ON`
CMake configure succeeded. Full build was killed by OOM at [463/709]. Targeted rebuild of the four `test-kv-stream-*` executables succeeded, and `llama-server` was rebuilt separately.

### 3. Test results (as directed)
`ctest -R test-kv-stream` returned no tests (the kv-stream tests are standalone executables, not CTest-registered). Ran them directly:

| Test | Result |
|------|--------|
| `test-kv-stream-plan` | 35 tests, 232 assertions, 0 failures ✅ |
| `test-kv-stream-config` | 5 tests, 19 assertions, 0 failures ✅ |
| `test-kv-stream-softmax` | 3 tests, 21 assertions, 0 failures ✅ |
| `test-kv-stream-backend` | 2 tests, 5 assertions, 0 failures ✅ |

All four pass on CPU. No GPU required for these tests.

### 4. Server smoke test — FAIL
```
./build/bin/llama-server -m <model> -c 65536 -ngl 99 -fa on \
  -ctk turbo3 -ctv turbo3 -np 1 --kv-stream 2304
```
Result:
```
failed to initialize the context: block KV streaming is not supported
for this K/V cache type pair on the active backend
```

## Root cause

The CUDA backend's `ggml_backend_reg_get_proc_address` does not export `ggml_backend_kv_stream_supported`. The capability query in `src/llama-kv-stream-backend.cpp` calls `ggml_backend_reg_get_proc_address(reg, "ggml_backend_kv_stream_supported")`, gets `nullptr`, and returns `streamable = false`. The config validator then rejects the request before any CUDA code runs.

The type-pair check (`ggml_backend_kv_stream_type_pair_supported`) was already wired up and points to `ggml_cuda_fattn_kv_type_pair_supported`, which correctly accepts turbo3/turbo3, q8_0/q8_0, and mixed turbo+q8_0 pairs. The missing piece is the device-capability gate.

## Current code changes

### `ggml/src/ggml-cuda/ggml-cuda.cu`
- Added `ggml_backend_kv_stream_type_pair_supported` → `ggml_cuda_fattn_kv_type_pair_supported` to the proc-address table. **Working.**
- Added `ggml_backend_kv_stream_supported` → forward-declared `ggml_backend_cuda_kv_stream_supported` to the proc-address table. **Broken:** the function body was removed after a compilation error (`info.devices.size()` is not a method; `ggml_cuda_device_info` uses a fixed-size `devices[GGML_CUDA_MAX_DEVICES]` array). The forward declaration is in the table but the definition is missing, so `ggml-cuda` does not compile.

### `src/llama-kv-stream-backend.cpp`
- Added debug `fprintf(stderr, ...)` logging to trace `type_k`, `type_v`, `supported_fn`, `type_pair_fn`, and `type_pair_supported` values during the capability query. **Should be removed before merge.**

## Blocker

`ggml_backend_cuda_kv_stream_supported` needs a real definition that checks whether the device has VMM (`info.devices[id].vmm`) and FlashAttention compiled in (`#ifdef FLASH_ATTN_AVAILABLE`), using `GGML_CUDA_MAX_DEVICES` as the array bound. The forward declaration in the proc-address table must remain; only the function body needs to be added at file scope after the `ggml_cuda_info()` accessor.

## What was NOT done
- No bytes-per-token accounting fix for the planner
- No tensor-creation-through-GGML-backend adaptation
- No asymmetric K/V planner extension
- No streaming runtime integration beyond the capability-query gate
- No benchmark runs with `--kv-stream` (blocked by the above)

## Files modified in this session
- `ggml/src/ggml-cuda/ggml-cuda.cu` — proc-address table additions + broken stub
- `src/llama-kv-stream-backend.cpp` — debug logging

## Next concrete step
Write `ggml_backend_cuda_kv_stream_supported` at file scope in `ggml-cuda.cu` using `GGML_CUDA_MAX_DEVICES` and `info.devices[id].vmm`, rebuild `ggml-cuda` + `llama-server`, and re-run the server smoke test.
