# Adaptive KV Streaming Integration — Session Summary

## Branch
`claude/adaptive-kv-stream-3jjugs` on `origin` (giveen/llama-cpp-turboquant)
Current HEAD: `a611ce26e docs: update adaptive-kv-stream summary for the constructor wiring`

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

**Critical finding: streaming is NOT engaging on Qwen3.8-27B.** The model's architecture sets `kv_unified = false` (it uses SWA/non-unified cache). The streaming constructor requires `unified_kv_cache == true` AND `kv_stream_stage_bytes > 0`. Since `unified_kv_cache` is false, the pre-scan short-circuits before attempting to create the streaming runtime. The server starts cleanly, the capability query passes, but the actual streaming buffer type is never requested. This is why VRAM is identical with and without `--kv-stream`.

### Step 6: Turbo fallback
Turbo3 + ≥8 layers with `--kv-stream 2304` runs correctly. The safety exclusion at line 267 (`kv_stream_adaptive_possible`) skips streaming silently at the pre-scan level when turbo types are detected with ≥8 layers. The model falls back to ordinary VRAM KV cache without any error. Output is coherent.

## What has NOT been verified yet
- Streaming actually engaging on a model with `unified_kv_cache == true` (requires a non-SWA, non-MLA, non-DSA model)
- VRAM difference with streaming actually active
- Correctness of resident-page cache + transfer ring under real decode (output quality)
- Throughput comparison with streaming active
- PPL/KLD with streaming active

## Next concrete step
Obtain or identify a model in the local cache that uses a unified KV cache (no SWA, no MLA, no DSA/DSV4) and has < 8 layers or uses non-turbo cache types. Test `--kv-stream` on that model and verify the streaming runtime is actually created (look for pool allocation log lines, VRAM difference, and correct output). Without such a model, steps 2–5 cannot be meaningfully completed.
