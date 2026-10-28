# KV Cache Quantization with TurboQuant

TurboQuant adds seven runtime-only KV cache quantization types that compress
the K/V cache far beyond the standard `q8_0` while keeping decode quality via a
Walsh-Hadamard rotation (WHT) that Gaussianizes the cache vectors before
quantization:

| Type               | Enum                       | Size            | Compression vs f16 K/V |
|--------------------|----------------------------|-----------------|------------------------|
| `turbo2`           | `GGML_TYPE_TURBO2_0` (43)  | 2.125 bits/value | up to 7.5x            |
| `turbo3`           | `GGML_TYPE_TURBO3_0` (44)  | 3.125 bits/value | up to 5.1x            |
| `turbo4`           | `GGML_TYPE_TURBO4_0` (47)  | 4.125 bits/value | up to 3.9x            |
| `turbo5`           | `GGML_TYPE_TURBO5_0` (51)  | 5.125 bits/value | up to 3.1x            |
| `turbo6`           | `GGML_TYPE_TURBO6_0` (52)  | 6.125 bits/value | up to 2.6x            |

Blocks hold 128 values each: 34, 50, 66, 82 and 98 bytes for `turbo2` to
`turbo6`. The compression column assumes the same type for K and V; the
standard `q8_0` K + `turbo3` V mix is about 36% of f16 K/V size.

These are KV-cache-only types: they are never stored in model files.
`turbo2`/`turbo3`/`turbo4` have CPU, CUDA/HIP, Metal, Vulkan and SYCL kernels;
`turbo5`/`turbo6` currently run on CPU and CUDA/HIP only. The corresponding
model-weight quantization types are `TQ3_1S` (45) and `TQ4_1S` (46) - 3/4-bit
WHT-rotated Lloyd-Max quantization, block size 32, exposed in `llama-quantize`
as `TQ3_1S` / `TQ4_1S`.

## Usage

```bash
llama-cli -m model.gguf -c 8192 -ngl 99 \
    --cache-type-k q8_0 --cache-type-v turbo3
```

The flag parser accepts any combination of `f32`, `f16`, `bf16`, `q8_0`,
`q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1` and `turbo2` to `turbo6` for K and V.
Which combinations have backend kernels is narrower and backend-dependent.
On Metal, flash attention accepts exactly three mixed pairs: turbo K + turbo V,
`q8_0` K + turbo V, and turbo K + `q8_0` V, where turbo is `turbo2/3/4`; any
other mixed pair aborts in `ggml_metal_op_flash_attn_ext`. Mixing quantized V
with unquantized K is the common configuration.

Turbo KV types require flash attention. If a turbo cache type is requested
with flash attention disabled, it is enabled automatically (a warning is
printed) first, so a turbo V cache never reaches the quantized-V error. A
quantized V cache of another type with flash attention explicitly disabled is
an error, matching upstream behavior.

The same flags work in `llama-server`, `llama-bench`, and `llama-perplexity`.

## Model-specific quality

Models with attention sinks can be unusually sensitive to K-cache quantization. GPT-OSS is a known case: even `q8_0` K changes the output distribution substantially, and lower-bit K types degrade it further despite normal codec and kernel accuracy. Use `f16` K for GPT-OSS and other sink-heavy models. Validate a quantized V cache separately against an `f16` K/V baseline before deploying it.

Short output samples are not a sufficient quality check for this class of model because the text can remain fluent while token probabilities move significantly. Use `llama-perplexity --kl-divergence` or an equivalent logit comparison when selecting cache types.

## Rotation

K and V vectors are rotated by a fixed 128x128 orthonormal Walsh-Hadamard
matrix before quantization and inverse-rotated after dequantization. Head
dimensions that are not multiples of 128 are zero-padded to the next multiple
of 128 for turbo types. MLA models have no separate V cache (V is a view of
K), so V rotation and padding are skipped for them, and K/V cache types must
be identical.

## Environment knobs

This table is the source of truth for these variables. If it and the code ever
disagree, the code wins and both need fixing in the same PR.

| Variable                        | Default | Effect                                                          |
|---------------------------------|---------|-----------------------------------------------------------------|
| `TURBO_LAYER_ADAPTIVE`          | auto    | Layer-adaptive KV precision. Unset: Boundary V (mode 7) auto-enables when V is `turbo2` and the model has 8+ layers; `0` opts out |
| `TURBO_AUTO_ASYMMETRIC`         | `1`     | Auto-select asymmetric K/V types for large-GQA models (`0` disables) |
| `TURBO_SPARSE_V`                | `1`     | Metal: sparse-V dequant skip for `turbo3` V (`0` disables)     |
| `GGML_TQ_NATIVE`                | unset   | CUDA: `1` keeps `TQ4_1S` weights native and uses the fused TQ kernels instead of converting them to `q8_0` at load; decode about 30% faster and weight VRAM about 1.7x smaller, prefill about 2x slower |
| `GGML_CUDA_FUSE_CHAIN`          | unset   | CUDA: `0` disables elementwise chain fusion, so ADD/MUL/DIV/SCALE/CLAMP/SQR and the SILU/SIGMOID/SOFTPLUS/GELU/RELU/NEG unaries run as separate kernels |
| `GGML_CUDA_Q8CACHE`             | unset   | CUDA: `0` disables the per-graph shared `q8_1` activation cache in `mmvq`, so a repeated activation is quantized again per use |
| `LLAMA_ATTN_ROT_K_OVERRIDE` / `LLAMA_ATTN_ROT_V_OVERRIDE` | off | Enable upstream #21038 attention rotation for K / V; needs a quantized type and a head dim that is a multiple of 64 |
| `LLAMA_ATTN_ROT_K_NROT`         | `64`    | Rotation tile size for the optional upstream K path (`0` = upstream's largest power-of-two divisor of the head dim) |
| `LLAMA_ATTN_ROT_DISABLE`        | `0`     | Hard lock-out: force rotation off on both sides (`1` disables) |

`TURBO_LAYER_ADAPTIVE` is not simply `0`. With the variable unset, Boundary V
(mode 7) turns on automatically when the V cache type is `turbo2` and the model
has 8 or more layers: the first two and last two V layers become `q8_0` and
every other V layer becomes `turbo2`. Setting the variable explicitly selects
the mode; modes 5, 6 and 7 assign the V type per layer regardless of `-ctv`,
so mode 7 on a `-ctv turbo3` or `-ctv turbo4` run downgrades the non-edge V
layers to `turbo2`. The full mode legend is in `src/llama-kv-cache.cpp`
(`kv_adaptive_mode`).

Upstream attention rotation is off by default: TurboQuant manages rotation
itself (the WHT applied at cache write is equivalent and interacts with the
cache types). `LLAMA_ATTN_ROT_*` only affects the optional upstream rotation
path for models that benefit from it.

## Model-weight quantization (TQ3_1S / TQ4_1S)

```bash
llama-quantize model-f16.gguf model-tq4.gguf TQ4_1S
```

`TQ3_1S` and `TQ4_1S` are first-class weight types with CPU, CUDA/HIP (warp
cooperative mmvq), Metal, and Vulkan kernels. ggml-sycl has no TQ kernels, and
its generic dequantize path leaves TQ weights in the WHT-rotated domain, so TQ
models are unsupported on SYCL. For TQ MoE weights, the small-batch
`MUL_MAT_ID` path routes experts on device and stays CUDA-graph capturable;
only the large-batch fallback dequantizes to f16 and synchronizes the stream.
