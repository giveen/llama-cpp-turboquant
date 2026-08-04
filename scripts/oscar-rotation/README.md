# OSCAR2 rotation matrices + GGUF baking

OSCAR2 quantizes KV cache to INT2 (8x smaller than f16) while preserving quality.
It uses orthogonal rotation matrices to spread information across dimensions before
quantization, preventing the 2-bit quantization noise from scrambling attention.

## Do I need to run the conversion script?

**It depends on the model architecture — most models do NOT need it.**

### Models that DON'T need conversion (rotation built into KV cache)

These models generate Hadamard rotation matrices at runtime — just use
`-ctk oscar2 -ctv oscar2` on the original GGUF:

| Architecture | KV cache backend | Notes |
|---|---|---|
| deepseek4 (DSv4 Flash) | dsv4 | MLA with built-in Hadamard via `self_k_rot` |
| dflash | dsv4 | Same as deepseek4 |
| glm-dsa | dsa | Lightning indexer with built-in rotation |
| minimax-m3 | standard | Hadamard via `self_k_rot` in graph |
| deepseek32 | dsa | Same as glm-dsa |
| Most standard-attention models | standard | Runtime Hadamard from KV cache |

**Test if your model works:** `llama-cli -m model.gguf -ctk oscar2 -ctv oscar2 -fa on`
If it loads and produces coherent output, no conversion needed.

### Models that NEED conversion (rotation via GGUF tensors)

These models load `blk.{i}.attn_k_rot.weight` / `blk.{i}.attn_v_rot.weight` from the GGUF.
The conversion script bakes calibrated or Hadamard rotation matrices into a copy of the GGUF:

| Architecture | Status | PPL (wikitext-2, ctx=128) |
|---|---|---|
| gemma4 (Gemma 4 12B) | ✅ Working | f16=546, oscar2+HP=**157** |
| gemma4 (Gemma 4 26B) | ✅ Working | Not yet benchmarked |
| qwen3 (Qwen3 8B) | ✅ Working | f16 baseline verified |
| llama (Llama 3/4) | ❌ Not yet | Needs `attn_k_rot` in model code |
| Other architectures | ❓ Unknown | Report on GitHub if yours isn't listed |

## Conversion: one-command pipeline

For models that need conversion, use the data-free Hadamard method:

```bash
python3 scripts/oscar-rotation/generate_and_bake_rot.py \
    --base /path/to/model.gguf \
    --out  /path/to/model-rot-kv.gguf
```

This generates a normalized Hadamard matrix per unique head dimension, then bakes
`attn_k_rot` and `attn_v_rot` tensors into a new GGUF. The base weights are copied
unchanged — only rotation tensors (~50 MB for a 12B model) are added.

For calibrated rotations (best quality), you need QKV activation dumps from a
calibration dataset:

```bash
# Step 1: dump covariances from calibration text
./build/bin/llama-oscar-calib -m model.gguf -f calibration.txt -o covariances/

# Step 2: compute eigendecomposition + R·H·P composition
python3 scripts/oscar-rotation/calibrate_rotation.py \
    --cov-dir covariances/ \
    --out-dir rotations/ \
    --num-layers N --head-dim D

# Step 3: bake into GGUF
python3 scripts/oscar-rotation/export_rot_kv_gguf.py \
    --base model.gguf \
    --rot-dir rotations/ \
    --out model-rot-kv.gguf
```

## Usage

```bash
# Interactive chat
./build/bin/llama-cli -m model-rot-kv.gguf -ctk oscar2 -ctv oscar2 -fa on -ngl 99

# Perplexity evaluation
./build/bin/llama-perplexity -m model-rot-kv.gguf -ctk oscar2 -ctv oscar2 -fa on -ngl 99 \
    -f wikitext-2-raw/wiki.test.raw -c 128
```

For Gemma 4, HP (high-precision) sink+recent buffer auto-enables with 96 recent cells.
Override with `LLAMA_KV_HP_RECENT=N` (128 = full context, matches f16 quality).

## Files

- `generate_and_bake_rot.py` — One-command pipeline (Hadamard or calibrated)
- `generate_hadamard_rot.py` — Data-free Hadamard rotation generator
- `calibrate_rotation.py` — Calibrated rotation from covariance dumps
- `export_rot_kv_gguf.py` — Bakes rotation .pt files into GGUF
- `kld_local_logit.py` — KLD measurement utility

## If your model isn't supported

Check if it works without conversion first: `llama-cli -m model.gguf -ctk oscar2 -ctv oscar2`.

If it crashes, garbles output, or produces incoherent text — the model likely needs
rotation support added to its C++ code. **Open an issue** on the GitHub repo with:
- Model architecture and size
- The error or symptoms
- Whether f16 KV works on the same model (`-ctk f16 -ctv f16`)

Adding rotation to a new architecture is typically ~30 lines of C++:
1. Add `attn_k_rot` / `attn_v_rot` tensor loading in `load_arch_tensors`
2. Apply `llama_mul_mat_hadamard(Q, attn_k_rot)` and same for K in the graph
3. Optionally apply V rotation with undo after attention
