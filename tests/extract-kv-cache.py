#!/usr/bin/env python3
"""
Extract real KV cache tensors from a HuggingFace model.

Loads a model on CPU, runs a prompt, hooks into attention layers
to capture K and V tensors, and saves them as binary files for
the AnchorKV CPU reference to consume.

Usage:
    python3 tests/extract-kv-cache.py --model Qwen/Qwen3-8B --prompt "..." --seq-len 4096 --layer 0 --head 0
    python3 tests/extract-kv-cache.py --model-path /path/to/model.gguf ...

Output:
    /tmp/anchor-kv-k.bin  -- [seq_len * head_dim] float32 key vectors
    /tmp/anchor-kv-v.bin  -- [seq_len * head_dim] float32 value vectors
"""

import argparse
import struct
import sys
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Extract KV cache from a model")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B",
                        help="HuggingFace model name or path")
    parser.add_argument("--prompt", type=str,
                        default=None,
                        help="Prompt text (or --prompt-file for a text file)")
    parser.add_argument("--prompt-file", type=str,
                        default=None,
                        help="Text file to use as prompt (e.g. wikitext)")
    parser.add_argument("--seq-len", type=int, default=4096,
                        help="Target sequence length (truncate/pad prompt)")
    parser.add_argument("--layer", type=int, default=0,
                        help="Layer index to extract")
    parser.add_argument("--head", type=int, default=0,
                        help="KV head index to extract")
    parser.add_argument("--output-dir", type=str, default="/tmp",
                        help="Output directory for binary files")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    print(f"  (CPU-only, this may take a while for large models)")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()

    # Get model config
    config = model.config
    n_layer = config.num_hidden_layers
    n_head = config.num_attention_heads
    n_kv_head = getattr(config, 'num_key_value_heads', n_head)
    head_dim = config.hidden_size // n_head

    print(f"  Layers: {n_layer}, Q heads: {n_head}, KV heads: {n_kv_head}, head_dim: {head_dim}")

    if args.layer >= n_layer:
        print(f"Error: layer {args.layer} >= n_layer {n_layer}")
        sys.exit(1)
    if args.head >= n_kv_head:
        print(f"Error: head {args.head} >= n_kv_head {n_kv_head}")
        sys.exit(1)

    # Load prompt
    if args.prompt_file:
        with open(args.prompt_file, 'r') as f:
            prompt = f.read()
        print(f"  Loaded prompt from {args.prompt_file} ({len(prompt)} chars)")
    elif args.prompt:
        prompt = args.prompt
    else:
        prompt = "The quick brown fox jumps over the lazy dog. " * 100

    # Tokenize prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt")
    if tokens.shape[1] > args.seq_len:
        tokens = tokens[:, :args.seq_len]
    elif tokens.shape[1] < args.seq_len:
        # Pad with pad token
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        pad = torch.full((1, args.seq_len - tokens.shape[1]), pad_id, dtype=tokens.dtype)
        tokens = torch.cat([tokens, pad], dim=1)

    seq_len = tokens.shape[1]
    print(f"  Sequence length: {seq_len}")

    # Hook to capture K and V
    captured = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is (attn_output, attn_weights, past_key_value)
            # We need the K and V before attention
            # Different model architectures store these differently
            pass
        return hook_fn

    # For Qwen3 and similar models, we need to hook into the self-attention
    # to capture the key and value projections.
    # The attention module typically has shape [batch, n_kv_head, seq, head_dim]

    k_cache = [None]
    v_cache = [None]

    def capture_kv(module, args_in, output):
        """Hook on the attention forward to capture K and V."""
        # For Qwen3: the attention module receives hidden_states and returns
        # (attn_output, attn_weights, past_key_value)
        # We need to get K and V from the computation
        pass

    # Alternative: hook into the key and value projection layers directly
    target_layer = model.model.layers[args.layer]
    self_attn = target_layer.self_attn

    # Qwen3 attention has k_proj and v_proj
    if hasattr(self_attn, 'k_proj'):
        k_proj = self_attn.k_proj
        v_proj = self_attn.v_proj

        def k_hook(module, inp, out):
            # out shape: [batch, seq_len, n_kv_head * head_dim]
            # Reshape to [n_kv_head, seq_len, head_dim]
            k_cache[0] = out.detach().view(1, seq_len, n_kv_head, head_dim)

        def v_hook(module, inp, out):
            v_cache[0] = out.detach().view(1, seq_len, n_kv_head, head_dim)

        k_proj.register_forward_hook(k_hook)
        v_proj.register_forward_hook(v_hook)
    else:
        print(f"Error: cannot find k_proj in attention module")
        print(f"  Attention module type: {type(self_attn)}")
        print(f"  Attributes: {[a for a in dir(self_attn) if 'proj' in a.lower() or 'key' in a.lower()]}")
        sys.exit(1)

    print(f"Running forward pass...")
    with torch.no_grad():
        output = model(tokens)

    if k_cache[0] is None or v_cache[0] is None:
        print("Error: KV cache not captured")
        sys.exit(1)

    # Extract the specific head
    # k_cache shape: [1, seq_len, n_kv_head, head_dim]
    k_head = k_cache[0][0, :, args.head, :].numpy()  # [seq_len, head_dim]
    v_head = v_cache[0][0, :, args.head, :].numpy()  # [seq_len, head_dim]

    print(f"  K shape: {k_head.shape}, V shape: {v_head.shape}")
    print(f"  K stats: mean={k_head.mean():.4f}, std={k_head.std():.4f}, "
          f"norm_mean={np.linalg.norm(k_head, axis=1).mean():.4f}")
    print(f"  V stats: mean={v_head.mean():.4f}, std={v_head.std():.4f}, "
          f"norm_mean={np.linalg.norm(v_head, axis=1).mean():.4f}")

    # Save as binary files
    k_path = f"{args.output_dir}/anchor-kv-k.bin"
    v_path = f"{args.output_dir}/anchor-kv-v.bin"
    k_head.tofile(k_path)
    v_head.tofile(v_path)

    print(f"\nSaved:")
    print(f"  K: {k_path} ({k_head.nbytes} bytes, {k_head.shape})")
    print(f"  V: {v_path} ({v_head.nbytes} bytes, {v_head.shape})")
    print(f"\nRun the C test with:")
    print(f"  ./test-anchor-kv --from-file --seq-len {seq_len} --head-dim {head_dim}")


if __name__ == "__main__":
    main()
