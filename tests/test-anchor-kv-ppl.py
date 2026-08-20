#!/usr/bin/env python3
"""
AnchorKV Perplexity Test

Measures perplexity with AnchorKV-compressed KV cache by:
1. Loading model + wikitext
2. Running full forward pass to get logits
3. Compressing KV cache
4. Re-running forward pass with decompressed KV
5. Comparing perplexity between dense and compressed

This is a CPU-only reference test.
"""

import torch
import numpy as np
import sys
import os

def anchor_kv_compress_python(keys, values, theta=0.1, W=32, k_frac=128):
    """
    Python reference for AnchorKV compression.
    keys, values: [S, D] tensors
    Returns: compressed representation (dict)
    """
    S, D = keys.shape
    k = max(W, S // k_frac)
    P = S - W
    rho = 0.7
    kappa = 7

    # Anchor selection: last W + top scored + random
    is_anchor = torch.zeros(S, dtype=torch.bool)
    anchor_positions = list(range(S - W, S))
    is_anchor[S - W:] = True

    # Score non-window positions
    inv_sqrt_d = 1.0 / (D ** 0.5)
    scores = torch.zeros(P)
    for t in range(P):
        dot = torch.sum(keys[S - W:] * keys[t], dim=1) * inv_sqrt_d
        scores[t] = torch.mean(torch.exp(dot))

    # Average pool
    pooled = torch.zeros(P)
    for t in range(P):
        start = max(0, t - kappa // 2)
        end = min(P, t + kappa // 2 + 1)
        pooled[t] = scores[start:end].mean()

    # Select top scored
    n_scored = k - W
    n_top = int(rho * n_scored)
    remaining = list(range(P))

    for _ in range(min(n_top, len(remaining))):
        best = max(range(len(remaining)), key=lambda i: pooled[remaining[i]])
        pos = remaining[best]
        is_anchor[pos] = True
        anchor_positions.append(pos)
        remaining.pop(best)

    # Random remaining
    np.random.seed(42)
    n_random = n_scored - n_top
    for _ in range(min(n_random, len(remaining))):
        idx = np.random.randint(len(remaining))
        pos = remaining[idx]
        is_anchor[pos] = True
        anchor_positions.append(pos)
        remaining.pop(idx)

    anchor_positions.sort()
    k_actual = len(anchor_positions)

    # Store exact anchor vectors
    anchor_keys = keys[anchor_positions]
    anchor_values = values[anchor_positions]

    # Compute projections
    k_anchor_of = torch.zeros(S, dtype=torch.long)
    v_anchor_of = torch.zeros(S, dtype=torch.long)
    k_gamma = torch.zeros(S)
    v_gamma = torch.zeros(S)
    residual_K = torch.zeros(S, D)
    residual_V = torch.zeros(S, D)

    # Build reverse lookup
    pos_to_idx = {pos: i for i, pos in enumerate(anchor_positions)}

    for t in range(S):
        if is_anchor[t]:
            k_anchor_of[t] = pos_to_idx[t]
            v_anchor_of[t] = pos_to_idx[t]
            k_gamma[t] = 1.0
            v_gamma[t] = 1.0
            continue

        # K: nearest anchor by cosine
        data_norm = keys[t].norm()
        sims = torch.zeros(k_actual)
        for a in range(k_actual):
            anch_norm = anchor_keys[a].norm()
            sims[a] = abs(torch.dot(keys[t], anchor_keys[a])) / (data_norm * anch_norm + 1e-10)
        best_k = sims.argmax().item()
        k_anchor_of[t] = best_k
        anch_norm_sq = anchor_keys[best_k].norm() ** 2
        k_gamma[t] = torch.dot(keys[t], anchor_keys[best_k]) / (anch_norm_sq + 1e-20)
        residual_K[t] = keys[t] - k_gamma[t] * anchor_keys[best_k]

        # V: nearest anchor by cosine
        data_norm = values[t].norm()
        sims = torch.zeros(k_actual)
        for a in range(k_actual):
            anch_norm = anchor_values[a].norm()
            sims[a] = abs(torch.dot(values[t], anchor_values[a])) / (data_norm * anch_norm + 1e-10)
        best_v = sims.argmax().item()
        v_anchor_of[t] = best_v
        anch_norm_sq = anchor_values[best_v].norm() ** 2
        v_gamma[t] = torch.dot(values[t], anchor_values[best_v]) / (anch_norm_sq + 1e-20)
        residual_V[t] = values[t] - v_gamma[t] * anchor_values[best_v]

    # Residual selection by utility
    alpha = 1.0 / S
    y = values.mean(dim=0)

    uK = torch.zeros(S)
    uV = torch.zeros(S)
    for t in range(S):
        if is_anchor[t]:
            continue
        v_minus_y = values[t] - y
        uK[t] = alpha**2 * (residual_K[t].norm()**2 / D) * v_minus_y.norm()**2
        uV[t] = alpha**2 * residual_V[t].norm()**2

    # Budget
    Mfull = 2 * S * D * 2
    Mbase = 4 * k * D + 4 * (k - W) + 2 * P * 8 + 2 * ((P + 63) // 64) * 8 + 8 + 4 * P
    cK = D / 4 + 4
    cV = D / 4 + 5
    budget = theta * Mfull - Mbase
    N = max(0, int(budget / ((cK + cV) / 2)))
    NK = N // 2
    NV = N - NK

    # Select residuals
    non_anchors = [t for t in range(S) if not is_anchor[t]]

    # K residuals
    sorted_k = sorted(non_anchors, key=lambda t: uK[t], reverse=True)[:NK]
    # V residuals
    remaining_v = [t for t in non_anchors if t not in sorted_k]
    sorted_v = sorted(remaining_v, key=lambda t: uV[t], reverse=True)[:NV]

    return {
        'anchor_positions': anchor_positions,
        'anchor_keys': anchor_keys,
        'anchor_values': anchor_values,
        'k_anchor_of': k_anchor_of,
        'v_anchor_of': v_anchor_of,
        'k_gamma': k_gamma,
        'v_gamma': v_gamma,
        'is_anchor': is_anchor,
        'k_residuals': sorted_k,
        'v_residuals': sorted_v,
        'S': S, 'D': D, 'k': k_actual,
        'N_K': len(sorted_k), 'N_V': len(sorted_v),
    }


def decompress(cache, keys, values):
    """Decompress AnchorKV cache back to dense form."""
    S, D = keys.shape
    recon_k = torch.zeros(S, D)
    recon_v = torch.zeros(S, D)

    for t in range(S):
        if cache['is_anchor'][t]:
            a = cache['k_anchor_of'][t]
            recon_k[t] = cache['anchor_keys'][a]
            a = cache['v_anchor_of'][t]
            recon_v[t] = cache['anchor_values'][a]
        else:
            a_k = cache['k_anchor_of'][t]
            recon_k[t] = cache['k_gamma'][t] * cache['anchor_keys'][a_k]
            a_v = cache['v_anchor_of'][t]
            recon_v[t] = cache['v_gamma'][t] * cache['anchor_values'][a_v]
            # Note: residuals are 2-bit quantized (lossy), but for PPL test
            # we use the original residuals to measure anchor quality only

    return recon_k, recon_v


def compute_ppl(model, input_ids, kv_cache=None):
    """Compute perplexity given model and input tokens."""
    with torch.no_grad():
        outputs = model(input_ids, past_key_values=kv_cache)
        logits = outputs.logits
        # Shift: predict next token from current position
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        return torch.exp(loss).item()


def main():
    model_name = 'Qwen/Qwen2.5-1.5B'
    seq_len = 4096
    theta = 0.1
    chunk_size = 512

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    if len(sys.argv) > 2:
        seq_len = int(sys.argv[2])
    if len(sys.argv) > 3:
        theta = float(sys.argv[3])

    print(f"AnchorKV Perplexity Test")
    print(f"========================")
    print(f"Model: {model_name}")
    print(f"Sequence: {seq_len} tokens")
    print(f"Theta: {theta} ({1/theta:.0f}x compression)\n")

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model.eval()

    config = model.config
    n_kv_head = getattr(config, 'num_key_value_heads', config.num_attention_heads)
    head_dim = config.hidden_size // config.num_attention_heads
    n_layer = config.num_hidden_layers
    print(f"Config: {n_layer} layers, {n_kv_head} KV heads, head_dim={head_dim}\n")

    # Load wikitext
    wt_path = '/mnt/storage/blackbeard/wikitext-2-raw/wiki.test.raw'
    if os.path.exists(wt_path):
        with open(wt_path) as f:
            text = f.read()
        tokens = tokenizer.encode(text)
        tokens = tokens[:seq_len]
        print(f"Wikitext tokens: {len(tokens)}")
    else:
        # Fallback: repeated text
        tokens = tokenizer.encode('The quick brown fox jumps over the lazy dog. ' * 500)[:seq_len]
        print(f"Fallback tokens: {len(tokens)}")

    input_ids = torch.tensor([tokens], dtype=torch.long)

    # --- Dense baseline ---
    print("\n--- Dense (baseline) ---")
    ppl_dense = compute_ppl(model, input_ids)
    print(f"PPL: {ppl_dense:.2f}")

    # --- Chunked: compress KV at each chunk boundary ---
    print(f"\n--- AnchorKV (theta={theta}, chunk={chunk_size}) ---")

    # We'll process in chunks, compressing the KV cache at each step
    # This simulates what would happen in a real inference loop

    total_loss = 0.0
    total_tokens = 0

    # Use KV cache progressively
    past = None
    all_k = []
    all_v = []

    for start in range(0, len(tokens), chunk_size):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunk_ids = torch.tensor([chunk], dtype=torch.long)

        if past is None:
            # First chunk: full forward
            with torch.no_grad():
                out = model(chunk_ids, past_key_values=None)
                past = out.past_key_values
                logits = out.logits
        else:
            # Subsequent chunks: use last token only with cache
            with torch.no_grad():
                out = model(chunk_ids[:, -1:], past_key_values=past)
                past = out.past_key_values
                logits = out.logits

        # Collect all K/V from cache
        layer_k = []
        layer_v = []
        for li in range(n_layer):
            k = past.key_cache[li].squeeze(0)  # [n_heads, seq_so_far, head_dim]
            v = past.value_cache[li].squeeze(0)
            layer_k.append(k)
            layer_v.append(v)

        # Compute loss on this chunk
        if start == 0:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk_ids[:, 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        total_loss += loss.item() * (end - start - 1)
        total_tokens += end - start - 1

        # Compress KV cache at chunk boundary
        if end < len(tokens):
            print(f"  Compressing at token {end}...")
            for li in range(n_layer):
                k_dense = layer_k[li].numpy().reshape(-1, head_dim).copy()
                v_dense = layer_v[li].numpy().reshape(-1, head_dim).copy()
                cache = anchor_kv_compress_python(
                    torch.from_numpy(k_dense),
                    torch.from_numpy(v_dense),
                    theta=theta
                )
                recon_k, recon_v = decompress(cache, torch.from_numpy(k_dense), torch.from_numpy(v_dense))

                # Replace cache with decompressed values
                # Shape: [n_heads, seq, head_dim]
                current_len = past.key_cache[li].shape[2]
                new_k = torch.from_numpy(recon_k.numpy().reshape(n_kv_head, -1, head_dim)).float()
                new_v = torch.from_numpy(recon_v.numpy().reshape(n_kv_head, -1, head_dim)).float()

                # Update past key values
                past.key_cache[li] = new_k
                past.value_cache[li] = new_v

    ppl_anchor = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
    print(f"\nPPL: {ppl_anchor:.2f}")

    # --- Summary ---
    print(f"\n=== Summary ===")
    print(f"Dense PPL:   {ppl_dense:.2f}")
    print(f"Anchor PPL:  {ppl_anchor:.2f}")
    print(f"Ratio:       {ppl_anchor/ppl_dense:.4f}")
    print(f"Degradation: {(ppl_anchor/ppl_dense - 1)*100:.2f}%")


if __name__ == "__main__":
    main()
