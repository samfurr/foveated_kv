"""Tier ratio ablation: cosine fidelity vs near_pct at 2K context.

Generates Table 5 in the paper.

Usage:
  uv run python paper/scripts/tier_ratio_ablation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mlx.core as mx
from mlx_lm import load
from foveated_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig, standard_attention_mlx
from foveated_kv.mlx_generate import prefill_and_compress


def cosine(a, b):
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    c = mx.sum(a_f * b_f) / (mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f)) + 1e-8)
    mx.eval(c)
    return c.item()


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = mx.array(tokenizer.encode(filler)[:2048]).reshape(1, -1)
    print(f"Context: {tokens.shape[1]} tokens")

    from mlx_lm.models.cache import make_prompt_cache
    std_cache = make_prompt_cache(model)
    logits = model(tokens, cache=std_cache)
    mx.eval(logits)

    k0, v0 = std_cache[0].state
    query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
    mx.eval(query)
    exact_out = standard_attention_mlx(query, k0, v0)
    mx.eval(exact_out)

    print(f"{'Near %':>8} {'Near tok':>9} {'Far tok':>8} {'Cosine':>10} {'Error':>10}")
    print("-" * 50)

    for pct in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00]:
        cfg = MLXTierConfig(near_pct=pct)
        fov_cache, _, _ = prefill_and_compress(model, tokens, cfg)
        layer = fov_cache.layers[0]
        near_valid = int(mx.max(layer.near_valid).item())
        far_count = layer.far_k.shape[2]

        fov_out = fov_cache.attend(0, query)
        mx.eval(fov_out)
        cos = cosine(fov_out, exact_out)

        print(f"{pct:>7.0%} {near_valid:>9} {far_count:>8} {cos:>10.6f} {1 - cos:>10.6f}")


if __name__ == "__main__":
    main()
