"""Llama-3.2-1B verification: kernel correctness + foveated quality.

Generates Llama results for Section 4 of the paper.

Usage:
  uv run python paper/scripts/llama_verification.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import time
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from foveated_kv.mlx_foveated import MLXTierConfig, standard_attention_mlx
from foveated_kv.mlx_generate import prefill_and_compress, generate_fused


def cosine(a, b):
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    c = mx.sum(a_f * b_f) / (mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f)) + 1e-8)
    mx.eval(c)
    return c.item()


def main():
    model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    model, tokenizer = load(model_name)
    print(f"Model: {model_name}")

    # 1. Kernel correctness (fused vs dequant-reference)
    filler = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = mx.array(tokenizer.encode(filler)[:2048]).reshape(1, -1)
    print(f"Context: {tokens.shape[1]} tokens")

    # Standard cache for reference
    std_cache = make_prompt_cache(model)
    logits = model(tokens, cache=std_cache)
    mx.eval(logits)

    k0, v0 = std_cache[0].state
    head_dim = k0.shape[-1]
    n_kv_heads = k0.shape[1]
    print(f"GQA: H_kv={n_kv_heads}, D={head_dim}")

    query = mx.random.normal((1, n_kv_heads, 1, head_dim)).astype(mx.float16)
    mx.eval(query)
    exact_out = standard_attention_mlx(query, k0, v0)
    mx.eval(exact_out)

    # Foveated cache
    cfg = MLXTierConfig(near_pct=0.10)
    fov_cache, _, _ = prefill_and_compress(model, tokens, cfg)
    fov_out = fov_cache.attend(0, query)
    mx.eval(fov_out)

    fov_cos = cosine(fov_out, exact_out)
    print(f"\nFoveated vs exact fp16 cosine: {fov_cos:.6f}")

    # 2. Short-context token match
    short_prompt = "Explain the theory of relativity in simple terms:"
    short_tokens = mx.array(tokenizer.encode(short_prompt)).reshape(1, -1)

    # Standard generation
    std_cache2 = make_prompt_cache(model)
    std_logits = model(short_tokens, cache=std_cache2)
    mx.eval(std_logits)
    std_generated = []
    next_logits = std_logits[:, -1, :]
    for _ in range(30):
        tok = mx.argmax(next_logits, axis=-1)
        tid = tok.item()
        if tid == tokenizer.eos_token_id:
            break
        std_generated.append(tid)
        next_logits = model(tok.reshape(1, 1), cache=std_cache2)[:, -1, :]
        mx.eval(next_logits)

    # Foveated generation
    fov_text, fov_stats = generate_fused(
        model, tokenizer, short_prompt, max_tokens=30, cfg=cfg,
    )
    fov_generated = tokenizer.encode(fov_text)

    match = sum(1 for a, b in zip(std_generated, fov_generated) if a == b)
    total = min(len(std_generated), len(fov_generated))
    print(f"Short-context token match: {match}/{total} ({100*match/max(total,1):.0f}%)")

    # 3. Throughput
    prompt_long = "Write a detailed essay about the history of computing:\n"
    t0 = time.time()
    text, stats = generate_fused(
        model, tokenizer, prompt_long, max_tokens=100, cfg=cfg,
    )
    elapsed = time.time() - t0
    n_tokens = len(tokenizer.encode(text))
    tps = n_tokens / elapsed
    print(f"Throughput: {tps:.1f} tok/s ({n_tokens} tokens in {elapsed:.1f}s)")


if __name__ == "__main__":
    main()
