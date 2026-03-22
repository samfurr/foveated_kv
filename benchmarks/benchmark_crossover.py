"""
Crossover benchmark: find where fused kernel beats standard SDPA end-to-end.

The fused kernel is always faster in isolation (1.1-2.1x at 512-32K).
But the Python interceptor adds ~4ms/layer of dispatch overhead. This
benchmark measures the TOTAL decode step cost (kernel + Python overhead)
and finds the context length where fused wins despite the Python tax.

Two measurements:
  1. Kernel-only: attend_fused() vs standard_attention_mlx() — one layer
  2. End-to-end: full model forward pass with interceptor vs standard cache

Usage:
  uv run python benchmarks/benchmark_crossover.py
  uv run python benchmarks/benchmark_crossover.py --contexts 512 1024 2048 4096 8192
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import (
    MLXFoveatedKVCache,
    MLXTierConfig,
    standard_attention_mlx,
)


def bench(fn, warmup=5, iters=50):
    for _ in range(warmup):
        out = fn()
        mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000


def run_kernel_crossover(contexts, H_kv, H_q, D, iters=50):
    """Kernel-only: one layer, no model, no interceptor."""
    cfg = MLXTierConfig()
    B = 1

    print(f"\n{'='*65}")
    print(f"KERNEL-ONLY (1 layer, H_kv={H_kv}, H_q={H_q}, D={D})")
    print(f"{'='*65}")
    print(f"{'Context':>8} {'fp16':>10} {'Fused':>10} {'Ratio':>8} {'Winner':>8}")
    print(f"{'-'*50}")

    results = []
    for S in contexts:
        keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
        query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
        mx.eval(keys, values, query)

        # Standard fp16
        std_ms = bench(lambda: standard_attention_mlx(query, keys, values), iters=iters)

        # Fused kernel
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()
        layer = cache.layers[0]
        fused_ms = bench(lambda: layer.attend_fused(query), iters=iters)

        ratio = fused_ms / std_ms
        winner = "FUSED" if ratio < 1.0 else "fp16"
        print(f"{S:>8} {std_ms:>9.1f}ms {fused_ms:>9.1f}ms {ratio:>7.2f}x {winner:>8}")
        results.append({
            "context": S, "std_ms": round(std_ms, 2),
            "fused_ms": round(fused_ms, 2), "ratio": round(ratio, 3),
        })
        del keys, values, query, cache, layer

    return results


def run_model_crossover(contexts, iters=10):
    """End-to-end: full model with interceptor overhead."""
    from mlx_lm import load
    from foveated_kv.mlx_generate import (
        FusedCacheWrapper, install_fused_sdpa, uninstall_fused_sdpa,
        reset_fused_layer_counter, prefill_and_compress, _fused_state,
    )
    from mlx_lm.models.cache import make_prompt_cache

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    cfg = MLXTierConfig()
    tok = mx.array([[1]])
    base_text = "The quick brown fox jumps over the lazy dog and runs through the forest. " * 500

    print(f"\n{'='*65}")
    print(f"END-TO-END (Qwen2.5-0.5B, 24 layers, with interceptor)")
    print(f"{'='*65}")
    print(f"{'Context':>8} {'Standard':>10} {'Fused':>10} {'Ratio':>8} {'Winner':>8}")
    print(f"{'-'*50}")

    results = []
    for S in contexts:
        tokens = mx.array(tokenizer.encode(base_text)[:S]).reshape(1, -1)
        actual = tokens.shape[1]

        # Standard
        cache = make_prompt_cache(model)
        logits = model(tokens, cache=cache); mx.eval(logits)
        for _ in range(3): out = model(tok, cache=cache); mx.eval(out)
        t0 = time.perf_counter()
        for _ in range(iters): out = model(tok, cache=cache); mx.eval(out)
        std_ms = (time.perf_counter() - t0) / iters * 1000
        del cache

        # Fused
        fov, pl, _ = prefill_and_compress(model, tokens, cfg)
        wr = [FusedCacheWrapper(l, i) if l else None for i, l in enumerate(fov.layers)]
        install_fused_sdpa(fov); _fused_state._fused_wrappers = wr
        for _ in range(3): reset_fused_layer_counter(); out = model(tok, cache=wr); mx.eval(out)
        t0 = time.perf_counter()
        for _ in range(iters): reset_fused_layer_counter(); out = model(tok, cache=wr); mx.eval(out)
        fused_ms = (time.perf_counter() - t0) / iters * 1000
        uninstall_fused_sdpa()

        ratio = fused_ms / std_ms
        winner = "FUSED" if ratio < 1.0 else "std"
        print(f"{actual:>8} {std_ms:>9.1f}ms {fused_ms:>9.1f}ms {ratio:>7.2f}x {winner:>8}")
        results.append({
            "context": actual, "std_ms": round(std_ms, 2),
            "fused_ms": round(fused_ms, 2), "ratio": round(ratio, 3),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Crossover Benchmark")
    parser.add_argument("--contexts", type=int, nargs="+", default=[256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--kernel-only", action="store_true", help="Skip model benchmark")
    parser.add_argument("--model-only", action="store_true", help="Skip kernel benchmark")
    parser.add_argument("--h-kv", type=int, default=4)
    parser.add_argument("--h-q", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", default="results/paper/crossover.json")
    args = parser.parse_args()

    all_results = {}

    if not args.model_only:
        all_results["kernel"] = run_kernel_crossover(
            args.contexts, args.h_kv, args.h_q, args.head_dim)

    if not args.kernel_only:
        # Use smaller context set for model (slower)
        model_contexts = [c for c in args.contexts if c <= 4096]
        all_results["model"] = run_model_crossover(model_contexts)

    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")
    if "kernel" in all_results:
        k_wins = [r for r in all_results["kernel"] if r["ratio"] < 1.0]
        print(f"Kernel-only: fused wins at {len(k_wins)}/{len(all_results['kernel'])} contexts")
        if k_wins:
            best = min(k_wins, key=lambda r: r["ratio"])
            print(f"  Best: {best['ratio']:.2f}x at {best['context']} tokens")
    if "model" in all_results:
        m_wins = [r for r in all_results["model"] if r["ratio"] < 1.0]
        print(f"End-to-end:  fused wins at {len(m_wins)}/{len(all_results['model'])} contexts")
        if m_wins:
            best = min(m_wins, key=lambda r: r["ratio"])
            print(f"  Best: {best['ratio']:.2f}x at {best['context']} tokens")
        else:
            print(f"  Python interceptor overhead dominates at 0.5B model scale")
            print(f"  C++ extension needed for end-to-end wins")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
