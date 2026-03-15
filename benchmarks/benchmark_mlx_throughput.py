"""
Throughput + memory benchmark for FoveatedKV paper.

Measures decode tok/s, KV memory, compression ratio, and prefill time
with a real model on Apple Silicon.

Usage:
  uv run python benchmarks/benchmark_mlx_throughput.py
  uv run python benchmarks/benchmark_mlx_throughput.py --contexts 2048 4096
"""

import argparse
import gc
import json
import os
import sys
import tempfile
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipmap_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig
from mipmap_kv.mlx_generate import (
    generate_fused, generate_foveated, _generate_short, prefill_and_compress,
)
from mipmap_kv.disk_archive import offload_cache_to_disk


def measure_method(model, tokenizer, prompt, method, gen_tokens, cfg=None):
    """Run one method, return timing and memory stats."""
    t_start = time.perf_counter()

    if method == "standard":
        _generate_short(model, tokenizer, prompt, max_tokens=gen_tokens)
        elapsed = time.perf_counter() - t_start
        return {
            "tok_per_s": gen_tokens / elapsed,
            "elapsed_s": round(elapsed, 2),
            "kv_memory_mb": "N/A (full fp16)",
        }

    elif method == "foveated_unfused":
        _, stats = generate_foveated(model, tokenizer, prompt, max_tokens=gen_tokens, cfg=cfg)
        elapsed = time.perf_counter() - t_start
        return {
            "tok_per_s": gen_tokens / elapsed,
            "elapsed_s": round(elapsed, 2),
            "kv_memory_mb": round(stats.get("tiers", {}).get("foveal", 0) * 0.001, 1),
        }

    elif method == "fused":
        _, stats = generate_fused(model, tokenizer, prompt, max_tokens=gen_tokens,
                                   cfg=cfg, enable_promotion=False)
        elapsed = time.perf_counter() - t_start
        return {
            "tok_per_s": stats.get("tokens_per_second", gen_tokens / elapsed),
            "elapsed_s": round(elapsed, 2),
        }

    elif method == "fused_disk":
        _, stats = generate_fused(model, tokenizer, prompt, max_tokens=gen_tokens,
                                   cfg=cfg, enable_promotion=True)
        elapsed = time.perf_counter() - t_start
        return {
            "tok_per_s": stats.get("tokens_per_second", gen_tokens / elapsed),
            "elapsed_s": round(elapsed, 2),
            "mem_saved_mb": round(stats.get("mem_saved_mb", 0), 1),
        }


def measure_compression(model, tokenizer, prompt, cfg):
    """Measure compression ratio and memory."""
    tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)

    t0 = time.perf_counter()
    fov_cache, _, _ = prefill_and_compress(model, tokens, cfg)
    compress_time = time.perf_counter() - t0

    mem = fov_cache.memory_bytes()
    n_tokens = tokens.shape[1]
    H_kv = fov_cache.layers[0].foveal_k.shape[1]
    D = fov_cache.layers[0].foveal_k.shape[-1]
    n_layers = len(fov_cache.layers)
    fp16_kv_bytes = n_layers * 2 * H_kv * n_tokens * D * 2

    return {
        "context_tokens": n_tokens,
        "n_layers": n_layers,
        "fp16_kv_mb": round(fp16_kv_bytes / 1e6, 1),
        "foveated_kv_mb": round(mem["total_quantized"] / 1e6, 1),
        "compression_ratio": round(fp16_kv_bytes / max(mem["total_quantized"], 1), 2),
        "archive_mb": round(mem["archive"] / 1e6, 1),
        "compress_time_s": round(compress_time, 2),
        "foveal_tokens": fov_cache.layers[0].foveal_k.shape[2],
        "periph_tokens": fov_cache.layers[0].periph_k.shape[2],
        "far_tokens": fov_cache.layers[0].far_k.shape[2],
    }


def main():
    parser = argparse.ArgumentParser(description="Throughput + Memory Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--contexts", nargs="+", type=int, default=[2048, 4096, 8192])
    parser.add_argument("--gen-tokens", type=int, default=100)
    parser.add_argument("--output", default="results/paper/throughput.json")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    cfg = MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)
    filler = "This document discusses various topics in science and technology. " * 5

    all_results = {"model": args.model, "gen_tokens": args.gen_tokens, "contexts": {}}

    for ctx in args.contexts:
        print(f"\n{'='*60}")
        print(f"Context: {ctx} tokens, generating {args.gen_tokens}")
        print(f"{'='*60}")

        toks = tokenizer.encode(filler * 200)[:ctx]
        prompt = tokenizer.decode(toks)

        # Compression stats
        comp = measure_compression(model, tokenizer, prompt, cfg)
        print(f"  Compression: {comp['fp16_kv_mb']:.1f}MB → {comp['foveated_kv_mb']:.1f}MB ({comp['compression_ratio']:.2f}x)")
        print(f"  Tiers: fov={comp['foveal_tokens']}, per={comp['periph_tokens']}, far={comp['far_tokens']}")
        print(f"  Compress time: {comp['compress_time_s']:.2f}s")

        # Throughput (each method in clean state)
        methods = [
            ("standard", "Standard fp16"),
            ("fused", "Fused kernel"),
        ]

        ctx_results = {"compression": comp, "throughput": {}}
        for method_key, method_name in methods:
            gc.collect()
            result = measure_method(model, tokenizer, prompt, method_key, args.gen_tokens, cfg)
            tps = result["tok_per_s"]
            print(f"  {method_name:<20} {tps:>6.1f} tok/s ({result['elapsed_s']}s)")
            ctx_results["throughput"][method_key] = result

        all_results["contexts"][str(ctx)] = ctx_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
