"""
MLX native benchmark: Foveated vs standard attention on Apple Silicon.

Measures latency, memory, and quality for the MLX foveated KV cache
implementation against standard fp16 attention.

Usage:
  uv run python benchmarks/benchmark_mlx.py
  uv run python benchmarks/benchmark_mlx.py --contexts 1024 4096 16384
  uv run python benchmarks/benchmark_mlx.py --n-kv-heads 8 --n-q-heads 32 --head-dim 128
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import (
    MLXFoveatedKVCache,
    MLXFoveatedLayer,
    MLXTierConfig,
    standard_attention_mlx,
)
from foveated_kv.mlx_quantize import (
    dequantize_int4_per_token,
    dequantize_int8_per_channel,
    dequantize_int8_per_token,
    quantize_int4_per_token,
    quantize_int8_per_channel,
    quantize_int8_per_token,
)


def cosine_similarity(a: mx.array, b: mx.array) -> float:
    a_flat = a.reshape(-1).astype(mx.float32)
    b_flat = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_flat * b_flat)
    norm_a = mx.sqrt(mx.sum(a_flat * a_flat))
    norm_b = mx.sqrt(mx.sum(b_flat * b_flat))
    result = dot / (norm_a * norm_b + 1e-8)
    mx.eval(result)
    return result.item()


def mae(a: mx.array, b: mx.array) -> float:
    diff = mx.abs(a.astype(mx.float32) - b.astype(mx.float32))
    result = mx.mean(diff)
    mx.eval(result)
    return result.item()


def benchmark_latency(
    fn, warmup: int = 10, iterations: int = 100, label: str = ""
) -> dict:
    """Time a function with proper MLX synchronization."""
    # Warmup
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    # Timed
    start = time.perf_counter()
    for _ in range(iterations):
        out = fn()
        mx.eval(out)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iterations) * 1000
    return {"label": label, "avg_ms": avg_ms, "iterations": iterations}


def generate_synthetic_kv(
    B: int, n_kv_heads: int, seq_len: int, head_dim: int
) -> tuple[mx.array, mx.array]:
    """Generate random KV tensors (simulating post-prefill cache)."""
    keys = mx.random.normal((B, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    values = mx.random.normal((B, n_kv_heads, seq_len, head_dim)).astype(mx.float16)
    mx.eval(keys, values)
    return keys, values


def run_quality_benchmark(
    B: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    configs: list[tuple[str, MLXTierConfig]],
) -> list[dict]:
    """Measure attention quality: foveated vs exact fp16."""
    keys, values = generate_synthetic_kv(B, n_kv_heads, seq_len, head_dim)
    query = mx.random.normal((B, n_q_heads, 1, head_dim)).astype(mx.float16)
    mx.eval(query)

    # Exact fp16 reference
    ref_output = standard_attention_mlx(query, keys, values)
    mx.eval(ref_output)

    results = []
    for name, cfg in configs:
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        stats = cache.compress()

        fov_output = cache.attend(0, query)
        mx.eval(fov_output)

        cos = cosine_similarity(fov_output, ref_output)
        err = mae(fov_output, ref_output)
        mem = cache.layers[0].memory_bytes()

        bytes_per_elem = mem["total_quantized"] / (seq_len * n_kv_heads * head_dim * 2)

        results.append(
            {
                "config": name,
                "seq_len": seq_len,
                "cosine": cos,
                "mae": err,
                "bytes_per_elem": bytes_per_elem,
                "compression": stats["compression"],
                "near_tokens": cache.layers[0].near_k.shape[2],
                "far_tokens": cache.layers[0].far_k.shape[2],
            }
        )

    return results


def run_latency_benchmark(
    B: int,
    n_q_heads: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    warmup: int = 10,
    iterations: int = 100,
) -> dict:
    """Compare decode latency: standard fp16 vs foveated vs fused vs uniform INT8."""
    keys, values = generate_synthetic_kv(B, n_kv_heads, seq_len, head_dim)
    query = mx.random.normal((B, n_q_heads, 1, head_dim)).astype(mx.float16)
    mx.eval(query)

    # Standard fp16
    std_timing = benchmark_latency(
        lambda: standard_attention_mlx(query, keys, values),
        warmup=warmup,
        iterations=iterations,
        label=f"fp16 S={seq_len}",
    )

    # Foveated (eager)
    cfg = MLXTierConfig()
    cache = MLXFoveatedKVCache(cfg)
    cache.update(keys, values, 0)
    cache.compress()

    fov_timing = benchmark_latency(
        lambda: cache.attend(0, query),
        warmup=warmup,
        iterations=iterations,
        label=f"foveated S={seq_len}",
    )

    # Foveated (fused Metal kernel — zero intermediate materialization)
    fov_fused_timing = benchmark_latency(
        lambda: cache.attend_fused(0, query),
        warmup=warmup,
        iterations=iterations,
        label=f"foveated-fused S={seq_len}",
    )

    # Uniform INT8 baseline (dequant all to fp16 + SDPA)
    k_q, k_s, k_z = quantize_int8_per_channel(keys)
    v_q, v_s, v_z = quantize_int8_per_token(values)
    mx.eval(k_q, k_s, k_z, v_q, v_s, v_z)

    def uniform_int8_attend():
        k_fp = dequantize_int8_per_channel(k_q, k_s, k_z)
        v_fp = dequantize_int8_per_token(v_q, v_s, v_z)
        return standard_attention_mlx(query, k_fp, v_fp)

    int8_timing = benchmark_latency(
        uniform_int8_attend,
        warmup=warmup,
        iterations=iterations,
        label=f"uniform-int8 S={seq_len}",
    )

    return {
        "seq_len": seq_len,
        "fp16_ms": std_timing["avg_ms"],
        "foveated_ms": fov_timing["avg_ms"],
        "foveated_fused_ms": fov_fused_timing["avg_ms"],
        "uniform_int8_ms": int8_timing["avg_ms"],
        "speedup_vs_fp16": std_timing["avg_ms"] / max(fov_timing["avg_ms"], 1e-6),
        "speedup_fused_vs_fp16": std_timing["avg_ms"]
        / max(fov_fused_timing["avg_ms"], 1e-6),
    }


def run_memory_benchmark(
    B: int, n_kv_heads: int, head_dim: int, seq_len: int
) -> dict:
    """Compare memory footprint across methods."""
    fp16_bytes = B * n_kv_heads * seq_len * head_dim * 2 * 2  # K + V

    keys, values = generate_synthetic_kv(B, n_kv_heads, seq_len, head_dim)

    configs = [
        ("10/90", MLXTierConfig()),
    ]

    results = {"seq_len": seq_len, "fp16_mb": fp16_bytes / (1024 * 1024)}

    for name, cfg in configs:
        cache = MLXFoveatedKVCache(cfg)
        cache.update(keys, values, 0)
        cache.compress()
        mem = cache.layers[0].memory_bytes()
        results[f"{name}_quantized_mb"] = mem["total_quantized"] / (1024 * 1024)
        results[f"{name}_compression"] = fp16_bytes / max(mem["total_quantized"], 1)
        results[f"{name}_with_archive_mb"] = mem["total_with_archive"] / (1024 * 1024)

    return results


def run_bandwidth_analysis(
    n_kv_heads: int, head_dim: int, seq_len: int
) -> dict:
    """Theoretical bandwidth analysis for Apple Silicon.

    Computes bytes read from memory for each method, which determines
    decode latency since single-token decode is memory-bandwidth-bound.
    """
    D = head_dim
    S = seq_len

    # Standard fp16: read all K + V
    fp16_bytes = S * n_kv_heads * D * 2 * 2  # 2 bytes per fp16, K + V

    # Foveated 10/90 tier sizes (2-tier)
    R = int(S * 0.10)  # near
    F = S - R  # far

    # Near: fp16 K + fp16 V = 4 bytes/elem
    near_bytes = R * n_kv_heads * D * 4
    # Far: fp8 E4M3 K (1 byte/elem) + int4 V (0.5 bytes/elem)
    #   + per-token scale/zero (2 × fp16 = 4 bytes/token)
    far_k_bytes = F * n_kv_heads * D * 1        # fp8 K
    far_v_bytes = F * n_kv_heads * D // 2        # int4 nibble-packed V
    far_sz_bytes = F * n_kv_heads * 4            # per-token scale + zero (fp16 each)
    far_bytes = far_k_bytes + far_v_bytes + far_sz_bytes

    fov_total = near_bytes + far_bytes

    # NOTE: Without fused dequant, actual bandwidth is higher because
    # dequantized fp16 intermediates are written and read again.
    dequant_overhead = F * n_kv_heads * D * 4  # K + V materialized as fp16

    return {
        "seq_len": S,
        "fp16_read_mb": fp16_bytes / (1024 * 1024),
        "foveated_read_mb": fov_total / (1024 * 1024),
        "foveated_with_dequant_mb": (fov_total + dequant_overhead) / (1024 * 1024),
        "theoretical_speedup_fused": fp16_bytes / max(fov_total, 1),
        "theoretical_speedup_unfused": fp16_bytes / max(fov_total + dequant_overhead, 1),
        "tier_sizes": {"near": R, "far": F},
    }


def main():
    parser = argparse.ArgumentParser(description="MLX Foveated KV Cache Benchmark")
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[512, 1024, 4096, 8192, 16384, 32768],
        help="Context lengths to benchmark",
    )
    parser.add_argument("--n-q-heads", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--output", type=str, default=None, help="JSON output file path"
    )
    parser.add_argument(
        "--skip-latency", action="store_true", help="Skip latency benchmarks"
    )
    args = parser.parse_args()

    B = args.batch_size
    H_q = args.n_q_heads
    H_kv = args.n_kv_heads
    D = args.head_dim

    print(f"MLX Foveated KV Cache Benchmark")
    print(f"  Model config: B={B}, H_q={H_q}, H_kv={H_kv}, D={D}")
    print(f"  GQA ratio: {H_q // H_kv}:1")
    print(f"  Contexts: {args.contexts}")
    print()

    all_results = {
        "config": {
            "batch_size": B,
            "n_q_heads": H_q,
            "n_kv_heads": H_kv,
            "head_dim": D,
            "gqa_ratio": H_q // H_kv,
        },
        "quality": [],
        "latency": [],
        "memory": [],
        "bandwidth": [],
    }

    # === Quality ===
    print("=" * 70)
    print("QUALITY: Foveated vs Exact fp16")
    print("=" * 70)

    quality_configs = [
        ("foveated-10/90", MLXTierConfig()),
    ]

    for seq_len in args.contexts:
        results = run_quality_benchmark(B, H_q, H_kv, D, seq_len, quality_configs)
        all_results["quality"].extend(results)

        print(f"\n  Context: {seq_len:,} tokens")
        print(
            f"  {'Config':<22} {'Cosine':>10} {'MAE':>12} {'Bytes/elem':>12} {'Compression':>12}"
        )
        print(f"  {'-'*68}")
        for r in results:
            print(
                f"  {r['config']:<22} {r['cosine']:>10.6f} {r['mae']:>12.6f} "
                f"{r['bytes_per_elem']:>12.3f} {r['compression']:>11.2f}x"
            )

    # === Memory ===
    print()
    print("=" * 70)
    print("MEMORY: Compression ratios")
    print("=" * 70)

    for seq_len in args.contexts:
        mem = run_memory_benchmark(B, H_kv, D, seq_len)
        all_results["memory"].append(mem)

        print(f"\n  Context: {seq_len:,} tokens | fp16: {mem['fp16_mb']:.1f} MB")
        for cfg_name in ["10/90"]:
            q_mb = mem[f"{cfg_name}_quantized_mb"]
            comp = mem[f"{cfg_name}_compression"]
            print(f"    {cfg_name}: {q_mb:.1f} MB ({comp:.2f}x compression)")

    # === Bandwidth Analysis ===
    print()
    print("=" * 70)
    print("BANDWIDTH: Theoretical speedup analysis")
    print("=" * 70)

    for seq_len in args.contexts:
        bw = run_bandwidth_analysis(H_kv, D, seq_len)
        all_results["bandwidth"].append(bw)

        print(f"\n  Context: {seq_len:,} tokens")
        print(f"    fp16 read:              {bw['fp16_read_mb']:.1f} MB")
        print(f"    Foveated read (fused):  {bw['foveated_read_mb']:.1f} MB")
        print(f"    Foveated read (unfused):{bw['foveated_with_dequant_mb']:.1f} MB")
        print(
            f"    Speedup (fused):        {bw['theoretical_speedup_fused']:.2f}x"
        )
        print(
            f"    Speedup (unfused):      {bw['theoretical_speedup_unfused']:.2f}x"
        )

    # === Latency ===
    if not args.skip_latency:
        print()
        print("=" * 70)
        print("LATENCY: Decode step timing")
        print("=" * 70)

        print(
            f"\n  {'Context':>10} {'fp16':>10} {'Foveated':>10} "
            f"{'Fused':>10} {'INT8':>10} {'Fsd.Spd':>10}"
        )
        print(f"  {'-'*66}")

        for seq_len in args.contexts:
            lat = run_latency_benchmark(
                B,
                H_q,
                H_kv,
                D,
                seq_len,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            all_results["latency"].append(lat)

            print(
                f"  {seq_len:>10,} {lat['fp16_ms']:>9.3f}ms {lat['foveated_ms']:>9.3f}ms "
                f"{lat['foveated_fused_ms']:>9.3f}ms "
                f"{lat['uniform_int8_ms']:>9.3f}ms "
                f"{lat['speedup_fused_vs_fp16']:>9.2f}x"
            )

    # === Save results ===
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        # Default output path
        os.makedirs("results", exist_ok=True)
        out_path = "results/mlx_benchmark.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
