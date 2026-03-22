"""
Dense needle-in-a-haystack heatmap for FoveatedKV paper.

5 context lengths x 11 depths x 3 trials (majority vote) x 3 methods.
Runs on 8GB Apple Silicon with Qwen2.5-0.5B-bf16.

Usage:
  uv run python benchmarks/benchmark_mlx_needle_heatmap.py
  uv run python benchmarks/benchmark_mlx_needle_heatmap.py --contexts 2048 4096 --trials 1
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import MLXTierConfig
from foveated_kv.mlx_generate import needle_test


def main():
    parser = argparse.ArgumentParser(description="Needle Heatmap for Paper")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 2048, 4096, 6144, 8192])
    parser.add_argument("--depths", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--output", default="results/paper/needle_heatmap.json")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)

    configs = [
        ("standard", None),
        ("foveated_10_90", MLXTierConfig()),
    ]

    results = {"model": args.model, "trials": args.trials, "grid": []}

    total = len(args.contexts) * len(args.depths)
    done = 0
    t_start = time.perf_counter()

    print(f"Grid: {len(args.contexts)} contexts x {len(args.depths)} depths x {args.trials} trials")
    print(f"{'Context':>8} {'Depth':>6} {'Std':>5} {'10/90':>5}")
    print("-" * 30)

    for ctx in args.contexts:
        for depth in args.depths:
            cell = {"context": ctx, "depth": depth}

            for method_name, cfg in configs:
                passes = 0
                for trial in range(args.trials):
                    try:
                        std_found, fov_found, info = needle_test(
                            model, tokenizer, context_len=ctx,
                            needle_depth=depth, cfg=cfg if cfg else MLXTierConfig(),
                        )
                        found = fov_found if cfg else std_found
                        if found:
                            passes += 1
                    except Exception:
                        pass
                cell[method_name] = passes >= (args.trials // 2 + 1)  # majority vote

            s = "Y" if cell.get("standard") else "N"
            f10 = "Y" if cell.get("foveated_10_90") else "N"
            done += 1
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"{ctx:>8} {depth:>6.1f} {s:>5} {f10:>5}  [{done}/{total}, ETA {eta:.0f}s]")

            results["grid"].append(cell)

    # Summary
    for method_name, _ in configs:
        total_pass = sum(1 for c in results["grid"] if c.get(method_name))
        total_cells = len(results["grid"])
        print(f"\n{method_name}: {total_pass}/{total_cells} ({100*total_pass/total_cells:.0f}%)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
