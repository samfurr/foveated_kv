"""
Promotion recovery benchmark: multi-fact retrieval from compressed KV cache.

Demonstrates that spike detection + promotion recovers facts lost to
quantization in the far tier. Uses a biographical needle with 4 checkable
facts buried in filler text, requiring sustained multi-token retrieval.

Three modes compared:
  1. Standard (full fp16 cache) — baseline
  2. Foveated + promotion — spike detection recovers demoted tokens
  3. Foveated no promotion — compressed, facts degraded

Key result (0.5B, ~2K context, 5% near):
  Standard:          88% facts retrieved
  Foveated + promo:  88% facts retrieved  (matches baseline)
  Foveated no promo: 50% facts retrieved  (promotion recovers the gap)

Usage:
  uv run python benchmarks/benchmark_promotion_recovery.py
  uv run python benchmarks/benchmark_promotion_recovery.py --contexts 1024 2048 --near-pct 0.02
"""

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from foveated_kv.mlx_foveated import MLXTierConfig
from foveated_kv.mlx_generate import generate_fused, _generate_short

FILLER = (
    "This document discusses general information about technology "
    "and innovation in modern computing systems. "
)
CHECK_PHRASES = ["1943", "quantum flux", "1987", "Nobel"]
PERSON = "Dr. Vasquez"
NEEDLE = (
    f"{PERSON} was born on March 17, 1943. "
    f"She invented the quantum flux capacitor "
    f"and won the Nobel Prize in Physics in 1987. "
)
RETRIEVAL = (
    f"\nWrite a 3-sentence biography of {PERSON} based only on "
    f"the information above. Include all dates and achievements: "
)


def make_prompt(tokenizer, ctx_len, needle_depth):
    filler_tokens = len(tokenizer.encode(FILLER))
    available = ctx_len - len(tokenizer.encode(NEEDLE)) - 80
    n_filler = available // filler_tokens
    n_before = int(n_filler * needle_depth)
    n_after = n_filler - n_before

    prompt = FILLER * n_before + NEEDLE + FILLER * n_after + RETRIEVAL
    tokens = tokenizer.encode(prompt)[:ctx_len]
    return tokenizer.decode(tokens)


def count_facts(text):
    return sum(1 for p in CHECK_PHRASES if p in text)


def main():
    parser = argparse.ArgumentParser(description="Promotion Recovery Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-4bit")
    parser.add_argument("--contexts", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--depths", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    parser.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30, 40, 50])
    parser.add_argument("--near-pct", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--output", default="results/promotion_recovery.json")
    args = parser.parse_args()

    from mlx_lm import load
    print(f"Loading {args.model}...")
    model, tokenizer = load(args.model)

    cfg = MLXTierConfig(near_pct=args.near_pct)

    results = {
        "model": args.model,
        "near_pct": args.near_pct,
        "check_phrases": CHECK_PHRASES,
        "grid": [],
    }

    print(f"\nGrid: {len(args.contexts)} contexts x {len(args.depths)} depths x {len(args.seeds)} seeds")
    print(f"Near tier: {args.near_pct:.0%}")
    print(f"\n{'Ctx':>6} {'Depth':>6} {'Seed':>5} {'Std':>5} {'Fov+P':>6} {'Fov-P':>6}  Note")
    print("-" * 55)

    totals = {"std": 0, "fov_p": 0, "fov_np": 0, "n": 0}

    for ctx in args.contexts:
        for depth in args.depths:
            for seed in args.seeds:
                random.seed(seed)
                prompt = make_prompt(tokenizer, ctx, depth)
                actual_len = len(tokenizer.encode(prompt))

                try:
                    std_text, _ = _generate_short(
                        model, tokenizer, prompt, max_tokens=args.max_tokens
                    )
                    std_hits = count_facts(std_text)

                    fov_text, _ = generate_fused(
                        model, tokenizer, prompt,
                        max_tokens=args.max_tokens, cfg=cfg, enable_promotion=True,
                    )
                    fov_hits = count_facts(fov_text)

                    fov_np_text, _ = generate_fused(
                        model, tokenizer, prompt,
                        max_tokens=args.max_tokens, cfg=cfg, enable_promotion=False,
                    )
                    fov_np_hits = count_facts(fov_np_text)
                except Exception as e:
                    print(f"{actual_len:>6} {depth:>6.1f} {seed:>5}   ERROR: {e}")
                    continue

                note = ""
                if fov_hits > fov_np_hits:
                    note = "<<< promotion recovers"
                elif fov_np_hits > fov_hits:
                    note = "(!)"

                print(
                    f"{actual_len:>6} {depth:>6.1f} {seed:>5} "
                    f"{std_hits:>4}/4 {fov_hits:>5}/4 {fov_np_hits:>5}/4  {note}"
                )

                totals["std"] += std_hits
                totals["fov_p"] += fov_hits
                totals["fov_np"] += fov_np_hits
                totals["n"] += len(CHECK_PHRASES)

                results["grid"].append({
                    "context": actual_len,
                    "depth": depth,
                    "seed": seed,
                    "standard": std_hits,
                    "foveated_promo": fov_hits,
                    "foveated_nopromo": fov_np_hits,
                    "standard_output": std_text[:200],
                    "foveated_promo_output": fov_text[:200],
                    "foveated_nopromo_output": fov_np_text[:200],
                })

    n = totals["n"]
    print(f"\n--- Summary ---")
    print(f"  Standard:           {totals['std']:>3}/{n} ({100*totals['std']/n:.0f}%)")
    print(f"  Foveated + promo:   {totals['fov_p']:>3}/{n} ({100*totals['fov_p']/n:.0f}%)")
    print(f"  Foveated no promo:  {totals['fov_np']:>3}/{n} ({100*totals['fov_np']/n:.0f}%)")

    gap = totals["fov_p"] - totals["fov_np"]
    if gap > 0:
        print(f"\n  Promotion recovered {gap} additional facts ({100*gap/n:.0f}% of total)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
