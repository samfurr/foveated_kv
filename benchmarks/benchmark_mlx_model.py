"""
Real model benchmarks for MLX foveated KV cache.

Tests accuracy with actual LLM inference using mlx-lm models.
Measures needle-in-a-haystack retrieval, perplexity, and generation quality.

Usage:
  uv run python benchmarks/benchmark_mlx_model.py --model mlx-community/Qwen2.5-0.5B-Instruct-4bit
  uv run python benchmarks/benchmark_mlx_model.py --needle --contexts 2048 4096 8192
  uv run python benchmarks/benchmark_mlx_model.py --ppl --context-len 4096
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipmap_kv.mlx_foveated import MLXTierConfig
from mipmap_kv.mlx_generate import (
    compute_perplexity,
    generate_fused,
    needle_test,
)


def load_model(model_name: str):
    """Load an mlx-lm model."""
    from mlx_lm import load

    print(f"Loading {model_name}...")
    model, tokenizer = load(model_name)
    print(f"  Model loaded.")
    return model, tokenizer


def run_needle_benchmark(
    model, tokenizer, contexts: list[int], depths: list[float], cfg: MLXTierConfig
) -> list[dict]:
    """Run needle-in-a-haystack across context lengths and depths."""
    results = []

    print(f"\n{'Context':>8} {'Depth':>6} {'Std':>5} {'Fov':>5} {'Passkey':>8}")
    print(f"{'-'*40}")

    for ctx in contexts:
        for depth in depths:
            try:
                std_found, fov_found, info = needle_test(
                    model, tokenizer, context_len=ctx, needle_depth=depth, cfg=cfg,
                )
                status_std = "✓" if std_found else "✗"
                status_fov = "✓" if fov_found else "✗"
                print(
                    f"{ctx:>8} {depth:>6.1f} {status_std:>5} {status_fov:>5} {info['passkey']:>8}"
                )
                results.append({
                    "context_len": ctx,
                    "needle_depth": depth,
                    "standard_found": std_found,
                    "foveated_found": fov_found,
                    "passkey": info["passkey"],
                })
            except Exception as e:
                print(f"{ctx:>8} {depth:>6.1f}  ERROR: {e}")
                results.append({
                    "context_len": ctx,
                    "needle_depth": depth,
                    "error": str(e),
                })

    # Summary
    total = len([r for r in results if "error" not in r])
    std_pass = sum(1 for r in results if r.get("standard_found"))
    fov_pass = sum(1 for r in results if r.get("foveated_found"))
    print(f"\nNeedle results: Standard {std_pass}/{total}, Foveated {fov_pass}/{total}")

    return results


def run_ppl_benchmark(
    model, tokenizer, context_len: int, eval_len: int, cfg: MLXTierConfig, text: str
) -> dict:
    """Run perplexity comparison."""
    print(f"\nComputing PPL (context={context_len}, eval={eval_len})...")
    start = time.perf_counter()
    std_ppl, fov_ppl = compute_perplexity(
        model, tokenizer, text,
        context_len=context_len, eval_len=eval_len, cfg=cfg,
    )
    elapsed = time.perf_counter() - start

    print(f"  Standard PPL: {std_ppl:.4f}")
    print(f"  Foveated PPL: {fov_ppl:.4f}")
    print(f"  Ratio:        {fov_ppl / std_ppl:.4f}x")
    print(f"  Time:         {elapsed:.1f}s")

    return {
        "context_len": context_len,
        "eval_len": eval_len,
        "standard_ppl": std_ppl,
        "foveated_ppl": fov_ppl,
        "ppl_ratio": fov_ppl / std_ppl,
        "elapsed_s": elapsed,
    }


def run_generation_benchmark(
    model, tokenizer, prompts: list[str], max_tokens: int, cfg: MLXTierConfig
) -> list[dict]:
    """Compare generation output between standard and foveated."""
    results = []
    for prompt in prompts:
        print(f"\nPrompt: {prompt[:80]}...")

        # Foveated (fused path)
        fov_text, fov_stats = generate_fused(
            model, tokenizer, prompt, max_tokens=max_tokens, cfg=cfg,
            enable_promotion=False,
        )
        print(f"  Foveated: {fov_text[:120]}...")

        results.append({
            "prompt": prompt[:200],
            "foveated_output": fov_text,
            "stats": fov_stats,
        })

    return results


def run_promotion_benchmark(
    model, tokenizer, context_len: int, gen_tokens: int, cfg: MLXTierConfig,
    use_disk: bool = True,
) -> dict:
    """Test the full promotion pipeline: spike detection → archive read → tier update.

    Generates tokens with promotion-aware decode, tracking how many spikes
    are detected and promotions executed. Optionally offloads archives to disk.
    """
    import tempfile

    filler = (
        "This document discusses various topics in science, technology, and history. "
        "The weather patterns in different regions vary significantly throughout the year. "
    )
    needle = "The secret activation code is ALPHA-7749. Remember this code."
    retrieval = "\nRepeat the activation code mentioned earlier: "

    filler_tokens = len(tokenizer.encode(filler))
    n_filler = max(1, (context_len - 200) // filler_tokens)
    prompt = filler * n_filler + needle + filler * (n_filler // 4) + retrieval

    # Truncate
    prompt_tokens = tokenizer.encode(prompt)[:context_len]
    prompt = tokenizer.decode(prompt_tokens)
    actual_len = len(prompt_tokens)

    print(f"\n  Context: {actual_len} tokens, generating {gen_tokens} tokens")
    print(f"  Disk offload: {use_disk}")

    tmpdir = tempfile.mkdtemp() if use_disk else None

    start = time.perf_counter()
    output, stats = generate_fused(
        model, tokenizer, prompt,
        max_tokens=gen_tokens, cfg=cfg,
        disk_archive_dir=tmpdir,
        enable_promotion=True,
    )
    elapsed = time.perf_counter() - start

    print(f"  Output: {output[:100]}...")
    print(f"  Spikes detected: {stats.get('spikes_detected', 0)}")
    print(f"  Promotions done: {stats.get('promotions_completed', 0)}")
    print(f"  Promotions used: {stats.get('promotions_applied', 0)}")
    print(f"  Deduplicated:    {stats.get('spikes_deduplicated', 0)}")
    print(f"  Memory saved:    {stats['mem_saved_mb']:.1f} MB (archive → disk)")
    print(f"  Time:            {elapsed:.1f}s ({gen_tokens / elapsed:.1f} tok/s)")

    found_code = "ALPHA-7749" in output or "7749" in output
    print(f"  Code retrieved:  {'Yes' if found_code else 'No'}")

    # Baseline speed: standard cache (no foveation)
    from mipmap_kv.mlx_generate import _generate_short  # noqa: E402
    std_start = time.perf_counter()
    _generate_short(model, tokenizer, prompt, max_tokens=gen_tokens)
    std_elapsed = time.perf_counter() - std_start
    std_tps = gen_tokens / std_elapsed
    print(f"  Standard:        {std_elapsed:.1f}s ({std_tps:.1f} tok/s)")
    print(f"  Foveated+promo:  {elapsed:.1f}s ({gen_tokens / elapsed:.1f} tok/s)")

    stats.update({
        "context_len": actual_len,
        "elapsed_s": elapsed,
        "tok_per_s": gen_tokens / elapsed,
        "std_elapsed_s": std_elapsed,
        "std_tok_per_s": std_tps,
        "code_found": found_code,
    })

    # Cleanup
    if tmpdir:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    return stats


def get_eval_text(tokenizer, min_tokens: int = 10000) -> str:
    """Get evaluation text for PPL measurement."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = " ".join([x["text"] for x in ds if len(x["text"].strip()) > 100])
        tokens = tokenizer.encode(text)
        if len(tokens) >= min_tokens:
            return text
    except Exception:
        pass

    # Fallback: repeated diverse text
    text = (
        "The development of artificial intelligence has progressed rapidly. "
        "Machine learning models now process vast amounts of data efficiently. "
        "Neural networks have transformed computer vision and natural language processing. "
        "Large language models demonstrate remarkable capabilities in text generation. "
        "The field continues to evolve with new architectures and training methods. "
    ) * (min_tokens // 20)
    return text


def main():
    parser = argparse.ArgumentParser(description="MLX Foveated KV Cache Model Benchmarks")
    parser.add_argument(
        "--model", type=str,
        default="mlx-community/Qwen2.5-0.5B-Instruct-bf16",
        help="mlx-lm model name (use bf16/fp16 for fair KV cache testing)",
    )
    parser.add_argument("--needle", action="store_true", help="Run needle-in-haystack")
    parser.add_argument("--ppl", action="store_true", help="Run perplexity comparison")
    parser.add_argument("--generate", action="store_true", help="Run generation comparison")
    parser.add_argument("--promotion", action="store_true", help="Test promotion pipeline with disk offload")
    parser.add_argument(
        "--contexts", nargs="+", type=int, default=[2048, 4096],
        help="Context lengths for needle test",
    )
    parser.add_argument(
        "--depths", nargs="+", type=float,
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help="Needle depths",
    )
    parser.add_argument("--context-len", type=int, default=4096, help="PPL context length")
    parser.add_argument("--eval-len", type=int, default=128, help="PPL eval length")
    parser.add_argument("--foveal-pct", type=float, default=0.05)
    parser.add_argument("--periph-pct", type=float, default=0.25)
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for generation")
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    # Default: run all
    if not (args.needle or args.ppl or args.generate or args.promotion):
        args.needle = True
        args.promotion = True

    cfg = MLXTierConfig(
        foveal_pct=args.foveal_pct,
        periph_pct=args.periph_pct,
    )

    model, tokenizer = load_model(args.model)
    all_results = {"model": args.model, "config": {"foveal_pct": cfg.foveal_pct, "periph_pct": cfg.periph_pct}}

    if args.needle:
        print("\n" + "=" * 60)
        print("NEEDLE-IN-A-HAYSTACK")
        print("=" * 60)
        all_results["needle"] = run_needle_benchmark(
            model, tokenizer, args.contexts, args.depths, cfg,
        )

    if args.ppl:
        print("\n" + "=" * 60)
        print("PERPLEXITY")
        print("=" * 60)
        text = get_eval_text(tokenizer, min_tokens=args.context_len + args.eval_len + 500)
        all_results["ppl"] = run_ppl_benchmark(
            model, tokenizer, args.context_len, args.eval_len, cfg, text,
        )

    if args.promotion:
        print("\n" + "=" * 60)
        print("PROMOTION PIPELINE + DISK OFFLOAD")
        print("=" * 60)
        promo_results = []
        for ctx in args.contexts:
            result = run_promotion_benchmark(
                model, tokenizer, ctx, args.max_tokens, cfg, use_disk=True,
            )
            promo_results.append(result)
        all_results["promotion"] = promo_results

    if args.generate:
        print("\n" + "=" * 60)
        print("GENERATION")
        print("=" * 60)
        prompts = [
            "Explain the concept of attention in transformer models in simple terms:",
            "Write a Python function that computes the Fibonacci sequence:",
        ]
        all_results["generation"] = run_generation_benchmark(
            model, tokenizer, prompts, args.max_tokens, cfg,
        )

    # Save results
    out_path = args.output or "results/mlx_model_benchmark.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
