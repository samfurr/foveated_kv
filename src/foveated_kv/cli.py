"""CLI for FoveatedKV: generate text with importance-adaptive KV cache."""

import argparse
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        prog="foveated-kv",
        description="Generate text with FoveatedKV compressed KV cache on Apple Silicon",
    )
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate text from a prompt")
    gen.add_argument("--model", type=str, default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
                     help="HuggingFace model ID (default: Qwen2.5-0.5B-Instruct-4bit)")
    gen.add_argument("--prompt", type=str, required=True, help="Input prompt")
    gen.add_argument("--max-tokens", type=int, default=200, help="Max tokens to generate")
    gen.add_argument("--temp", type=float, default=0.0, help="Sampling temperature (0 = greedy)")
    gen.add_argument("--near-pct", type=float, default=0.10,
                     help="Fraction of tokens kept at full precision (default: 0.10)")
    gen.add_argument("--compress", type=str, default="fp8", choices=["fp8", "turbo"],
                     help="Compression method: fp8 (2x, default) or turbo (3.2x TurboQuant)")
    gen.add_argument("--no-promotion", action="store_true", help="Disable spike promotion")
    gen.add_argument("--standard", action="store_true",
                     help="Use standard (non-foveated) cache for baseline comparison")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "generate":
        _run_generate(args)


def _run_generate(args):
    import mlx.core as mx

    print(f"Loading model: {args.model}")
    t0 = time.perf_counter()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.1f}s")

    if args.standard:
        _run_standard(model, tokenizer, args)
    else:
        _run_foveated(model, tokenizer, args)


def _run_standard(model, tokenizer, args):
    import mlx.core as mx
    from .mlx_generate import _generate_short

    print(f"Generating with STANDARD cache (max {args.max_tokens} tokens, temp={args.temp})...\n")

    tokens = mx.array(tokenizer.encode(args.prompt)).reshape(1, -1)
    prompt_len = tokens.shape[1]

    t0 = time.perf_counter()
    text, info = _generate_short(model, tokenizer, args.prompt, max_tokens=args.max_tokens)
    elapsed = time.perf_counter() - t0

    n_gen = info["generated_tokens"]
    print(text)
    print("\n--- Stats (standard) ---")
    print(f"Prompt tokens:    {prompt_len}")
    print(f"Generated tokens: {n_gen}")
    print(f"Total time:       {elapsed:.3f}s")
    print(f"Tokens/sec:       {n_gen / max(elapsed, 1e-6):.1f}")


def _run_foveated(model, tokenizer, args):
    from .mlx_foveated import MLXTierConfig
    from .mlx_generate import generate_fused

    cfg = MLXTierConfig(near_pct=args.near_pct, compress_method=args.compress)
    method_label = "TurboQuant 3.2x" if args.compress == "turbo" else "fp8 2x"

    print(f"Generating with FOVEATED cache [{method_label}] (max {args.max_tokens} tokens, temp={args.temp})...\n")

    text, stats = generate_fused(
        model, tokenizer, args.prompt,
        max_tokens=args.max_tokens,
        cfg=cfg,
        temp=args.temp,
        enable_promotion=not args.no_promotion,
    )

    print(text)
    print("\n--- Stats (foveated) ---")
    print(f"Prompt tokens:    {stats['prompt_tokens']}")
    print(f"Generated tokens: {stats['generated_tokens']}")
    print(f"Prefill time:     {stats['prefill_time_s']:.3f}s")
    print(f"Decode time:      {stats['decode_time_s']:.3f}s")
    print(f"Tokens/sec:       {stats['tokens_per_second']:.1f}")
    print(f"Compression:      {method_label}")
    print(f"Near tier:        {args.near_pct:.0%} of context at fp16")
    print(f"Memory saved:     {stats['mem_saved_mb']:.1f} MB (disk offload)")
    print(f"Compression:      {stats['mem_quantized_mb']:.1f} MB quantized cache")

    if "promotions" in stats:
        print(f"Promotions:       {stats['promotions']}")
    if "promotion_latency_p99_ms" in stats:
        print(f"Promotion p99:    {stats['promotion_latency_p99_ms']:.1f} ms")


if __name__ == "__main__":
    main()
