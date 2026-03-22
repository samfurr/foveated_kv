"""
Promotion quality benchmark: does the override buffer improve accuracy?

Places a passkey at a controlled depth (early = far tier after compression),
then generates with the fused kernel comparing:
  1. Frozen tiers (no overrides)
  2. Fused kernel + async overrides (promoted)

Both always use the fused Metal kernel. The only difference is whether the
background worker writes promoted fp16 K,V into the override buffer.

Key hypothesis: passkeys placed early in context land in the far tier (INT4 V).
When the model attends to them, spike detection fires and the kernel reads exact
fp16 from the override buffer instead of dequanting INT4. This should improve
retrieval accuracy at longer contexts.

Multiple trials per config smooth out argpartition non-determinism.

Usage:
  uv run python benchmarks/benchmark_promotion_quality.py
  uv run python benchmarks/benchmark_promotion_quality.py --contexts 1024 2048 4096
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig
from foveated_kv.mlx_generate import (
    FusedCacheWrapper,
    install_fused_sdpa,
    uninstall_fused_sdpa,
    reset_fused_layer_counter,
    prefill_and_compress,
    _fused_state,
)
from foveated_kv.disk_archive import offload_cache_to_disk


def build_passkey_prompt(tokenizer, context_len: int, depth: float, passkey: str) -> str:
    """Build a prompt with a passkey buried at a specific depth.

    Args:
        tokenizer: tokenizer
        context_len: total context length in tokens
        depth: 0.0 = very start, 1.0 = end. 0.1 = early (likely far tier).
        passkey: the string to bury

    Returns:
        prompt string (truncated to context_len tokens)
    """
    filler = (
        "This document contains various observations about natural phenomena. "
        "The weather patterns indicate a shift in atmospheric conditions. "
        "Researchers have noted significant changes in ocean temperatures. "
    )
    needle = f"The secret passkey is {passkey}. Remember this passkey."
    retrieval = f"\n\nWhat is the secret passkey mentioned in the text above? The passkey is: "

    filler_tokens = len(tokenizer.encode(filler))
    needle_tokens = len(tokenizer.encode(needle))
    retrieval_tokens = len(tokenizer.encode(retrieval))
    available = context_len - needle_tokens - retrieval_tokens - 10

    n_before = max(1, int(available * depth) // filler_tokens)
    n_after = max(1, int(available * (1 - depth)) // filler_tokens)

    prompt = filler * n_before + " " + needle + " " + filler * n_after + retrieval
    tokens = tokenizer.encode(prompt)[:context_len]
    return tokenizer.decode(tokens)


def generate_short(model, tokenizer, cache_wrappers, prefill_logits, max_tokens=20):
    """Generate tokens using fused kernel. Returns token list."""
    gen = []
    logits = prefill_logits[:, -1, :]
    for step in range(max_tokens):
        mx.eval(logits)
        if bool(mx.any(mx.isnan(logits)).item()):
            break
        tok = mx.argmax(logits, axis=-1)
        tok_id = tok.item()
        if tok_id == tokenizer.eos_token_id:
            break
        gen.append(tok_id)
        reset_fused_layer_counter()
        logits = model(tok.reshape(1, 1), cache=cache_wrappers)[:, -1, :]
    return gen


def run_frozen(model, tokenizer, tokens, cfg, max_tokens=20):
    """Fused kernel, frozen tiers (no overrides)."""
    fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)

    wrappers = [
        FusedCacheWrapper(l, i) if l else None
        for i, l in enumerate(fov_cache.layers)
    ]
    install_fused_sdpa(fov_cache)
    _fused_state._fused_wrappers = wrappers  # enable counter reset

    gen = generate_short(model, tokenizer, wrappers, prefill_logits, max_tokens)
    uninstall_fused_sdpa()
    return gen


def run_promoted(model, tokenizer, tokens, cfg, max_tokens=20):
    """Fused kernel + C++ promotion pipeline."""
    tmpdir = tempfile.mkdtemp(prefix="fov_promo_bench_")
    try:
        fov_cache, prefill_logits, _ = prefill_and_compress(model, tokens, cfg)
        disk_archives = offload_cache_to_disk(fov_cache, tmpdir)

        from foveated_kv.mlx_foveated import _cpp_available, _PromotionPipeline
        if not _cpp_available or _PromotionPipeline is None:
            raise RuntimeError("C++ extension required for promotion benchmark")

        import numpy as _np
        n_layers = len(fov_cache.layers)
        cpp_pipeline = _PromotionPipeline(n_layers)
        for i, layer in enumerate(fov_cache.layers):
            if layer is None:
                continue
            layer._ensure_kcache()
            handle = layer._kcache.get("cpp_handle")
            if handle is not None:
                cpp_pipeline.register_blob(i, handle.get_blob_info())
            if i < len(disk_archives) and disk_archives[i] is not None:
                archive = disk_archives[i]
                archive_idx = _np.array(archive.idx).flatten().tolist()
                cpp_pipeline.register_archive(
                    i, archive.path_k, archive.path_v,
                    archive.H_kv, archive.S_arc, archive.D,
                    archive_idx)

        wrappers = [
            FusedCacheWrapper(l, i) if l else None
            for i, l in enumerate(fov_cache.layers)
        ]
        install_fused_sdpa(fov_cache)
        _fused_state._fused_wrappers = wrappers
        _fused_state._cpp_pipeline_handle = cpp_pipeline

        gen = generate_short(model, tokenizer, wrappers, prefill_logits, max_tokens)

        stats = cpp_pipeline.get_stats()
        cpp_pipeline.stop()
        uninstall_fused_sdpa()
        return gen, stats
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_standard(model, tokenizer, tokens, max_tokens=20):
    """Standard fp16 attention (ground truth)."""
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    logits = model(tokens, cache=cache)
    mx.eval(logits)

    gen = []
    next_logits = logits[:, -1, :]
    for _ in range(max_tokens):
        tok = mx.argmax(next_logits, axis=-1)
        tok_id = tok.item()
        if tok_id == tokenizer.eos_token_id:
            break
        gen.append(tok_id)
        next_logits = model(tok.reshape(1, 1), cache=cache)[:, -1, :]
        mx.eval(next_logits)
    return gen


def main():
    parser = argparse.ArgumentParser(description="Promotion Quality Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--contexts", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--output", default="results/paper/promotion_quality.json")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    cfg = MLXTierConfig()

    results = []

    for ctx_len in args.contexts:
        for depth in args.depths:
            passkey = str(random.randint(10000, 99999))
            prompt = build_passkey_prompt(tokenizer, ctx_len, depth, passkey)
            tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
            actual_len = tokens.shape[1]

            print(f"\n{'='*60}")
            print(f"Context={actual_len}, depth={depth}, passkey={passkey}")
            print(f"{'='*60}")

            # Standard fp16 (ground truth, run once)
            std_gen = run_standard(model, tokenizer, tokens, args.max_tokens)
            std_text = tokenizer.decode(std_gen)
            std_found = passkey in std_text
            print(f"  Standard:  {'PASS' if std_found else 'FAIL'} -> {std_text[:60]!r}")

            frozen_pass = 0
            promo_pass = 0
            promo_spikes_total = 0
            promo_ov_total = 0

            for trial in range(args.trials):
                # Frozen
                fgen = run_frozen(model, tokenizer, tokens, cfg, args.max_tokens)
                ftext = tokenizer.decode(fgen)
                if passkey in ftext:
                    frozen_pass += 1

                # Promoted
                pgen, pstats = run_promoted(model, tokenizer, tokens, cfg, args.max_tokens)
                ptext = tokenizer.decode(pgen)
                if passkey in ptext:
                    promo_pass += 1
                promo_spikes_total += pstats["spikes_detected"]
                promo_ov_total += pstats["promotions_completed"]

                tag = f"t{trial}"
                f_ok = "PASS" if passkey in ftext else "FAIL"
                p_ok = "PASS" if passkey in ptext else "FAIL"
                print(f"  {tag} frozen={f_ok} promo={p_ok} spikes={pstats['spikes_detected']} ov={pstats['promotions_completed']}")

            frozen_rate = frozen_pass / args.trials
            promo_rate = promo_pass / args.trials
            avg_spikes = promo_spikes_total / args.trials
            avg_ov = promo_ov_total / args.trials

            print(f"  ----")
            print(f"  Frozen retrieval:   {frozen_pass}/{args.trials} ({frozen_rate:.0%})")
            print(f"  Promoted retrieval: {promo_pass}/{args.trials} ({promo_rate:.0%})")
            print(f"  Avg spikes: {avg_spikes:.1f}, avg overrides: {avg_ov:.1f}")

            results.append({
                "context_len": actual_len,
                "depth": depth,
                "passkey": passkey,
                "standard_found": std_found,
                "frozen_pass_rate": frozen_rate,
                "promoted_pass_rate": promo_rate,
                "frozen_pass": frozen_pass,
                "promoted_pass": promo_pass,
                "trials": args.trials,
                "avg_spikes": avg_spikes,
                "avg_overrides": avg_ov,
            })

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Context':>8} {'Depth':>6} {'Standard':>10} {'Frozen':>10} {'Promoted':>10} {'Spikes':>8}")
    print(f"{'-'*70}")
    for r in results:
        std = "PASS" if r["standard_found"] else "FAIL"
        frz = f"{r['frozen_pass']}/{r['trials']}"
        prm = f"{r['promoted_pass']}/{r['trials']}"
        print(f"{r['context_len']:>8} {r['depth']:>6.1f} {std:>10} {frz:>10} {prm:>10} {r['avg_spikes']:>7.1f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"model": args.model, "config": {"near_pct": 0.10}, "results": results}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
