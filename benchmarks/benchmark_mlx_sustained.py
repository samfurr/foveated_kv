"""
Sustained accuracy benchmark: does promotion keep quality flat over long generation?

Generates a long sequence where the topic shifts, comparing three modes:
  1. Standard fp16 — ground truth
  2. Frozen tiers — compress once, never promote
  3. With promotion — async coprocessor adapts tiers

Measures per-step logit divergence from standard. If promotion works, the
"with promotion" curve stays flat while "frozen" may drift as attention
patterns shift away from the original tier assignment.

This is the key benchmark that proves the coprocessor adds value beyond
static compression.

Usage:
  uv run python benchmarks/benchmark_mlx_sustained.py
  uv run python benchmarks/benchmark_mlx_sustained.py --context-len 4096 --gen-tokens 200
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipmap_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig
from mipmap_kv.mlx_generate import (
    FusedCacheWrapper,
    install_fused_sdpa,
    uninstall_fused_sdpa,
    reset_fused_layer_counter,
    prefill_and_compress,
    _generate_short,
)


def _log_softmax(x, axis=-1):
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


def build_topic_shift_prompt(tokenizer, context_len: int) -> str:
    """Build a prompt that discusses one topic, then asks about another.

    The first half is filler about science. A passkey is buried in it.
    The second half shifts to history. The generation prompt asks about
    the passkey — forcing the model to attend to tokens from the first
    half that are now likely in the far tier.
    """
    science = (
        "Recent advances in quantum computing have shown that superconducting "
        "qubits can maintain coherence for milliseconds. Researchers at MIT "
        "demonstrated a novel error correction code that reduces logical error "
        "rates by three orders of magnitude. The implications for cryptography "
        "and drug discovery are significant. "
    )
    history = (
        "The Roman Empire's decline was driven by multiple factors including "
        "economic instability, military overextension, and political corruption. "
        "The fall of Constantinople in 1453 marked the end of the Eastern Roman "
        "Empire and shifted trade routes toward the Atlantic. "
    )
    passkey = "The secret research code is DELTA-4827."
    retrieval = "\n\nBased on everything above, what was the secret research code? The code is: "

    # Build: science filler + passkey + more science + topic shift to history + retrieval
    sci_tokens = len(tokenizer.encode(science))
    hist_tokens = len(tokenizer.encode(history))
    passkey_tokens = len(tokenizer.encode(passkey))
    retrieval_tokens = len(tokenizer.encode(retrieval))

    available = context_len - passkey_tokens - retrieval_tokens - 50
    sci_repeats = max(1, int(available * 0.4) // sci_tokens)
    hist_repeats = max(1, int(available * 0.4) // hist_tokens)

    prompt = (
        science * sci_repeats
        + " " + passkey + " "
        + science * (sci_repeats // 4)
        + "\n\n"  # topic shift
        + history * hist_repeats
        + retrieval
    )

    # Truncate
    tokens = tokenizer.encode(prompt)[:context_len]
    return tokenizer.decode(tokens), "DELTA-4827"


def generate_with_logit_trace(
    model, tokenizer, prompt: str, max_tokens: int,
    cache_wrappers=None, prefill_logits=None, fused: bool = False,
) -> tuple[list[int], list[mx.array]]:
    """Generate tokens and collect per-step logits for comparison.

    For foveated paths: pass cache_wrappers (already prefilled via
    prefill_and_compress) and prefill_logits. Decode-only -- no prefill
    through the wrappers. Set fused=True when using FusedCacheWrapper.

    For standard: cache_wrappers=None does a normal prefill + decode.
    """
    from mlx_lm.models.cache import make_prompt_cache

    if cache_wrappers is None:
        # Standard path: prefill + decode
        tokens = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
        cache = make_prompt_cache(model)
        logits = model(tokens, cache=cache)
        mx.eval(logits)
        next_logits = logits[:, -1:, :]
    else:
        # Foveated path: prefill already done, use provided logits
        cache = cache_wrappers
        next_logits = prefill_logits[:, -1:, :] if prefill_logits is not None else None
        if next_logits is None:
            raise ValueError("Foveated path needs prefill_logits")

    generated = []
    logit_trace = []

    for step in range(max_tokens):
        mx.eval(next_logits)
        logit_trace.append(next_logits)

        next_token = mx.argmax(next_logits[:, -1, :], axis=-1)
        token_id = next_token.item()
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)

        if fused:
            reset_fused_layer_counter()

        next_input = next_token.reshape(1, 1)
        next_logits = model(next_input, cache=cache)
        next_logits = next_logits[:, -1:, :]

    return generated, logit_trace


def compute_divergence(
    ref_trace: list[mx.array], test_trace: list[mx.array]
) -> list[dict]:
    """Compute per-step divergence metrics between two logit traces."""
    n = min(len(ref_trace), len(test_trace))
    results = []

    for i in range(n):
        ref = ref_trace[i].reshape(-1).astype(mx.float32)
        test = test_trace[i].reshape(-1).astype(mx.float32)

        # Cosine similarity
        dot = mx.sum(ref * test)
        nr = mx.sqrt(mx.sum(ref * ref))
        nt = mx.sqrt(mx.sum(test * test))
        cosine = dot / (nr * nt + 1e-8)

        # KL divergence (ref || test) on softmax distributions
        ref_lp = _log_softmax(ref)
        test_lp = _log_softmax(test)
        ref_p = mx.softmax(ref)
        kl = mx.sum(ref_p * (ref_lp - test_lp))

        # Top-1 agreement
        ref_top = mx.argmax(ref)
        test_top = mx.argmax(test)
        agree = (ref_top == test_top)

        mx.eval(cosine, kl, agree)
        results.append({
            "step": i,
            "cosine": round(cosine.item(), 6),
            "kl_divergence": round(max(0, kl.item()), 6),
            "top1_agree": bool(agree.item()),
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Sustained Accuracy Benchmark")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--gen-tokens", type=int, default=200)
    parser.add_argument("--foveal-pct", type=float, default=0.05)
    parser.add_argument("--periph-pct", type=float, default=0.25)
    parser.add_argument("--output", default="results/paper/sustained_accuracy.json")
    args = parser.parse_args()

    from mlx_lm import load
    from mipmap_kv.disk_archive import offload_cache_to_disk
    from mipmap_kv.mlx_async_promoter import AsyncPromoter

    model, tokenizer = load(args.model)
    cfg = MLXTierConfig(foveal_pct=args.foveal_pct, periph_pct=args.periph_pct)

    prompt, passkey = build_topic_shift_prompt(tokenizer, args.context_len)
    actual_ctx = len(tokenizer.encode(prompt))
    print(f"Context: {actual_ctx} tokens, generating {args.gen_tokens}")
    print(f"Passkey: {passkey}")

    # === 1. Standard fp16 (ground truth) ===
    print("\n[1/3] Standard fp16...")
    t0 = time.perf_counter()
    std_tokens, std_trace = generate_with_logit_trace(
        model, tokenizer, prompt, args.gen_tokens,
    )
    std_time = time.perf_counter() - t0
    std_text = tokenizer.decode(std_tokens)
    std_found = passkey in std_text
    print(f"  {len(std_tokens)} tokens in {std_time:.1f}s | passkey: {'YES' if std_found else 'NO'}")
    print(f"  Output: {std_text[:100]}...")

    # === 2. Frozen tiers (compress once, no promotion) ===
    print("\n[2/3] Frozen tiers (no promotion)...")
    tokens_mx = mx.array(tokenizer.encode(prompt)).reshape(1, -1)
    fov_cache_frozen, frozen_prefill_logits, _ = prefill_and_compress(model, tokens_mx, cfg)
    frozen_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache_frozen.layers)
    ]
    install_fused_sdpa(fov_cache_frozen)

    t0 = time.perf_counter()
    frozen_tokens, frozen_trace = generate_with_logit_trace(
        model, tokenizer, prompt, args.gen_tokens,
        cache_wrappers=frozen_wrappers,
        prefill_logits=frozen_prefill_logits,
        fused=True,
    )
    frozen_time = time.perf_counter() - t0
    uninstall_fused_sdpa()
    frozen_text = tokenizer.decode(frozen_tokens)
    frozen_found = passkey in frozen_text
    print(f"  {len(frozen_tokens)} tokens in {frozen_time:.1f}s | passkey: {'YES' if frozen_found else 'NO'}")
    print(f"  Output: {frozen_text[:100]}...")

    # === 3. With promotion (async coprocessor) ===
    print("\n[3/3] With async promotion...")
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="foveated_sustained_")

    fov_cache_promo, promo_prefill_logits, _ = prefill_and_compress(model, tokens_mx, cfg)
    disk_archives = offload_cache_to_disk(fov_cache_promo, tmpdir)
    promoter = AsyncPromoter(fov_cache_promo, disk_archives)

    # Wire override buffers to layers so the Metal kernel can read them
    for i, layer in enumerate(fov_cache_promo.layers):
        if layer is not None:
            layer.overrides = promoter.overrides_for_layer(i)

    promo_wrappers = [
        FusedCacheWrapper(layer, i) if layer is not None else None
        for i, layer in enumerate(fov_cache_promo.layers)
    ]
    install_fused_sdpa(fov_cache_promo, promoter=promoter)
    from mipmap_kv.mlx_generate import _fused_state
    _fused_state._fused_wrappers = promo_wrappers

    t0 = time.perf_counter()
    promo_tokens, promo_trace = generate_with_logit_trace(
        model, tokenizer, prompt, args.gen_tokens,
        cache_wrappers=promo_wrappers,
        prefill_logits=promo_prefill_logits,
        fused=True,
    )
    promo_time = time.perf_counter() - t0
    uninstall_fused_sdpa()
    promo_text = tokenizer.decode(promo_tokens)
    promo_found = passkey in promo_text
    promo_stats = promoter.get_stats()
    promoter.stop()
    print(f"  {len(promo_tokens)} tokens in {promo_time:.1f}s | passkey: {'YES' if promo_found else 'NO'}")
    print(f"  Spikes: {promo_stats['spikes_detected']}, Overrides written: {promo_stats['promotions_completed']}")
    print(f"  Output: {promo_text[:100]}...")

    # Cleanup
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    # === Compute divergence curves ===
    print("\nComputing per-step divergence from standard fp16...")
    frozen_div = compute_divergence(std_trace, frozen_trace)
    promo_div = compute_divergence(std_trace, promo_trace)

    # Summary stats
    n = min(len(frozen_div), len(promo_div))
    if n > 20:
        # Compare first 20% vs last 20% to detect drift
        early = n // 5
        late_start = n - n // 5

        frozen_early_cos = sum(d["cosine"] for d in frozen_div[:early]) / early
        frozen_late_cos = sum(d["cosine"] for d in frozen_div[late_start:]) / (n - late_start)
        promo_early_cos = sum(d["cosine"] for d in promo_div[:early]) / early
        promo_late_cos = sum(d["cosine"] for d in promo_div[late_start:]) / (n - late_start)

        frozen_top1 = sum(1 for d in frozen_div if d["top1_agree"]) / n * 100
        promo_top1 = sum(1 for d in promo_div if d["top1_agree"]) / n * 100

        print(f"\n{'Metric':<30} {'Frozen':>12} {'With Promo':>12}")
        print("-" * 56)
        print(f"{'Cosine (early steps)':<30} {frozen_early_cos:>12.6f} {promo_early_cos:>12.6f}")
        print(f"{'Cosine (late steps)':<30} {frozen_late_cos:>12.6f} {promo_late_cos:>12.6f}")
        print(f"{'Cosine drift (early-late)':<30} {frozen_early_cos - frozen_late_cos:>12.6f} {promo_early_cos - promo_late_cos:>12.6f}")
        print(f"{'Top-1 token agreement %':<30} {frozen_top1:>11.1f}% {promo_top1:>11.1f}%")
        print(f"{'Passkey retrieved':<30} {'YES' if frozen_found else 'NO':>12} {'YES' if promo_found else 'NO':>12}")

    # Save results
    results = {
        "model": args.model,
        "context_len": actual_ctx,
        "gen_tokens": args.gen_tokens,
        "passkey": passkey,
        "standard": {
            "text": std_text[:500],
            "passkey_found": std_found,
            "time_s": round(std_time, 1),
        },
        "frozen": {
            "text": frozen_text[:500],
            "passkey_found": frozen_found,
            "time_s": round(frozen_time, 1),
            "divergence": frozen_div,
        },
        "promoted": {
            "text": promo_text[:500],
            "passkey_found": promo_found,
            "time_s": round(promo_time, 1),
            "divergence": promo_div,
            "stats": promo_stats,
        },
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
