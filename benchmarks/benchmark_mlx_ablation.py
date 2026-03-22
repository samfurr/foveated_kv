"""
Ablation study for FoveatedKV paper — measures component contribution.

Uses real K,V from model prefill to measure attention quality (cosine, MAE)
under different ablation configs vs exact fp16.

Ablation axes:
  1. Full system (10% near fp16 + 90% far fp8 K / int4 V)
  2. No near tier: uniform fp8 K + int4 V for all tokens
  3. Symmetric quant: int4 K + int4 V (removing K-precision advantage)
  4. No sinks/window: topk-only near selection
  5. Uniform INT8: INT8 K + INT8 V for all tokens (no tiers)
  6. No far quantization: 10% near fp16, 90% far fp16 (tiers but no quant)

Usage:
  uv run python benchmarks/benchmark_mlx_ablation.py
  uv run python benchmarks/benchmark_mlx_ablation.py --context-len 8192
"""

import argparse
import json
import os
import sys

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import (
    MLXFoveatedKVCache,
    MLXTierConfig,
    standard_attention_mlx,
    _fp16_to_e4m3,
    _e4m3_to_fp16,
    _quantize_int4_per_token,
    _dequant_int4_per_token,
)
from foveated_kv.mlx_quantize import (
    quantize_int8_per_channel,
    dequantize_int8_per_channel,
    quantize_int8_per_token,
    dequantize_int8_per_token,
    quantize_int4_per_token,
    dequantize_int4_per_token,
)


def cosine_sim(a, b):
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    dot = mx.sum(a_f * b_f)
    na = mx.sqrt(mx.sum(a_f * a_f))
    nb = mx.sqrt(mx.sum(b_f * b_f))
    r = dot / (na * nb + 1e-8)
    mx.eval(r)
    return r.item()


def mae(a, b):
    r = mx.mean(mx.abs(a.astype(mx.float32) - b.astype(mx.float32)))
    mx.eval(r)
    return r.item()


def uniform_fp8_int4_attend(keys, values, query):
    """All tokens quantized: fp8 E4M3 K + int4 per-token V (no near tier)."""
    k_fp8 = _fp16_to_e4m3(keys)
    k_recon = _e4m3_to_fp16(k_fp8)
    v_packed, v_scale, v_zero = _quantize_int4_per_token(values)
    v_recon = _dequant_int4_per_token(v_packed, v_scale, v_zero)
    return standard_attention_mlx(query, k_recon, v_recon)


def symmetric_int4_attend(keys, values, query):
    """Symmetric: int4 K + int4 V (removes asymmetric K-precision advantage)."""
    kq, ks, kz = quantize_int4_per_token(keys)
    vq, vs, vz = quantize_int4_per_token(values)
    k_recon = dequantize_int4_per_token(kq, ks, kz)
    v_recon = dequantize_int4_per_token(vq, vs, vz)
    return standard_attention_mlx(query, k_recon, v_recon)


def uniform_int8_attend(keys, values, query):
    """Uniform INT8 K (per-channel) + INT8 V (per-token), no tiers."""
    kq, ks, kz = quantize_int8_per_channel(keys)
    vq, vs, vz = quantize_int8_per_token(values)
    k_recon = dequantize_int8_per_channel(kq, ks, kz)
    v_recon = dequantize_int8_per_token(vq, vs, vz)
    return standard_attention_mlx(query, k_recon, v_recon)


def run_ablation(model, tokenizer, context_len: int = 4096):
    """Run ablation on real model K,V."""
    from mlx_lm.models.cache import make_prompt_cache

    # Get real K,V from model prefill
    filler = "This document discusses various topics in science and technology. " * 5
    tokens = mx.array(tokenizer.encode(filler * 200)[:context_len]).reshape(1, -1)

    cache = make_prompt_cache(model)
    model(tokens, cache=cache)

    # Use middle layer for representative results
    mid = len(cache) // 2
    keys, values = cache[mid].state
    mx.eval(keys, values)

    # Convert to fp16 for consistent comparisons — the foveated pipeline
    # and _fp16_to_e4m3 / _quantize_int4_per_token all operate on fp16.
    keys = keys.astype(mx.float16)
    values = values.astype(mx.float16)
    mx.eval(keys, values)

    B, H_kv, S, D = keys.shape
    H_q = H_kv  # simplified: same heads for ablation
    query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    mx.eval(query)

    # Reference: exact fp16 attention
    ref = standard_attention_mlx(query, keys, values)
    mx.eval(ref)

    # Ablation configs: (name, handler)
    # handler is either an MLXTierConfig (run through foveated pipeline)
    # or a callable (keys, values, query) -> output
    configs = [
        ("Full system (10/90)",           MLXTierConfig()),
        ("No near (uniform fp8+int4)",    uniform_fp8_int4_attend),
        ("Symmetric (int4 K + int4 V)",   symmetric_int4_attend),
        ("No sinks (pure topk near)",     MLXTierConfig(n_sinks=0, window_size=0)),
        ("Uniform INT8 (no tiers)",       uniform_int8_attend),
        ("Tiers, no quant (fp16 far)",    MLXTierConfig(near_pct=0.10)),
    ]

    results = []
    print(f"\nAblation at context={S} tokens (layer {mid}, H_kv={H_kv}, D={D})")
    print(f"{'Config':<35} {'Cosine':>10} {'MAE':>12} {'vs Full':>10}")
    print("-" * 70)

    full_cos = None
    for name, handler in configs:
        if isinstance(handler, MLXTierConfig):
            if name.startswith("Tiers, no quant"):
                # Use foveated tier selection but keep far at fp16
                # by compressing then reconstructing from archive
                fov_cache = MLXFoveatedKVCache(handler)
                fov_cache.update(keys, values, 0)
                fov_cache.compress()
                layer = fov_cache.layers[0]
                # Attend using archive (fp16) for far tier
                near_k = layer.near_k
                near_v = layer.near_v
                near_valid = layer.near_valid
                arch_k = layer.archive_k
                arch_v = layer.archive_v
                # Reconstruct full fp16 KV: near + archive (far in fp16)
                nv = int(mx.max(near_valid).item())
                full_k = mx.concatenate([near_k[:, :, :nv], arch_k], axis=2)
                full_v = mx.concatenate([near_v[:, :, :nv], arch_v], axis=2)
                mx.eval(full_k, full_v)
                out = standard_attention_mlx(query, full_k, full_v)
            else:
                fov_cache = MLXFoveatedKVCache(handler)
                fov_cache.update(keys, values, 0)
                fov_cache.compress()
                out = fov_cache.attend(0, query)
        else:
            out = handler(keys, values, query)

        mx.eval(out)
        cos = cosine_sim(out, ref)
        err = mae(out, ref)

        if full_cos is None:
            full_cos = cos
            delta = "baseline"
        else:
            diff = cos - full_cos
            delta = f"{diff:+.6f}"

        print(f"{name:<35} {cos:>10.6f} {err:>12.6f} {delta:>10}")
        results.append({"config": name, "cosine": cos, "mae": err})

    return results


def main():
    parser = argparse.ArgumentParser(description="FoveatedKV Ablation Study")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--context-len", type=int, default=4096)
    parser.add_argument("--output", default="results/paper/ablation.json")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    results = run_ablation(model, tokenizer, args.context_len)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(
            {"model": args.model, "context_len": args.context_len, "results": results},
            f, indent=2,
        )
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
