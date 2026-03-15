"""
Ablation study for FoveatedKV paper — measures component contribution.

Uses real K,V from model prefill to measure attention quality (cosine, MAE)
under different ablation configs vs exact fp16.

Usage:
  uv run python benchmarks/benchmark_mlx_ablation.py
"""

import argparse
import json
import math
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mipmap_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig, standard_attention_mlx
from mipmap_kv.mlx_quantize import (
    quantize_int8_per_channel, dequantize_int8_per_channel,
    quantize_int8_per_token, dequantize_int8_per_token,
    quantize_int4_per_token, dequantize_int4_per_token,
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


def uniform_quantize_attend(keys, values, query, bits=8):
    """Uniform quantization baseline (no adaptive tiers)."""
    B, H_kv, S, D = keys.shape
    if bits == 8:
        kq, ks, kz = quantize_int8_per_channel(keys)
        vq, vs, vz = quantize_int8_per_token(values)
        k_recon = dequantize_int8_per_channel(kq, ks, kz)
        v_recon = dequantize_int8_per_token(vq, vs, vz)
    else:  # INT4
        kq, ks, kz = quantize_int8_per_channel(keys)  # K stays INT8 (asymmetric)
        vq, vs, vz = quantize_int4_per_token(values)
        k_recon = dequantize_int8_per_channel(kq, ks, kz)
        v_recon = dequantize_int4_per_token(vq, vs, vz)
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

    B, H_kv, S, D = keys.shape
    H_q = H_kv  # simplified: same heads for ablation
    query = mx.random.normal((B, H_q, 1, D)).astype(keys.dtype)
    mx.eval(query)

    # Reference: exact fp16
    ref = standard_attention_mlx(query, keys, values)
    mx.eval(ref)

    configs = [
        ("Full system (5/25/70)", MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)),
        ("No foveal (uniform INT8)", None),  # special case
        ("No asymmetric (sym INT4)", "sym_int4"),  # special case
        ("No sinks (pure topk)", MLXTierConfig(foveal_pct=0.05, periph_pct=0.25, n_sinks=0, window_size=0)),
        ("Aggressive (2/18/80)", MLXTierConfig(foveal_pct=0.02, periph_pct=0.18)),
        ("Uniform INT4", "uniform_int4"),  # special case
    ]

    results = []
    print(f"\nAblation at context={S} tokens (layer {mid}, H_kv={H_kv}, D={D})")
    print(f"{'Config':<30} {'Cosine':>10} {'MAE':>12}")
    print("-" * 55)

    for name, cfg in configs:
        if cfg is None:
            # Uniform INT8 (no tiers)
            out = uniform_quantize_attend(keys, values, query, bits=8)
        elif cfg == "sym_int4":
            # Symmetric INT4 for both K and V (no asymmetric)
            kq, ks, kz = quantize_int8_per_channel(keys)
            vq, vs, vz = quantize_int4_per_token(values)
            k_recon = dequantize_int8_per_channel(kq, ks, kz)
            v_recon = dequantize_int4_per_token(vq, vs, vz)
            # Also quantize K to INT4 (symmetric = same precision for K and V)
            kq4, ks4, kz4 = quantize_int4_per_token(keys)
            k_recon_4 = dequantize_int4_per_token(kq4, ks4, kz4)
            out = standard_attention_mlx(query, k_recon_4, v_recon)
        elif cfg == "uniform_int4":
            out = uniform_quantize_attend(keys, values, query, bits=4)
        else:
            fov_cache = MLXFoveatedKVCache(cfg)
            fov_cache.update(keys, values, 0)
            fov_cache.compress()
            out = fov_cache.attend(0, query)

        mx.eval(out)
        cos = cosine_sim(out, ref)
        err = mae(out, ref)
        print(f"{name:<30} {cos:>10.6f} {err:>12.6f}")
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
        json.dump({"model": args.model, "context_len": args.context_len, "results": results}, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
