"""Key precision ablation: cosine error vs key quantization format.

Generates Table (key_ablation) in the paper.

Usage:
  uv run python paper/scripts/key_precision_ablation.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from foveated_kv.mlx_foveated import standard_attention_mlx
from foveated_kv.mlx_quantize import (
    quantize_fp8_e4m3, dequantize_fp8_e4m3,
    quantize_int4_packed, dequantize_int4_packed,
)


def cosine(a, b):
    a_f = a.reshape(-1).astype(mx.float32)
    b_f = b.reshape(-1).astype(mx.float32)
    c = mx.sum(a_f * b_f) / (mx.sqrt(mx.sum(a_f * a_f)) * mx.sqrt(mx.sum(b_f * b_f)) + 1e-8)
    mx.eval(c)
    return c.item()


def quantize_int8(x):
    """Per-token INT8 quantization."""
    x_f = x.astype(mx.float32)
    amax = mx.max(mx.abs(x_f), axis=-1, keepdims=True)
    scale = amax / 127.0
    q = mx.round(x_f / (scale + 1e-8)).astype(mx.int8)
    return q, scale


def dequantize_int8(q, scale):
    return q.astype(mx.float32) * scale


def main():
    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")

    filler = "The quick brown fox jumps over the lazy dog. " * 500
    tokens = mx.array(tokenizer.encode(filler)[:4096]).reshape(1, -1)
    print(f"Context: {tokens.shape[1]} tokens")

    std_cache = make_prompt_cache(model)
    logits = model(tokens, cache=std_cache)
    mx.eval(logits)

    k0, v0 = std_cache[0].state
    query = mx.random.normal((1, 8, 1, 64)).astype(mx.float16)
    mx.eval(query)
    exact_out = standard_attention_mlx(query, k0, v0)
    mx.eval(exact_out)

    # Quantize values to INT4 (held constant)
    v_packed, v_scales, v_zeros = quantize_int4_packed(v0)

    print(f"{'Key Format':>20} {'Cosine':>12} {'Error':>12} {'vs fp8':>10}")
    print("-" * 58)

    results = {}

    # fp16 keys (exact)
    v_deq = dequantize_int4_packed(v_packed, v_scales, v_zeros, v0.shape[-1])
    out = standard_attention_mlx(query, k0, v_deq)
    mx.eval(out)
    cos = cosine(out, exact_out)
    results["fp16"] = 1 - cos
    print(f"{'fp16 (exact)':>20} {cos:>12.6f} {1-cos:>12.6f} {'---':>10}")

    # INT8 keys
    k_q8, k_s8 = quantize_int8(k0)
    k_deq8 = dequantize_int8(k_q8, k_s8).astype(mx.float16)
    mx.eval(k_deq8)
    out = standard_attention_mlx(query, k_deq8, v_deq)
    mx.eval(out)
    cos = cosine(out, exact_out)
    results["INT8"] = 1 - cos
    print(f"{'INT8 per-token':>20} {cos:>12.6f} {1-cos:>12.6f}")

    # fp8 E4M3 keys
    k_fp8, k_s_fp8 = quantize_fp8_e4m3(k0)
    k_deq_fp8 = dequantize_fp8_e4m3(k_fp8, k_s_fp8)
    mx.eval(k_deq_fp8)
    out = standard_attention_mlx(query, k_deq_fp8, v_deq)
    mx.eval(out)
    cos = cosine(out, exact_out)
    results["fp8"] = 1 - cos
    fp8_err = 1 - cos
    print(f"{'fp8 E4M3 (ours)':>20} {cos:>12.6f} {1-cos:>12.6f} {'1.0x':>10}")

    # INT4 keys
    k_packed, k_scales, k_zeros = quantize_int4_packed(k0)
    k_deq4 = dequantize_int4_packed(k_packed, k_scales, k_zeros, k0.shape[-1])
    mx.eval(k_deq4)
    out = standard_attention_mlx(query, k_deq4, v_deq)
    mx.eval(out)
    cos = cosine(out, exact_out)
    results["INT4"] = 1 - cos
    ratio = (1 - cos) / max(fp8_err, 1e-10)
    print(f"{'INT4 per-token':>20} {cos:>12.6f} {1-cos:>12.6f} {ratio:>9.1f}x")


if __name__ == "__main__":
    main()
