"""
Definitive bottleneck profiler: verify overhead fixes.

Tests the two key fixes:
  1. Cached zero override arrays (was 151ms → should be ~0ms)
  2. Pre-allocated decode buffer (was 215ms+ → should be ~0ms)
  3. Full model decode: standard vs fused end-to-end
"""

import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from foveated_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*[x for x in out if x is not None])
        else:
            mx.eval(out)
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        if isinstance(out, (list, tuple)):
            mx.eval(*[x for x in out if x is not None])
        else:
            mx.eval(out)
    return (time.perf_counter() - t0) / iters * 1000


def test_overhead_after_fix():
    """Verify overhead sources are eliminated."""
    print("=" * 65)
    print("OVERHEAD VERIFICATION (after fixes)")
    print("=" * 65)

    B, H_kv, H_q, D = 1, 2, 14, 64
    cfg = MLXTierConfig()

    # Build a foveated layer to test dispatch
    S = 1000
    keys = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
    values = mx.random.normal((B, H_kv, S, D)).astype(mx.float16)
    mx.eval(keys, values)

    cache = MLXFoveatedKVCache(cfg)
    cache.update(keys, values, 0)
    cache.compress()
    layer = cache.layers[0]

    # Verify decode buffer state
    print(f"  Decode buffer type: {type(layer._decode_k_buf).__name__}")
    print(f"  Decode buffer length: {len(layer._decode_k_buf)}")
    print(f"  Decode K cached: {layer._decode_k_cached is not None}")

    # Warm up kernel cache
    query = mx.random.normal((B, H_q, 1, D)).astype(mx.float16)
    mx.eval(query)
    out = layer.attend_fused_with_spikes(query)
    mx.eval(*[x for x in out if x is not None])

    # Test: single kernel call
    single_ms = bench(lambda: layer.attend_fused_with_spikes(query), iters=50)
    print(f"\n  Single kernel call: {single_ms:.2f}ms")

    # Test: 24 kernel calls (simulating one decode step)
    batch_ms = bench(
        lambda: [layer.attend_fused_with_spikes(query) for _ in range(24)],
        iters=30,
    )
    print(f"  24 kernel calls:   {batch_ms:.2f}ms")

    # Test: add tokens then dispatch (simulating decode steps 1-10)
    for n_decode in [0, 5, 10, 20]:
        layer2 = cache.layers[0]  # fresh layer
        new_k = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        new_v = mx.random.normal((B, H_kv, 1, D)).astype(mx.float16)
        mx.eval(new_k, new_v)
        for _ in range(n_decode):
            layer2.add_token(new_k, new_v)
        # Warm kernel cache
        out = layer2.attend_fused_with_spikes(query)
        mx.eval(*[x for x in out if x is not None])

        ms = bench(lambda: layer2.attend_fused_with_spikes(query), iters=30)
        print(f"  1 kernel + {n_decode:>2} decode tokens: {ms:.2f}ms")

    del keys, values, cache, layer


def test_full_model():
    """Full model decode: standard vs fused."""
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache

    from foveated_kv.mlx_generate import (
        FusedCacheWrapper,
        install_fused_attention,
        prefill_and_compress,
        drain_spikes,
        uninstall_fused_attention,
    )

    print("\n" + "=" * 65)
    print("FULL MODEL DECODE (Qwen2.5-0.5B, 24 layers)")
    print("=" * 65)

    model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    cfg = MLXTierConfig()
    tok = mx.array([[1]])
    base_text = "The quick brown fox jumps over the lazy dog. " * 200

    for S in [512, 1000]:
        tokens = mx.array(tokenizer.encode(base_text)[:S]).reshape(1, -1)
        actual_s = tokens.shape[1]

        # === Standard decode ===
        std_cache = make_prompt_cache(model)
        logits = model(tokens, cache=std_cache)
        mx.eval(logits)
        for _ in range(5):
            out = model(tok, cache=std_cache)
            mx.eval(out)
        times_std = []
        for _ in range(20):
            t0 = time.perf_counter()
            out = model(tok, cache=std_cache)
            mx.eval(out)
            times_std.append((time.perf_counter() - t0) * 1000)
        std_ms = sum(sorted(times_std)[3:-3]) / max(len(times_std) - 6, 1)
        del std_cache

        # === Fused decode ===
        fov, pl, _ = prefill_and_compress(model, tokens, cfg=cfg)
        wr = install_fused_attention(model, fov)
        for _ in range(5):
            drain_spikes(wr, None, 0)
            out = model(tok, cache=wr)
            mx.eval(out)
        times_fused = []
        for _ in range(20):
            drain_spikes(wr, None, 0)
            t0 = time.perf_counter()
            out = model(tok, cache=wr)
            mx.eval(out)
            times_fused.append((time.perf_counter() - t0) * 1000)
        fused_ms = sum(sorted(times_fused)[3:-3]) / max(len(times_fused) - 6, 1)
        uninstall_fused_attention(model)

        ratio = fused_ms / std_ms
        print(f"  S={actual_s:>5}: standard={std_ms:.1f}ms  fused={fused_ms:.1f}ms  ratio={ratio:.2f}x")

        # Show raw distribution
        print(f"           std: {[f'{t:.0f}' for t in sorted(times_std)]}")
        print(f"         fused: {[f'{t:.0f}' for t in sorted(times_fused)]}")


if __name__ == "__main__":
    test_overhead_after_fix()
    test_full_model()
