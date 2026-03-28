"""Profile where time goes in a single fused decode step vs standard."""

import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import mlx.core as mx
from mlx_lm import load
from foveated_kv.mlx_foveated import MLXFoveatedKVCache, MLXTierConfig
from foveated_kv.mlx_generate import (
    FusedCacheWrapper, install_fused_attention,
    prefill_and_compress, drain_spikes, uninstall_fused_attention,
)

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
cfg = MLXTierConfig()
tok = mx.array([[1]])
base_text = "The quick brown fox jumps over the lazy dog. " * 200
tokens = mx.array(tokenizer.encode(base_text)[:512]).reshape(1, -1)
N = 30

# === Standard ===
from mlx_lm.models.cache import make_prompt_cache
std_cache = make_prompt_cache(model)
logits = model(tokens, cache=std_cache)
mx.eval(logits)
for _ in range(5):
    out = model(tok, cache=std_cache)
    mx.eval(out)

t0 = time.perf_counter()
for _ in range(N):
    out = model(tok, cache=std_cache)
    mx.eval(out)
std_ms = (time.perf_counter() - t0) / N * 1000
del std_cache

# === Fused ===
fov, _, _ = prefill_and_compress(model, tokens, cfg=cfg)
wr = install_fused_attention(model, fov)

for _ in range(5):
    drain_spikes(wr, None, 0)
    out = model(tok, cache=wr)
    mx.eval(out)

t0 = time.perf_counter()
for _ in range(N):
    drain_spikes(wr, None, 0)
    out = model(tok, cache=wr)
    mx.eval(out)
fused_ms = (time.perf_counter() - t0) / N * 1000

# Now profile the interceptor vs kernel dispatch
# Time just the 24 kernel dispatches (no model overhead)
layer0 = fov.layers[0]
query = mx.random.normal((1, 14, 1, 64)).astype(mx.float16)
mx.eval(query)

# Single layer kernel dispatch
for _ in range(10):
    out = layer0.attend_fused_with_spikes(query)
    mx.eval(*[x for x in out if x is not None])

t0 = time.perf_counter()
for _ in range(N):
    out = layer0.attend_fused_with_spikes(query)
    mx.eval(*[x for x in out if x is not None])
kernel_ms = (time.perf_counter() - t0) / N * 1000

# 24 layer kernel dispatches
for _ in range(5):
    results = [l.attend_fused_with_spikes(query) for l in fov.layers if l]
    mx.eval(*[x for r in results for x in r if x is not None])

t0 = time.perf_counter()
for _ in range(N):
    results = [l.attend_fused_with_spikes(query) for l in fov.layers if l]
    mx.eval(*[x for r in results for x in r if x is not None])
all_kernels_ms = (time.perf_counter() - t0) / N * 1000

uninstall_fused_attention(model)

# === Standard SDPA for same shapes ===
from foveated_kv.mlx_foveated import standard_attention_mlx
keys = mx.random.normal((1, 2, 512, 64)).astype(mx.float16)
values = mx.random.normal((1, 2, 512, 64)).astype(mx.float16)
q = mx.random.normal((1, 14, 1, 64)).astype(mx.float16)
mx.eval(keys, values, q)

for _ in range(10):
    o = standard_attention_mlx(q, keys, values)
    mx.eval(o)

t0 = time.perf_counter()
for _ in range(N):
    o = standard_attention_mlx(q, keys, values)
    mx.eval(o)
sdpa_ms = (time.perf_counter() - t0) / N * 1000

print(f"\n{'='*50}")
print(f"DECODE STEP PROFILING (S=512, Qwen2.5-0.5B)")
print(f"{'='*50}")
print(f"Standard full step:      {std_ms:.1f}ms")
print(f"Fused full step:         {fused_ms:.1f}ms  ({fused_ms/std_ms:.1f}x)")
print(f"")
print(f"Single kernel dispatch:  {kernel_ms:.2f}ms")
print(f"24x kernel dispatch:     {all_kernels_ms:.1f}ms")
print(f"Raw SDPA (same shapes):  {sdpa_ms:.2f}ms")
print(f"")
print(f"Model overhead (fused):  {fused_ms - all_kernels_ms:.1f}ms")
print(f"  (= FFN + embedding + interceptor + cache update)")
print(f"Model overhead (std):    {std_ms:.1f}ms (everything)")
