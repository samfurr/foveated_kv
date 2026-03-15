# FoveatedKV Build Status

## Status: MLX Native Implementation Complete

### MLX Production Path (implemented)

```
src/mipmap_kv/
  mlx_foveated.py        -- MLXFoveatedKVCache, MLXFoveatedLayer, MLXTierConfig
                            3 precision tiers + decode buffer, per-head assignment
  mlx_quantize.py        -- INT8 per-channel/per-token + INT4 packed (MLX native)
                            Matches PyTorch reference exactly
  metal_foveated.py      -- Fused Split-K Metal kernel
                            Register-only K dequant, online softmax across all tiers
                            Spike detection piggybacked, compile-time N_FOV
  mlx_async_promoter.py  -- 2 background workers (spike processing + disk reads)
                            Fire-and-forget spike handoff, O(1) drain by layer
                            numpy-only workers (no MLX in threads)
  mlx_generate.py        -- SDPA monkey-patch for mlx-lm
                            Intercepts mx.fast.scaled_dot_product_attention
                            FusedCacheWrapper, generate_fused, prefill_and_compress
  disk_archive.py        -- NVMe-backed numpy.memmap fp16 archive
                            One file per layer, ~50us/token read
```

### PyTorch Reference Path (maintained for validation)

```
src/mipmap_kv/
  foveated.py            -- FoveatedKVCache, FoveatedLayer, TierConfig
                            Golden standard for correctness validation
  quantize.py            -- INT8 per-channel/per-token + INT4 packed (PyTorch)
  patch.py               -- SDPA interception for HuggingFace models
  async_manager.py       -- Central async tier manager (PyTorch reference)
```

### Removed (from earlier CUDA/A100 phase)

- `csrc/` directory (CUDA kernels)
- `setup_cuda.py` (CUDA build script)
- Triton kernel (`triton_foveated.py`)
- FlashInfer integration
- Cloud deployment scripts (vast.ai)

### Benchmarks

```
benchmarks/
  benchmark_mlx.py               -- Synthetic Metal kernel speed
  benchmark_mlx_longbench.py     -- LongBench-Lite (6 tasks)
  benchmark_mlx_needle_heatmap.py -- Needle retrieval heatmap (depth x context)
  benchmark_mlx_ablation.py      -- Component ablation study
  benchmark_mlx_throughput.py    -- Kernel throughput measurement
  benchmark_mlx_model.py         -- End-to-end model benchmark
  baselines.py                   -- KIVI + H2O (PyTorch, for comparison context)
  benchmark_foveated.py          -- PyTorch reference benchmark
```

### Tests (34 passing)

```
tests/
  test_mlx_foveated.py     -- 26 MLX tests: cache, quantize, tiers, attend, promote
  test_disk_archive.py      -- 8 tests: create, read, roundtrip, layer isolation
  test_foveated.py          -- PyTorch reference tests
  test_async_manager.py     -- Async tier manager tests
  test_patch.py             -- SDPA interception tests
  test_baselines.py         -- KIVI/H2O baseline tests
  test_longbench_scoring.py -- 29 scoring function tests
  test_backend_parity.py    -- Backend parity against golden fixtures
  backend_fixtures.py       -- Fixture generation utilities
```

### Scripts

```
scripts/
  run_mlx_benchmarks.sh    -- Run all MLX benchmarks
  generate_charts.py       -- Paper figures from JSON results
```

## Results (Qwen2.5-0.5B-Instruct-bf16, 8GB Mac)

| Metric | Result |
|--------|--------|
| LongBench-Lite | 9.7 avg (vs 9.8 standard) |
| Needle retrieval | 36/36 (100%) |
| PPL ratio (1K) | 0.998x |
| PPL ratio (2K) | 0.993x |
| PPL ratio (4K) | 1.003x |
| Memory compression | 2.21x |
| Kernel speed (4K, 7B shapes) | 1.49x |
| Kernel speed (8K, 7B shapes) | 1.46x |
| Kernel speed (32K, original) | 3.28x |
| Ablation: asymmetric K/V | 130x error without it |

## Kernel Latency (7B shapes: H_kv=4, H_q=16, D=128, single layer)

| Context | fp16 SDPA | Fused Kernel | Speedup |
|---------|-----------|-------------|---------|
| 512 | 364ms | 173ms | 2.11x |
| 1,024 | 340ms | 227ms | 1.50x |
| 2,048 | 457ms | 261ms | 1.75x |
| 4,096 | 400ms | 236ms | 1.70x |
| 8,192 | 217ms | 202ms | 1.08x |
| 16,384 | 311ms | 198ms | 1.57x |

Fused kernel beats standard fp16 SDPA at ALL context lengths in isolation.
End-to-end decode with a real model adds ~100ms/step Python interceptor overhead
(24 layers × Python function calls + input validation). A C++ extension would
eliminate this — blocked on MLX not shipping nanobind type caster headers.

## Promotion Override Buffer

The async promotion system uses a shared-memory override buffer:
- Background worker writes promoted fp16 K,V into double-buffered numpy arrays
- Metal kernel reads overrides via a pre-sorted merge-scan (O(N_FAR + MAX_OV))
- Spike detection is a free byproduct of the kernel's online softmax
- All 24 layers' spikes batched into one mx.eval per decode step

## Recency-Based Compression

Tier assignment uses pure position (no scoring):
- Sinks (first N) + recent window (last N) → foveal (fp16)
- Next most recent middle → peripheral (INT8)
- Oldest middle → far (INT8 K + INT4 V)
- Deterministic — no argpartition non-determinism

Promotion via override buffer handles any important early tokens that
land in the far tier.

## What Is Planned

- C++ MLX extension to eliminate Python interceptor overhead (~100ms/step)
- Scale to larger models (7B+) on higher-end Apple Silicon
- Formal paper write-up

## Kernel Precision Fix

The Metal Split-K kernel now produces byte-identical output to the reference
dequant+SDPA path. The fix: `to_fp16()` rounding on dequantized K,V values
in the kernel, matching the reference path's fp16 precision exactly.

Without this, float32 dequant introduced ~1 ULP fp16 differences per layer
that compounded through 24-layer autoregressive decoding (0.999999 cosine
per layer × 24 layers × multiple steps → generation divergence after 5 tokens).

The kernel still uses float32 for all accumulation (online softmax, V weighting)
for numerical stability. Only the dequantized input values are rounded to fp16.
