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

## What Is Planned

- Scale to larger models (7B+) on higher-end Apple Silicon
- Longer context benchmarks (16K-32K with real models)
- mlx-lm integration end-to-end validation with more model families
- Formal paper write-up

## Sustained Accuracy Benchmark

Added `benchmark_mlx_sustained.py`: generates 200 tokens at 4K context, compares
per-step logit divergence between standard fp16, frozen foveated tiers, and
foveated with async promotion.

**Finding**: Frozen tiers maintain 0.9998 cosine, 100% top-1 agreement, zero drift.
Promotion degrades quality due to cumulative swap noise — needs per-head variable-
length storage to apply correctly. Filed as future work.
