# FoveatedKV Claim Register

Last updated: 2026-03-14

## Goal

Every meaningful claim maps to working code, tests, and current results.

## Claim Status Labels

- `implemented` — backed by code, tests, and current results
- `partial` — core code exists, but gaps remain
- `planned` — design intent only

## Current Claim Register

| Claim | Status | Evidence |
|-------|--------|----------|
| All tokens contribute during decode; no eviction | `implemented` | `mlx_foveated.py` tiered attend path, `metal_foveated.py` kernel |
| Three-tier mixed precision with asymmetric K/V | `implemented` | `mlx_foveated.py`, `mlx_quantize.py`, Metal kernel |
| Fused Split-K Metal kernel with register dequant | `implemented` | `metal_foveated.py`, `benchmark_mlx.py` speed results |
| Spike detection piggybacked on kernel softmax | `implemented` | `metal_foveated.py`, tested via `test_mlx_foveated.py` |
| Lossless promotion via NVMe disk archive | `implemented` | `disk_archive.py`, 8 tests in `test_disk_archive.py` |
| Async promotion with 2 background workers | `implemented` | `mlx_async_promoter.py`, fire-and-forget handoff |
| SDPA monkey-patch for mlx-lm | `implemented` | `mlx_generate.py`, cache wrappers |
| MLX quantization matches PyTorch reference | `implemented` | `mlx_quantize.py`, `test_mlx_foveated.py` |
| 2.21x memory compression | `implemented` | `benchmark_mlx_throughput.py` results |
| LongBench-Lite within 0.1 points | `implemented` | `benchmark_mlx_longbench.py`, 9.7 vs 9.8 |
| 100% needle retrieval (36/36) | `implemented` | `benchmark_mlx_needle_heatmap.py` results |
| Non-accumulating PPL (0.998x-1.003x) | `implemented` | `benchmark_mlx_ablation.py` results |
| Kernel 1.49x at 4K (7B shapes) | `implemented` | `benchmark_mlx.py` synthetic results |
| Asymmetric K/V is critical (130x ablation) | `implemented` | `benchmark_mlx_ablation.py` results |
| PyTorch reference path for validation | `implemented` | `foveated.py`, `quantize.py`, `patch.py` |
| KIVI baseline (faithful reimplementation) | `implemented` | `baselines.py`, PyTorch-only |
| H2O baseline (faithful reimplementation) | `implemented` | `baselines.py`, PyTorch-only |
| LongBench scoring matches THUDM v1 | `implemented` | 29 tests in `test_longbench_scoring.py` |
| Scaling to larger models (7B+) | `planned` | Architecture supports it; needs higher-end hardware |
| Longer context (16K-32K real models) | `planned` | Kernel supports it; needs memory for model + cache |
| Paper write-up | `planned` | Results collected; write-up not started |

## Working Rules

1. No benchmark claims from stale artifacts.
2. No claim may be stronger than its status label.
3. PyTorch reference behavior is the source of truth for correctness.
4. MLX results must be reproducible on the stated hardware.
5. Baselines are PyTorch-only and cited for comparison context, not direct MLX comparison.

## 34 Tests Passing

- 26 MLX tests (`test_mlx_foveated.py`)
- 8 disk archive tests (`test_disk_archive.py`)
- Additional PyTorch reference tests in other test files

## File Inventory

### MLX Production (all implemented)

| File | Purpose | Tests |
|------|---------|-------|
| `mlx_foveated.py` | 3-tier KV cache + decode buffer | 26 in test_mlx_foveated.py |
| `mlx_quantize.py` | INT8/INT4 quantization | covered in test_mlx_foveated.py |
| `metal_foveated.py` | Split-K Metal kernel | benchmark_mlx.py |
| `mlx_async_promoter.py` | 2-worker async promotion | integration coverage |
| `mlx_generate.py` | mlx-lm SDPA monkey-patch | integration coverage |
| `disk_archive.py` | NVMe mmap archive | 8 in test_disk_archive.py |

### Benchmarks (all produce results)

| File | What It Measures |
|------|-----------------|
| `benchmark_mlx_longbench.py` | LongBench-Lite quality |
| `benchmark_mlx_needle_heatmap.py` | Needle retrieval grid |
| `benchmark_mlx_ablation.py` | Component ablation |
| `benchmark_mlx_throughput.py` | Kernel throughput |
| `benchmark_mlx.py` | Synthetic kernel speed |
| `benchmark_mlx_model.py` | End-to-end model |

## Future Work: Per-Head Variable-Length Storage

The async promotion pipeline detects spikes correctly and reads fp16 from disk.
But applying promotions to the `(B, H, N, D)` tensor layout requires either
broadcasting to all heads (wrong — each head has its own K/V space) or swapping
within a single head (introduces noise when done frequently).

**Required refactor**: Store foveal K/V as a list of per-head tensors with
independent lengths. This allows promoting a token to one specific head without
affecting others, and growing one head's foveal without growing all heads.

**Impact**: Enables correct promotion application, unlocking adaptive tier
management for very long generation (1000+ tokens).
