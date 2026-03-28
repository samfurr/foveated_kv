# FoveatedKV Claim Register

Last updated: 2026-03-28

## Goal

Every meaningful claim maps to working code, tests, and current results.

## Claim Status Labels

- `implemented` — backed by code, tests, and current results
- `partial` — core code exists, but gaps remain
- `planned` — design intent only

## Current Claim Register

| Claim | Status | Evidence |
|-------|--------|----------|
| All tokens contribute during decode; no eviction | `implemented` | `mlx_foveated.py` tiered attend path, Metal kernel |
| Two-tier mixed precision with asymmetric K/V | `implemented` | `mlx_foveated.py`, `mlx_quantize.py`, Metal kernel |
| Fused Split-K Metal kernel with register dequant | `implemented` | `kernels/foveated_attn.metal`, `foveated_attn.cpp` |
| Pre-scaled query (amortized 1/sqrt(D)) | `implemented` | Metal kernel loads q * INV_SQRT_D once |
| Single-exp online softmax with FMA | `implemented` | `softmax_accum` template in Metal kernel |
| Score-gated V loading (skip when < 1e-7) | `implemented` | `SCORE_SKIP = 16.0f` in Metal kernel |
| LUT fp8 decode (256-entry threadgroup table) | `implemented` | `e4m3_lut[256]` in Metal kernel |
| Spike detection piggybacked on kernel softmax | `implemented` | Metal kernel spike_flags/tokens output |
| Lossless promotion via NVMe disk archive | `implemented` | `disk_archive.py`, 8 tests |
| C++ promotion pipeline with near-tier headroom | `implemented` | `promotion_pipeline.cpp`, blob write + atomic near_valid |
| Direct attention module patching (install_fused_attention) | `implemented` | `mlx_generate.py`, FusedCacheWrapper |
| MLX quantization (fp8 E4M3 + INT4) | `implemented` | `mlx_quantize.py`, tests in `test_mlx_foveated.py` |
| C++ GPU compression kernels | `implemented` | `foveated_compress.cpp`, `foveated_compress.metal` |
| 2.02x memory compression | `implemented` | `benchmark_mlx_throughput.py` results |
| LongBench-Lite 15.1 vs 14.9 standard | `implemented` | `benchmark_mlx_longbench.py` |
| 100% needle retrieval (55/55) | `implemented` | `benchmark_mlx_needle_heatmap.py` results |
| Non-accumulating PPL (0.999-1.025x) | `implemented` | `benchmark_mlx_ablation.py` results |
| Kernel up to 3.34x at 16K (7B shapes) | `implemented` | `benchmark_mlx_throughput.py` results |
| End-to-end decode 0.63-0.81x on 4-bit 0.5B (slower due to interceptor overhead) | `implemented` | `benchmark_mlx_model.py` results |
| End-to-end decode 0.83-0.90x on bf16 0.5B (slower due to interceptor overhead) | `implemented` | `benchmark_mlx_model.py` results |
| End-to-end 2.3x faster on 7B 8GB Mac (swap-bound baseline) | `implemented` | `benchmark_mlx_model.py` results |
| TurboQuant ~4x compression (Python path) | `implemented` | `turbo_quantize.py`, 24 tests |
| TurboQuant Metal kernel | `partial` | Known accuracy bug in Metal path |
| Asymmetric K/V is critical (3.6x ablation) | `implemented` | `benchmark_mlx_ablation.py` results |
| PyTorch reference path for validation | `implemented` | `foveated.py`, `quantize.py`, `patch.py` |
| KIVI baseline (faithful reimplementation) | `implemented` | `baselines.py`, PyTorch-only |
| H2O baseline (faithful reimplementation) | `implemented` | `baselines.py`, PyTorch-only |
| LongBench scoring matches THUDM v1 | `implemented` | 29 tests in `test_longbench_scoring.py` |
| Scaling to larger models (7B+) | `planned` | Architecture supports it; needs higher-end hardware |
| Longer context (16K-32K real models) | `planned` | Kernel supports it; needs memory for model + cache |
| nanobind pinned at 2.10.2 | `implemented` | `pyproject.toml`, ABI compatibility with MLX |
| Paper write-up | `partial` | Draft exists in `paper/` directory |

## Working Rules

1. No benchmark claims from stale artifacts.
2. No claim may be stronger than its status label.
3. PyTorch reference behavior is the source of truth for correctness.
4. MLX results must be reproducible on the stated hardware.
5. Baselines are PyTorch-only and cited for comparison context, not direct MLX comparison.

## 97 Tests Passing

- 26 MLX tests (`test_mlx_foveated.py`)
- 8 disk archive tests (`test_disk_archive.py`)
- 29 scoring tests (`test_longbench_scoring.py`)
- 24 TurboQuant tests
- 10 other tests

## File Inventory

### MLX Production (all implemented)

| File | Purpose | Tests |
|------|---------|-------|
| `mlx_foveated.py` | 2-tier KV cache + decode buffer | 26 in test_mlx_foveated.py |
| `mlx_quantize.py` | fp8 E4M3 K + INT4 V quantization | covered in test_mlx_foveated.py |
| `metal_foveated.py` | Python Metal kernel (fallback) | benchmark_mlx.py |
| `mlx_generate.py` | Direct attention module patching + C++ pipeline drain | integration coverage |
| `disk_archive.py` | NVMe mmap archive | 8 in test_disk_archive.py |
| `turbo_quantize.py` | TurboQuant: 3.25-bit K + 2-bit V (~4x compression) | 24 TurboQuant tests |
| `turbo_constants.py` | Pre-computed Lloyd-Max codebooks for TurboQuant | covered in TurboQuant tests |

### C++ Extension

| File | Purpose |
|------|---------|
| `foveated_attn.h/.cpp` | FoveatedPrimitive + FoveatedHandle |
| `promotion_pipeline.h/.cpp` | C++ promotion worker (near-tier headroom) |
| `foveated_compress.h/.cpp` | GPU compression kernels |
| `kernels/foveated_attn.metal` | Fused Split-K attention kernel |
| `kernels/foveated_compress.metal` | Compression Metal kernels |
| `bindings.cpp` | nanobind bindings |
| `CMakeLists.txt` | Build system |

### Benchmarks (all produce results)

| File | What It Measures |
|------|-----------------|
| `benchmark_mlx_longbench.py` | LongBench-Lite quality |
| `benchmark_mlx_needle_heatmap.py` | Needle retrieval grid |
| `benchmark_mlx_ablation.py` | Component ablation |
| `benchmark_mlx_throughput.py` | Kernel throughput + memory |
| `benchmark_mlx.py` | Synthetic kernel speed |
| `benchmark_mlx_model.py` | End-to-end model inference |
| `benchmark_promotion_quality.py` | Passkey retrieval with C++ promotion |
| `benchmark_mlx_sustained.py` | Sustained accuracy over long generation |
| `benchmark_crossover.py` | Kernel vs end-to-end crossover |
