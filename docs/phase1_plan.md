# Development History

## Overview

FoveatedKV started as a CUDA/A100 project, validated quality on cloud GPUs, then
pivoted to Apple Silicon with a complete MLX native implementation including custom
Metal GPU kernels.

## Phase 1: Prototype and Validate (PyTorch, T4/A100)

Initial development on free T4 hardware and rented A100 instances.

- Built PyTorch reference implementation: `FoveatedKVCache`, `FoveatedLayer`, quantization
- Validated core quality hypothesis at 65K context on Qwen2.5-7B (A100-80GB):
  - 1.5x less attention error than KIVI (uniform INT8) at 28% less bandwidth
  - 100% passkey retrieval across all depths and contexts
  - PPL identical (1.14 standard vs 1.14 foveated)
  - 2.2-2.4x GPU KV compression
  - Asymmetric K/V confirmed as key component (130x error without it)
- Built Triton kernel (mixed-precision attention + spike detection)
- Built CPU coprocessor for async tier management
- Implemented KIVI and H2O baselines with faithful behavior
- Aligned LongBench scoring with official THUDM v1 pipeline

**Key lesson:** Triton kernel was ~10x slower than Flash Attention 2 on A100 for
single-query decode. Element-wise ops cannot compete with tensor cores when M=1.
This motivated the pivot to a platform where custom kernels could win.

## Phase 2: Pivot to Apple Silicon (MLX + Metal)

Rebuilt the entire system natively on MLX with custom Metal GPU kernels.

**What was removed:**
- All CUDA code (csrc/ directory)
- Triton kernel
- FlashInfer integration
- setup_cuda.py build script
- Cloud deployment scripts (vast.ai references)

**What was built:**

1. **Fused Split-K Metal kernel** (`metal_foveated.py`): register-only K dequant,
   online softmax across all tiers + decode buffer, spike detection piggybacked.
   Compile-time N_FOV for loop unrolling.

2. **MLX quantization** (`mlx_quantize.py`): INT8 per-channel/per-token, INT4 packed.
   Matches PyTorch reference exactly.

3. **MLXFoveatedLayer / MLXFoveatedKVCache** (`mlx_foveated.py`): full cache with
   3 precision tiers + decode buffer.

4. **Disk-backed mmap archive** (`disk_archive.py`): numpy.memmap for NVMe-backed
   fp16 promotion. One file per layer. ~50us per token read.

5. **Async promotion system** (`mlx_async_promoter.py`): 2 background workers
   (spike processing + disk reads). Fire-and-forget spike handoff. O(1) drain.

6. **SDPA monkey-patch for mlx-lm** (`mlx_generate.py`): intercepts
   `mx.fast.scaled_dot_product_attention`, routes decode through fused kernel.

7. **Cache wrapper** for mlx-lm integration: FusedCacheWrapper.

8. **Full benchmark suite**: LongBench-Lite, needle heatmap, ablation, throughput.

## Phase 3: Optimization (Current)

Results on 8GB Mac (kernel-only, 7B shapes, 100 iters):

- Kernel: 1.0x at 4K, 8.9x at 32K, 10.2x at 128K vs Apple SDPA
- Quality: 0.996+ cosine, 2.13-2.34x compression, PPL ratio 0.993-1.003x
- C++ extension: nanobind FoveatedHandle with blob-packed statics (9 inputs)
- Merged kernel: Split-K + Reduce in single dispatch via shared memory
- Override buffer: double-buffered, sorted insert, kernel-side spike detection
- 63 tests passing

## What the PyTorch Code Still Does

The PyTorch reference path (`foveated.py`, `quantize.py`, `patch.py`) remains in
the repository as a correctness oracle. All MLX backends are validated against
PyTorch reference outputs. The PyTorch code also hosts the KIVI and H2O baseline
implementations used for quality comparison context.
