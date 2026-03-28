# Development History

> **Note**: This document is primarily historical. For current project state,
> see `docs/build_plan.md` (architecture and performance) and `docs/cleanup_plan.md`
> (claim register).

## Overview

FoveatedKV started as a CUDA/A100 project, validated quality on cloud GPUs, then
pivoted to Apple Silicon with a complete MLX native implementation including custom
Metal GPU kernels and a C++ promotion pipeline.

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

1. **Fused Split-K Metal kernel** (`kernels/foveated_attn.metal`): pre-scaled query,
   single-exp `softmax_accum`, shared `attend_fp16` helper for near+decode, LUT fp8
   decode, score-gated V loading, 4-token blocked far loop, spike detection piggybacked.
   10 kernel variants (D=64/128 x MAX_SPLITS=1/2/4/8/16) in one metallib.

2. **MLX quantization** (`mlx_quantize.py`): fp8 E4M3 per-token K, INT4 packed V.

3. **MLXFoveatedLayer / MLXFoveatedKVCache** (`mlx_foveated.py`): full cache with
   2 precision tiers + decode buffer.

4. **Disk-backed mmap archive** (`disk_archive.py`): numpy.memmap for NVMe-backed
   fp16 promotion. Separate K/V files per layer. ~50us per token read.

5. **SDPA monkey-patch for mlx-lm** (`mlx_generate.py`): intercepts
   `mx.fast.scaled_dot_product_attention`, routes decode through fused kernel.
   Minimal `_FusedSDPAState` (8 fields), C++ pipeline spike drain.

6. **Cache wrapper** for mlx-lm integration: FusedCacheWrapper.

7. **Full benchmark suite**: LongBench-Lite, needle heatmap, ablation, throughput,
   promotion quality, sustained accuracy.

## Phase 3: C++ Extension + Promotion Pipeline

Results on 8GB Mac (kernel-only, 7B shapes):

- Kernel: up to 3.34x faster at 16K vs Apple SDPA
- Quality: 0.995+ cosine, 2.02x compression, PPL ratio 0.999-1.025x
- C++ extension: nanobind FoveatedHandle with blob-packed statics
- FoveatedPrimitive: subclasses mlx::core::Primitive, precompiled metallib
- Merged kernel: Split-K + Reduce in single dispatch via shared memory
- C++ PromotionPipeline: reads fp16 from disk mmap, writes into blob
  near-tier headroom, atomic near_valid[h] commit
- C++ CompressHandle: GPU compression kernels for fp8 E4M3 K + INT4 V

## Phase 4: Performance Optimization + TurboQuant (Current)

End-to-end decode went from 3-5x SLOWER than standard to competitive on 0.5B:

- 4-bit model (Qwen2.5-0.5B-Instruct-4bit): 96-98 tok/s fused vs 107-135 standard (0.63-0.81x — slower due to Python SDPA interceptor overhead on this small model)
- bf16 model (Qwen2.5-0.5B-Instruct-bf16): 53-55 tok/s fused vs 60-66 standard (0.83-0.90x)
- On memory-constrained 7B (8GB Mac, swap-bound): foveated is 2.3x faster because standard is swap-bound

Root causes of old slowdown (all fixed):
- Dtype mismatch: fp16 cache vs bf16 model causing silent conversion overhead
- SDPA interceptor overhead from monkey-patching
- O(n) chained concatenation in decode buffer

Architecture changes:
- Direct attention module patching (`install_fused_attention`) replaced SDPA monkey-patch
- Blob passed as tracked input array (`set_input_array`) instead of raw pointer
- Lazy decode buffer concatenation (flat list + lazy concat)
- Closure-based interceptor for fallback path
- Conditional `astype` in C++ to skip no-op dtype casts

TurboQuant added as optional compression (`compress_method="turbo"`):
- 3.25-bit keys (Lloyd-Max + QJL) + 2-bit values = ~4x compression
- Python path works, Metal kernel has accuracy bug (known issue)
- New files: `turbo_constants.py`, `turbo_quantize.py`
- nanobind pinned at 2.10.2 for ABI compatibility with MLX

97 tests passing (was 73), including 24 TurboQuant tests

## What the PyTorch Code Still Does

The PyTorch reference path (`foveated.py`, `quantize.py`, `patch.py`) remains in
the repository as a correctness oracle. All MLX backends are validated against
PyTorch reference outputs. The PyTorch code also hosts the KIVI and H2O baseline
implementations used for quality comparison context.
