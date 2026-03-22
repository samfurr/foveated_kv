# FoveatedKV Build Status

## Architecture

```
src/foveated_kv/
  mlx_foveated.py        -- MLXFoveatedKVCache, MLXFoveatedLayer, MLXTierConfig
                            2 precision tiers + decode buffer
                            Recency-based tier assignment (deterministic)
                            Kernel cache with pre-packed blob in unified memory
  mlx_quantize.py        -- fp8 E4M3 per-token K + INT4 packed V
  metal_foveated.py      -- Python Metal kernel fallback (when C++ ext not built)
  mlx_generate.py        -- SDPA monkey-patch for mlx-lm integration
                            FusedCacheWrapper, C++ pipeline spike drain
  disk_archive.py        -- NVMe-backed numpy.memmap fp16 archive

csrc/
  foveated_attn.h/.cpp   -- FoveatedPrimitive (subclasses mlx::core::Primitive)
                            FoveatedHandle (nanobind, precompiled metallib)
                            Pre-packed blob of 7 static arrays -> 1 Metal buffer
                            Cached pipeline states, zero per-call lookups
  promotion_pipeline.h/.cpp -- C++ promotion worker: reads fp16 from disk mmap,
                               writes directly into blob near-tier headroom.
                               Spike filtering: cooldown, dedup, budget, GQA.
                               One background std::thread per generation session.
  foveated_compress.h/.cpp  -- CompressHandle: GPU compression kernels for
                               fp8 E4M3 K + INT4 V quantization on Metal
  kernels/
    foveated_attn.metal   -- Merged Split-K + Reduce kernel, templated on
                             HEAD_DIM x MAX_SPLITS, function constants for
                             tier sizes. 10 kernel variants in one metallib.
                             Pre-scaled query, single-exp softmax, attend_fp16
                             helper, score-gated V loading, spike detection.
    foveated_compress.metal -- GPU compression kernels (foveal, fp8, int4)
  bindings.cpp            -- nanobind bindings (NB_DOMAIN mlx for type sharing)
  CMakeLists.txt          -- mlx_build_metallib + nanobind + Threads

benchmarks/
  benchmark_crossover.py  -- Kernel-only + end-to-end crossover measurement
  benchmark_promotion_quality.py -- Passkey retrieval with C++ promotion
  benchmark_mlx_sustained.py -- Sustained accuracy over long generation
  benchmark_mlx_model.py  -- End-to-end model inference with promotion
  benchmark_mlx.py        -- Synthetic kernel speed
  + 4 more benchmark files (longbench, needle, ablation, throughput)
```

## Kernel Performance (7B shapes, 100 iters, single layer)

| Context | fp16 SDPA | Fused Kernel | Speedup |
|---------|-----------|-------------|---------|
| 512 | 7.8ms | 7.6ms | 1.0x |
| 4K | 7.7ms | 8.0ms | 1.0x |
| 16K | 8.9ms | 8.2ms | 1.1x |
| 32K | 121ms | 15ms | 7.9x |
| 64K | 109ms | 25ms | 4.4x |
| 128K | 317ms | 46ms | 7.0x |

**Note**: Apple's SDPA hits a performance cliff at 32K+ (likely a cache
threshold or kernel path switch). Our fused kernel scales smoothly
(8ms -> 15ms -> 25ms -> 46ms). The large speedup numbers at 32K+ are
partly real bandwidth savings and partly the SDPA cliff — the exact
contribution is inconclusive. Needs testing on larger hardware with
models that exercise long context properly to separate the two effects.

## Quality

| Metric | Result |
|--------|--------|
| Cosine similarity vs fp16 | 0.996+ at all contexts |
| Memory compression | 2.02x |
| PPL ratio (1K-4K) | 0.999-1.025x |
| Needle retrieval | 55/55 (100%) |
| LongBench-Lite | 15.1 avg (vs 14.9 standard) |

## Key Design Decisions

**Merged kernel**: Split-K + Reduce combined into a single dispatch via
threadgroup shared memory. Multiple SIMD groups per threadgroup, each
processes a token range, barrier, first SIMD group reduces. Eliminates
global partial arrays and the second kernel dispatch.

**Pre-scaled query**: q *= 1/sqrt(D) at load time, amortizing HEAD_DIM
multiplies once vs one multiply per token across the entire sequence.

**Single-exp online softmax**: `softmax_accum` computes `alpha` and `w`
once each (two exp() calls total), then rescales and accumulates acc[]
in a single FMA pass. Standard online softmax redundantly recomputes
exp(score - m_new).

**Score-gated V loading**: Skip int4 dequant when score < m - 16.
exp(-16) ~ 1e-7 — tokens below this threshold contribute negligibly
to the output. Saves ALU + memory reads on cold far tokens.

**attend_fp16 helper**: Shared template for near tier and decode buffer,
both of which store fp16 K+V with identical layout. Eliminates code
duplication between the two loops.

**LUT fp8 decode**: 256-entry threadgroup memory LUT built once per
threadgroup. One LUT read vs 10+ ALU ops per fp8 element.

**Living near tier**: The C++ promotion worker writes fp16 K,V into
headroom slots in the blob and atomically increments `near_valid[h]`.
The kernel reads that count once per dispatch — promoted tokens appear
as ordinary near tokens with zero overhead, zero kernel changes.

**Adaptive split_size**: Grows with context to cap num_splits <= 16,
avoiding reduce bottleneck at long contexts while maintaining GPU
occupancy at short contexts.

**Packed inputs**: Scale+zero pairs concatenated during compression
(once, not per call). 7 static arrays packed into blob via C++ path.

**Kernel-side spike detection**: Free byproduct of the fused kernel's
online softmax (tracks max_far_score vs min_near_score). Feeds the
C++ PromotionPipeline which filters via cooldown, dedup, and budget.

**Recency-based compression**: Pure positional tier assignment — sinks +
recent window -> near, remaining by recency -> far. Deterministic, fast.
Promotion via near-tier headroom handles important early tokens.

**C++ extension**: nanobind 2.10.2 (pinned to match MLX ABI). FoveatedHandle
pre-packs 7 static arrays (near_k, near_v, far_k, far_v, far_v_scale,
far_v_zero, near_valid) into a single uint8 blob. Merged kernel with
minimal inputs per dispatch. Eliminates Python dispatch overhead.

## End-to-End Status

Kernel dispatch goes through `FoveatedPrimitive`, a direct subclass of
`mlx::core::Primitive`. The eval_gpu path is: set cached pipeline -> bind
pre-extracted blob buffer -> bind 3 dynamic inputs (query, decode_k,
decode_v) -> set_bytes for params and offsets -> dispatch_threadgroups.
No library lookups, no hash maps, no string comparisons, no contiguity
checks. Structurally identical to how MLX's own `ScaledDotProductAttention`
dispatches.

The kernel itself is precompiled into a `.metallib` at build time (no JIT).
Function constants specialize for tier sizes, enabling GPU loop optimization.
10 kernel variants (D=64/128 x MAX_SPLITS=1/2/4/8/16) in one metallib.

## Promotion Pipeline

The C++ `PromotionPipeline` replaces the earlier Python `AsyncPromoter`.
One standalone object per generation session, spanning all layers:

- `drain_spikes()` runs on main thread: reads spike_flags/tokens zero-copy
  from unified memory, filters (cooldown, dedup, budget, GQA), queues records
- Worker thread: reads fp16 from POSIX mmap, memcpy into blob near-tier
  headroom, atomic increment `near_valid[h]`
- Kernel sees promoted tokens on next dispatch — zero overhead

Thread safety: worker writes K,V data then increments count. ARM64 word-
atomic guarantees no torn reads. Worker writes to slot `near_valid[h]`
(unused), kernel reads slots `0..near_valid[h]-1` (used). No overlap.

## Tests

69 passing (26 MLX foveated + 8 disk archive + 29 scoring + 6 other)

## What's Next

- Demonstrate on larger models (7B+) at 32K+ context where the kernel
  speedup delivers end-to-end gains
- End-to-end benchmarks on the 0.5B model to measure decode tok/s
- Formal write-up of the architecture and benchmark results
