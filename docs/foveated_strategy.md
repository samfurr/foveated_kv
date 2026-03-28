# FoveatedKV: Technical Strategy

## Core Idea

All tokens contribute to attention at varying precision — no eviction, no softmax
renormalization error. Lossless promotion via NVMe-backed fp16 archives. Speed comes
from reduced memory bandwidth (reading fewer bytes per token), not from pruning tokens.

```
Apple Silicon Unified Memory (all KV reads during decode):
  +-- Near K,V:          fp16   (2 bytes/elem -- same as standard, no savings)
  +-- Far K:             fp8 E4M3 (1 byte/elem -- 2x bandwidth savings)
  +-- Far V:             INT4   (0.5 bytes/elem -- 4x bandwidth savings)
                         + per-token scale/zero

NVMe Disk Archive (not accessed during decode -- only during promotion):
  +-- fp16 mmap archive: exact originals for all non-near tokens
      Written once after prefill. Enables lossless promotion. ~50us/token read.
```

**We still attend to ALL S tokens every step.** The number of dot products, softmax
entries, and weighted-sum operations is unchanged. The speedup is purely from reading
fewer bytes from memory. Since decode attention is memory-bandwidth-bound, fewer
bytes = faster.

## The Foveated Rendering Analogy

| Foveated Rendering | FoveatedKV |
|--------------------|-----------|
| Eye position (gaze) | Current query vector |
| Sharp center (fovea) | Near tier: top 10% of tokens, fp16 |
| Blurry far field | Far tier: bottom 90% at fp8 E4M3 K + INT4 V |
| Full-res frame buffer | NVMe mmap archive of fp16 originals |
| Saccade (eye movement) | Query drift triggering promotion |
| Re-render at full quality | Promote: C++ worker reads fp16 from disk, writes into near headroom |
| Eye tracker (separate processor) | Metal kernel spike detection (free byproduct of softmax) |

The analogy is structurally sound but not perfect:
- Full-quality data exists in an archive (frame buffer / NVMe) -- yes
- Only the attended region is served at full quality (near / fp16 tier) -- yes
- Far regions contribute at reduced quality (blur / quantization) -- yes
- When focus shifts, exact data is fetched from the archive -- yes
- **Caveat**: VR foveation uses BLUR (low-pass filter). We use QUANTIZATION (adds noise).
  Our analog: low-attention-weight tokens have less IMPACT on the output, so noise on
  them matters less. Functionally similar, not technically identical.

## Design Decisions

### Per-head tier assignment

Each attention head maintains its own near/far sets. Different
heads attend to different tokens -- a retrieval head might focus on factual tokens while
a local head focuses on recent context. Sharing one global near set across all heads
would force compromises.

### Asymmetric K/V precision

K quantization is more dangerous than V quantization:
- **K error** -> noisy attention scores -> softmax amplifies via exp() -> shifts weights
  on ALL tokens including near
- **V error** -> direct additive noise scaled by attention weight -> linear, bounded

Following KIVI's insight, we use higher precision for K than V within the far tier:

| Tier | K precision | V precision | Rationale |
|------|------------|-------------|-----------|
| Near | fp16 | fp16 | Full quality for high-attention tokens |
| Far | fp8 E4M3 | INT4 | K stays fp8 to protect score quality; V tolerates more noise |

The far K decode is: rebias exponent (add 8), zero-pad mantissa. ~5 integer ops in
registers. Or via the 256-entry LUT in threadgroup memory (1 read vs 10+ ALU ops).
The far V dequant is: scale * nibble + zero, per-token scale/zero loaded once.

Ablation result: removing this asymmetry causes **3.6x more cosine error**. This is the
single most important design decision in the system.

### Quantization granularity

- **Keys**: fp8 E4M3 per-token quantization. LUT decode in threadgroup memory.
- **Values**: INT4 per-token quantization (one scale + one zero per token). Vectorized
  uint16 loads + FMA dequant.
- **Quantization is per-head**: each KV head has its own scales/zero-points.
- **MLX implementation** (`mlx_quantize.py`): handles fp8 E4M3 and INT4 packed formats.

### Newly generated tokens during decode

Each decode step produces one new token with new K,V. Policy:

1. **Add to decode buffer.** New tokens accumulate in an fp16 decode buffer within the
   Metal kernel's attention scope (uses `attend_fp16` helper, same as near tier).
2. **Archive to disk immediately.** The mmap write for 1 token is negligible.
3. **At next promotion cycle**, C++ pipeline evaluates whether tokens should be
   promoted based on spike detection.

## Speed Budget Analysis

### Target: Apple Silicon (M1/M2/M3/M4 class)

Apple Silicon unified memory bandwidth ranges from ~100 GB/s (M1) to ~800 GB/s
(M4 Max). The key advantage: no PCIe bottleneck between CPU and GPU. Unified memory
means tier management and promotion happen without explicit transfers.

**Standard decode (7B model, 4K context, batch=1):**
```
KV read:      32 layers x 8 KV heads x 4K x 128 dim x 2 bytes x 2 (K+V) = 0.5 GB
Weight read:  ~6.6 GB (fp16 weights)
M-series BW:  200-400 GB/s (typical)
KV time:      0.5 GB / 200 GB/s = 2.5 ms
Weight time:  ~33 ms
Total:        ~35.5 ms per step
```

**Foveated decode (7B model, 4K context, batch=1):**
```
Near   (400 tokens):    K fp16 + V fp16 = 25 MB
Far    (3.6K tokens):   K fp8 + V INT4 = ~54 MB (fp8 K = 1B, INT4 V = 0.5B + scale/zero)
Total KV read:  ~79 MB -> 0.40 ms at 200 GB/s
```

**Measured results (7B shapes, kernel microbenchmark):**
```
1K context:   0.84x (slight overhead at short context)
4K context:   1.72x faster than fp16 SDPA
8K context:   2.47x faster than fp16 SDPA
16K context:  3.34x faster than fp16 SDPA
32K context:  2.93x faster than fp16 SDPA
```

**End-to-end decode performance (Qwen2.5-0.5B):**
```
4-bit (Qwen2.5-0.5B-Instruct-4bit):
  512:  96 tok/s fused vs 135 standard (0.71x)
  1K:   97 tok/s fused vs 131 standard (0.74x)
  2K:   98 tok/s fused vs 121 standard (0.81x)
  4K:   67 tok/s fused vs 107 standard (0.63x)

bf16 (Qwen2.5-0.5B-Instruct-bf16):
  512:  55 tok/s fused vs 66 standard (0.83x)
  1K:   53 tok/s fused vs 64 standard (0.83x)
  2K:   54 tok/s fused vs 60 standard (0.90x)

Note: fused path is slower on 0.5B due to Python SDPA interceptor overhead.
The value is 2x memory compression enabling longer contexts, and on
memory-constrained 7B (8GB Mac), foveated is 2-8x faster because standard
is swap-bound. The kernel itself is 1.7-3.3x faster in isolation.
```

The Metal Split-K kernel achieves speedup by:
- Reading fp8/INT4 from unified memory (fewer bytes)
- Dequantizing in registers (never materializes fp16)
- Online softmax across both tiers in one pass
- Function constants for N_NEAR/N_FAR enabling GPU loop optimization

### Unified memory advantage

On Apple Silicon, there is no CPU-GPU memory transfer bottleneck:
- C++ worker writes directly into blob unified memory (raw pointers)
- Promoted fp16 values are read from NVMe mmap archive (~50us/token)
- `near_valid[h]` increment makes promoted tokens visible to GPU
- No PCIe transfers to schedule or overlap

## Lossless Promotion Protocol

**Promoted tokens are restored to bit-exact fp16 from the disk archive**, not
dequantized from a lossy quantized representation.

### Token lifecycle:

```
Prefill:    Token computed -> fp16 K,V in unified memory (all tokens, full attention)
            |
Compress:   Score all tokens by attention importance (per-head)
            +-- Top 10%:   Stay fp16 in memory (near)
            +-- Rest 90%:  Quantize K(fp8 E4M3)+V(INT4) in memory (far)
                           Write fp16 original to disk mmap archive
            |
Decode:     Attend over both tiers in one Metal kernel (every step)
            Newly generated tokens: added to decode buffer
            |
Promotion:  C++ pipeline detects spikes (fire-and-forget, background)
            Metal kernel spike detection flags urgent promotions
            +-- C++ worker reads exact fp16 from disk mmap
            +-- Writes into near-tier headroom in blob
            +-- Atomic increment near_valid[h] = commit point
            +-- Kernel sees promoted token next dispatch
```

### Why lossless promotion matters:

**Without it:** Each promote/demote cycle adds quantization noise. Over 1000+ generated
tokens with multiple promotion cycles, error accumulates.

**With it:** Promoted tokens are bit-exact fp16. Error never accumulates. The PPL ratios
confirm this: 0.999x at 2K, 1.007x at 4K. Error does NOT grow with context length.

## Intra-Step Spike Detection

Implemented in the Metal Split-K kernel at near-zero cost. The kernel tracks both the
spike flag AND the specific far-local token index -- the C++ pipeline gets the exact
far-tier token index without any re-scoring work.

**How it works inside the kernel:**
The kernel already computes scores for all tokens during online softmax. Spike detection
piggybacks on this with a few scalar operations per split: track min_near_score and
max_far_score, compare at the end, write one flag + one token index per query head.

The C++ pipeline reads these directly via `drain_spikes()` — zero-copy from unified
memory. Filtering (cooldown, dedup, budget, GQA dedup) happens in C++ before queuing
for the background worker.

## C++ Promotion Pipeline

The `PromotionPipeline` (one per generation session, all layers) replaces the earlier
Python `AsyncPromoter`:

- **drain_spikes()** (main thread): reads spike_flags/tokens zero-copy, filters:
  - Per-(layer, head) cooldown: 5-step minimum between spikes
  - Position dedup: splitmix64 hash set, never re-promote same token
  - Budget: max N promotions per drain call
  - GQA dedup: multiple q-heads → same kv-head, process once
- **Worker thread**: reads fp16 from POSIX mmap, memcpy into blob near headroom,
  atomic increment `near_valid[h]`

**No override buffers.** No merge-scan in kernel. No extra Metal buffer bindings.
Promoted tokens are just near tokens.

## Error Bound (non-accumulating)

```
At any decode step t, the attention output error is bounded by:

  ||output_foveated - output_exact||
    <= alpha_fp8 * eps_fp8  +  alpha_int4 * eps_int4

Where:
  alpha_fp8 = attention mass fraction in fp8 tier (K contribution)
  alpha_int4 = attention mass fraction in INT4 tier (V contribution)

Typical (10% near, 90% far with fp8 K + INT4 V):
  alpha_near ~ 0.80 (captures most attention mass)
  alpha_far ~ 0.20
  error <= 0.20 * max(eps_fp8, eps_int4) (bounded, small)

This bound is INDEPENDENT of decode step t -- it does not grow with time.
Confirmed empirically: PPL ratio stays flat across 1K-4K context.
```

## Kernel Architecture

The fused kernel merges Split-K + Reduce into a single Metal dispatch via
threadgroup shared memory. Multiple SIMD groups per threadgroup (one per
split), each processes a token range in parallel. After a threadgroup barrier,
the first SIMD group reduces all partials and writes the final output.

Key performance techniques:
- **Pre-scaled query**: q *= 1/sqrt(D) at load time, amortizing one multiply
  per token across the entire sequence
- **Single-exp softmax_accum**: computes alpha and w once each, FMA accumulation
  fuses rescale + accumulate in one pass over acc[]
- **attend_fp16 helper**: shared template for near tier and decode buffer
- **LUT fp8 decode**: 256-entry threadgroup memory table, 1 read vs 10+ ALU ops
- **Score-gated V**: skip int4 dequant when exp(score - m) < 1e-7
- **4-token blocked far loop**: ILP across independent K dot products
- **Vectorized loads**: uint32 for fp8 K, uint16 for int4 V
- **Branch-free near**: loop bound = min(split_end, near_valid[h])

Adaptive split_size caps num_splits <= 16 to avoid reduce bottleneck at long
contexts (256 base, growing to 8192 at 128K). Single dispatch eliminates
the second kernel and 6 global partial buffer arrays.

## Kernel Precision: fp16 Dequant Rounding

Critical finding: custom GPU kernels that dequantize quantized KV data must
round to fp16 precision before attention computation, even if the kernel
internally uses float32 accumulators.

The reference path (dequant to fp16 -> Apple's SDPA) produces outputs at fp16
precision. A custom kernel that dequants to float32 produces slightly different
values (~1 ULP per element). Over 24 transformer layers of autoregressive
decoding, these differences compound through nonlinear layers (LayerNorm, GELU)
and cause greedy generation to diverge after ~5 tokens.

The fix is simple: `to_fp16(raw * scale + zero)` in the kernel source. This
rounds the dequantized value to fp16 precision while keeping all subsequent
accumulation (dot products, softmax, V weighting) in float32 for stability.

Result: byte-identical greedy decode output to standard fp16 attention.
