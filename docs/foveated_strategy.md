# FoveatedKV: Technical Strategy

## Core Idea

All tokens contribute to attention at varying precision — no eviction, no softmax
renormalization error. Lossless promotion via NVMe-backed fp16 archives. Speed comes
from reduced memory bandwidth (reading fewer bytes per token), not from pruning tokens.

```
Apple Silicon Unified Memory (all KV reads during decode):
  +-- Foveal K,V:         fp16   (2 bytes/elem -- same as standard, no savings)
  +-- Peripheral K,V:     INT8   (1 byte/elem -- 2x bandwidth savings)
  +-- Far peripheral K,V: INT4   (0.5 bytes/elem -- 4x bandwidth savings)

NVMe Disk Archive (not accessed during decode -- only during promotion):
  +-- fp16 mmap archive: exact originals for all non-foveal tokens
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
| Foveal region (sharp center) | Top-R tokens by attention score, fp16 |
| Peripheral region (blurred) | Lower-attention tokens at INT8/INT4 |
| Full-res frame buffer | NVMe mmap archive of fp16 originals |
| Saccade (eye movement) | Query drift triggering tier reassignment |
| Re-render at full quality | Promote: fetch exact fp16 from disk archive |
| Eye tracker (separate processor) | Async background workers for tier scoring |

The analogy is structurally sound but not perfect:
- Full-quality data exists in an archive (frame buffer / NVMe) -- yes
- Only the attended region is served at full quality (fovea / fp16 tier) -- yes
- Peripheral regions contribute at reduced quality (blur / quantization) -- yes
- When focus shifts, exact data is fetched from the archive -- yes
- **Caveat**: VR foveation uses BLUR (low-pass filter). We use QUANTIZATION (adds noise).
  Our analog: low-attention-weight tokens have less IMPACT on the output, so noise on
  them matters less. Functionally similar, not technically identical.

## Design Decisions

### Per-head tier assignment

Each attention head maintains its own foveal/peripheral/far-peripheral sets. Different
heads attend to different tokens -- a retrieval head might focus on factual tokens while
a local head focuses on recent context. Sharing one global foveal set across all heads
would force compromises.

### Asymmetric K/V precision

K quantization is more dangerous than V quantization:
- **K error** -> noisy attention scores -> softmax amplifies via exp() -> shifts weights
  on ALL tokens including foveal
- **V error** -> direct additive noise scaled by attention weight -> linear, bounded

Following KIVI's insight, we use higher precision for K than V within each tier:

| Tier | K precision | V precision | Rationale |
|------|------------|-------------|-----------|
| Foveal | fp16 | fp16 | Full quality for high-attention tokens |
| Peripheral | INT8 | INT8 | Balanced -- K errors bounded at this precision |
| Far peripheral | INT8 | INT4 | K stays INT8 to protect score quality; V tolerates more noise |

Ablation result: removing this asymmetry causes **130x error increase**. This is the
single most important design decision in the system.

### Quantization granularity

- **Keys**: per-channel quantization (one scale per head dimension). Preserves relative
  score ordering within each head.
- **Values**: per-token quantization (one scale per token). Tighter fit per token's
  value range.
- **Quantization is per-head**: each KV head has its own scales/zero-points.
- **MLX implementation** (`mlx_quantize.py`): matches PyTorch reference exactly for
  INT8 per-channel/per-token and INT4 packed formats.

### Newly generated tokens during decode

Each decode step produces one new token with new K,V. Policy:

1. **Add to decode buffer.** New tokens accumulate in an fp16 decode buffer within the
   Metal kernel's attention scope.
2. **Archive to disk immediately.** The mmap write for 1 token is negligible.
3. **At next promotion cycle**, background workers evaluate whether tokens should be
   promoted or demoted based on current attention patterns.

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
Foveal   (200 tokens):   K fp16 + V fp16 = 12.5 MB
Periph   (1K tokens):    K INT8 + V INT8 = 32 MB
Far      (2.8K tokens):  K INT8 + V INT4 = 72 MB
Total KV read:  ~117 MB -> 0.58 ms at 200 GB/s
```

**Measured synthetic results (7B shapes):**
```
4K context:  1.49x faster than fp16 SDPA
8K context:  1.46x faster than fp16 SDPA
32K context: 3.28x faster (original benchmark shapes)
```

The Metal Split-K kernel achieves speedup by:
- Reading INT8/INT4 from unified memory (fewer bytes)
- Dequantizing in registers (never materializes fp16)
- Online softmax across all tiers in one pass
- Compile-time N_FOV for loop unrolling

### Unified memory advantage

On Apple Silicon, there is no CPU-GPU memory transfer bottleneck:
- Tier management runs on CPU threads using numpy (no MLX in workers)
- Promoted fp16 values are read from NVMe mmap archive (~50us/token)
- Updated tier data is immediately visible to the GPU
- No PCIe transfers to schedule or overlap

## Lossless Promotion Protocol

**Promoted tokens are restored to bit-exact fp16 from the disk archive**, not
dequantized from a lossy quantized representation.

### Token lifecycle:

```
Prefill:    Token computed -> fp16 K,V in unified memory (all tokens, full attention)
            |
Compress:   Score all tokens by attention importance (per-head)
            +-- Top-R:     Stay fp16 in memory (foveal)
            +-- Next-M:    Quantize K(INT8)+V(INT8) in memory (peripheral)
            |              Write fp16 original to disk mmap archive
            +-- Rest:      Quantize K(INT8)+V(INT4) in memory (far peripheral)
                           Write fp16 original to disk mmap archive
            |
Decode:     Attend over all tiers in one Metal kernel (every step)
            Newly generated tokens: added to decode buffer + archived to disk
            |
Promotion:  Async workers detect tier changes (fire-and-forget, background)
            Metal kernel spike detection flags urgent promotions
            +-- Tokens moving UP:
            |   Fetch EXACT fp16 from disk archive (lossless, bit-identical)
            +-- Tokens moving DOWN:
                Quantize in memory (original already archived from initial compress)
```

### Why lossless promotion matters:

**Without it:** Each promote/demote cycle adds quantization noise. Over 1000+ generated
tokens with multiple promotion cycles, error accumulates.

**With it:** Promoted tokens are bit-exact fp16. Error never accumulates. The PPL ratios
confirm this: 0.998x at 1K, 0.993x at 2K, 1.003x at 4K. Error does NOT grow with
context length.

## Intra-Step Spike Detection

Implemented in the Metal Split-K kernel at near-zero cost. The kernel tracks both the
spike flag AND the specific token ID -- the async promoter gets the exact far-tier
token index without any re-scoring work.

**How it works inside the kernel:**
The kernel already computes scores for all tokens during online softmax. Spike detection
piggybacks on this with a few scalar operations per tile: track min foveal score and
max far score, compare at the end, write one flag + one token index per KV head.

The async promoter reads these directly -- no re-scoring needed. Fire-and-forget handoff
from the main thread to 2 background workers (raw spike processing + disk reads).

## Async Promotion System

Two background workers replace the earlier per-layer coprocessor design:

1. **Spike worker**: Processes raw spike events from the Metal kernel. Maps far-local
   indices to archive indices. Queues promotion updates.
2. **Disk worker**: Reads fp16 originals from NVMe mmap archives. O(1) drain via dict
   keyed by layer index.

Key constraint: **no MLX operations in worker threads.** Workers use numpy only.
MLX's computation graph is not thread-safe, so all MLX operations happen on the main
thread when updates are applied.

## Error Bound (non-accumulating)

```
At any decode step t, the attention output error is bounded by:

  ||output_foveated - output_exact||
    <= alpha_int8 * eps_int8  +  alpha_int4 * eps_int4

Where:
  alpha_int8 = attention mass fraction in INT8 tier
  alpha_int4 = attention mass fraction in INT4 tier

Typical (2% foveal, 18% peripheral INT8, 80% far INT4):
  alpha_foveal ~ 0.80 (captures most attention mass)
  alpha_int8 ~ 0.15
  alpha_int4 ~ 0.05
  error <= 0.15 * 0.004 + 0.05 * 0.06 = 0.0036 (0.36%)

This bound is INDEPENDENT of decode step t -- it does not grow with time.
Confirmed empirically: PPL ratio stays flat across 1K-4K context.
```

## Sustained Accuracy + Promotion

Frozen foveated tiers maintain 0.996+ cosine similarity to standard fp16
across all context lengths. The override buffer provides lossless promotion:

- Metal kernel detects spikes as a free byproduct of online softmax
- Background worker reads exact fp16 from NVMe mmap archive (~50us/token)
- Worker does sorted insert into double-buffered numpy arrays + atomic swap
- Kernel merge-scans the pre-sorted buffer: O(N_FAR + MAX_OV) per layer
- No tensor mutation during decode, no GPU faults, no locks

Tier assignment uses pure recency (sinks + recent window → foveal, rest by
distance). Deterministic, fast, no argpartition non-determinism. Promotion
via override buffer rescues important early tokens that land in the far tier.

## Kernel Architecture

The fused kernel merges Split-K + Reduce into a single Metal dispatch via
threadgroup shared memory. Multiple SIMD groups per threadgroup (one per
split), each processes a token range in parallel. After a threadgroup barrier,
the first SIMD group reduces all partials and writes the final output.

Adaptive split_size caps num_splits ≤ 16 to avoid reduce bottleneck at long
contexts (256 base, growing to 8192 at 128K). Single dispatch eliminates
the second kernel and 6 global partial buffer arrays.

Scale+zero pairs are pre-packed during compression (concat once, not per call).
C++ extension packs all 11 static arrays into a single uint8 blob with
16-byte alignment — 9 Metal buffer arguments total per dispatch.

## Kernel Precision: fp16 Dequant Rounding

Critical finding: custom GPU kernels that dequantize quantized KV data must
round to fp16 precision before attention computation, even if the kernel
internally uses float32 accumulators.

The reference path (dequant to fp16 → Apple's SDPA) produces outputs at fp16
precision. A custom kernel that dequants to float32 produces slightly different
values (~1 ULP per element). Over 24 transformer layers of autoregressive
decoding, these differences compound through nonlinear layers (LayerNorm, GELU)
and cause greedy generation to diverge after ~5 tokens.

The fix is simple: `to_fp16(raw * scale + zero)` in the kernel source. This
rounds the dequantized value to fp16 precision while keeping all subsequent
accumulation (dot products, softmax, V weighting) in float32 for stability.

Result: byte-identical greedy decode output to standard fp16 attention.
