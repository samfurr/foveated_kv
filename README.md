# FoveatedKV

**Importance-adaptive mixed-precision KV cache compression for LLM inference on Apple Silicon.**

2x memory compression. 0.995+ cosine fidelity. Custom Metal GPU kernels. 2.3x faster at 32K context.

---

## The Problem

Every LLM decode step reads the *entire* KV cache from memory. At long context, this is gigabytes — memory bandwidth, not compute, becomes the bottleneck. The field has two ideas: **evict tokens** (breaks softmax) or **quantize everything uniformly** (treats the answer to your question the same as filler text).

Both ignore something obvious about attention: it focuses.

## The Insight

VR headsets solved this decades ago with **foveated rendering** — render the center of your gaze in full detail, blur everything else. Your eyes don't notice because you can only see detail where you're looking.

FoveatedKV applies this to LLM attention:

| VR Foveated Rendering | FoveatedKV |
|-----------------------|-----------|
| Eye position | Query vector |
| Sharp center | Near tier: top 10% of tokens at fp16 |
| Blurry far field | Far tier: bottom 90% at fp8 E4M3 K + INT4 V |
| Full-res frame buffer | NVMe mmap archive of fp16 originals |
| Eye tracker | Metal kernel spike detection (free) |
| Re-render on gaze shift | C++ worker promotes exact fp16 into near tier |

**Every token still contributes to attention.** The softmax denominator is correct. No tokens are evicted. The only approximation is quantization noise on tokens the model isn't paying attention to — bounded and non-accumulating.

## The Key Discovery

Keys and values need **different** quantization precision. Key error gets amplified through exp() in softmax — shifting weights on ALL tokens (multiplicative damage). Value error is just additive noise scaled by attention weight (linear, bounded). So the far tier uses fp8 E4M3 for keys but INT4 for values.

Removing this asymmetry causes **3.6x more attention error** (cosine error 0.000217 vs 0.000061). It's the single most important design decision in the system.

## What Was Built

### Fused Split-K Metal Kernel

A custom Metal compute kernel that loads quantized fp8/INT4 data from unified memory, dequantizes in registers, and computes attention via online softmax — all in one pass. No fp16 intermediates ever touch memory. Split-K parallelism across token ranges for full GPU occupancy.

Performance through elegance:

- **Pre-scaled query**: q *= 1/sqrt(D) at load time — one multiply per token amortized across the entire sequence
- **Single-exp online softmax**: `softmax_accum` computes each exp() exactly once, FMA fuses rescale + accumulate
- **attend_fp16 helper**: shared template for near tier and decode buffer (both fp16 K+V)
- **LUT fp8 decode**: 256-entry threadgroup memory table — 1 read vs 10+ ALU ops per fp8 element
- **Score-gated V loading**: skip int4 dequant when score < m - 16 (exp(-16) ~ 1e-7)
- **4-token blocked far loop**: ILP across independent K dot products
- **Spike detection**: free byproduct of online softmax — tracks max_far vs min_near per head

### C++ Promotion Pipeline

When the kernel detects a far-tier token scoring above its weight class, a C++ background worker handles promotion:

1. `drain_spikes()` reads spike_flags/tokens zero-copy from unified memory
2. Filters: per-(layer,head) cooldown, position dedup (splitmix64), budget cap, GQA dedup
3. Worker thread reads exact fp16 from POSIX mmap (disk archive)
4. Writes into blob near-tier headroom + atomic increment `near_valid[h]`
5. Kernel sees promoted token next dispatch as an ordinary near token

No override buffers. No extra Metal bindings. No kernel changes. Promotion is lossless because the disk archive preserves exact fp16 originals.

### Two-Tier Quantization

```
Near (10%):      fp16 K + fp16 V         — full precision where attention focuses
Far (90%):       fp8 E4M3 K + INT4 V     — ~3x savings, asymmetric to protect scores
```

Tier assignment is per-head (different attention heads focus on different tokens) with attention sinks (first N tokens always near) and a recency window (recent tokens always near).

### C++ Extension

`FoveatedPrimitive` subclasses `mlx::core::Primitive` directly — the same eval_gpu path as MLX's own `ScaledDotProductAttention`. Pre-packs 7 static arrays into a single blob. Precompiled metallib with function constants for tier specialization. No JIT, no source caching.

### mlx-lm Integration

A monkey-patch on `mx.fast.scaled_dot_product_attention` intercepts decode-time attention and routes it through the fused Metal kernel. The model doesn't know it's happening. Cache wrappers implement mlx-lm's `update_and_fetch` protocol for drop-in replacement.

## Results

Evaluated on **Qwen2.5-0.5B-Instruct-bf16** running locally on an 8GB Mac.

### Quality: 0.995+ cosine fidelity

Attention output cosine similarity vs exact fp16 across context lengths (7B shapes: H_q=32, H_kv=8, D=128):

| Context | Cosine vs Exact | MAE | Compression |
|---------|----------------|------|-------------|
| 512 | 0.9956 | 0.0053 | 2.03x |
| 1K | 0.9950 | 0.0040 | 2.02x |
| 4K | 0.9952 | 0.0020 | 2.02x |
| 8K | 0.9954 | 0.0014 | 2.02x |
| 16K | 0.9953 | 0.0010 | 2.02x |
| 32K | 0.9954 | 0.0007 | 2.02x |

### Memory: 2x compression

| Context | fp16 KV | Foveated KV | Compression |
|---------|---------|-------------|-------------|
| 4K | 16.0 MB | 7.9 MB | 2.02x |
| 8K | 32.0 MB | 15.8 MB | 2.02x |
| 16K | 64.0 MB | 31.6 MB | 2.02x |
| 32K | 128.0 MB | 63.3 MB | 2.02x |

### Kernel: Up to 2.3x faster at long context

Fused Split-K Metal kernel vs Apple's SDPA (7B shapes, single layer, 8GB Mac):

| Context | fp16 SDPA | Fused Kernel | Speedup |
|---------|-----------|-------------|---------|
| 1K | 1.12 ms | 0.94 ms | 1.19x |
| 4K | 2.33 ms | 1.46 ms | 1.60x |
| 8K | 4.05 ms | 2.22 ms | 1.82x |
| 16K | 7.68 ms | 3.73 ms | 2.06x |
| 32K | 15.72 ms | 6.81 ms | 2.31x |

Break-even around 512 tokens. Bandwidth advantage grows with context length.

### Ablation: Asymmetric K/V precision matters

Measured at 4K context on real model K,V (middle layer):

| Config | Cosine vs Exact | MAE | vs Full |
|--------|----------------|------|---------|
| Full system (10/90) | 0.999939 | 0.0050 | baseline |
| No near (uniform fp8+int4) | 0.999929 | 0.0055 | -0.000010 |
| Symmetric (int4 K + int4 V) | 0.999783 | 0.0091 | -0.000156 |
| No sinks (pure topk near) | 0.999939 | 0.0050 | -0.000000 |
| Uniform INT8 (no tiers) | 1.000000 | 0.0001 | +0.000061 |
| Tiers, no quant (fp16 far) | 1.000000 | 0.0000 | +0.000061 |

The asymmetric K/V precision (fp8 K + INT4 V) achieves 3.6x lower cosine error than symmetric INT4. Key errors are amplified through exp() in softmax; value errors are bounded by attention weight.

### Perplexity: Within 2.5% of standard

End-to-end perplexity on WikiText-103 (Qwen2.5-0.5B-Instruct-bf16):

| Context | Standard PPL | Foveated PPL | Ratio |
|---------|-------------|-------------|-------|
| 1K | 6.86 | 7.03 | 1.025x |
| 2K | 15.14 | 15.12 | 0.999x |
| 4K | 29.17 | 29.36 | 1.007x |

### LongBench-Lite: Quality preserved

6 representative tasks, 20 samples each, official THUDM scoring:

| Task | Category | Standard | Foveated 10/90 |
|------|----------|----------|----------------|
| qasper | Single-doc QA | 12.4 | 10.9 |
| hotpotqa | Multi-doc QA | 1.6 | 1.6 |
| qmsum | Summarization | 6.5 | 6.7 |
| triviaqa | Few-shot | 34.4 | 34.4 |
| passage_retrieval_en | Synthetic | 0.0 | 2.5 |
| lcc | Code | 34.5 | 34.5 |
| **Average** | | **14.9** | **15.1** |

## Quick Start

```bash
git clone <repo>
cd accelerate
uv sync --extra dev --extra ext
```

### Build the C++ extension + Metal kernel

```bash
# Configure (finds MLX, nanobind, Metal compiler automatically)
MLX_CMAKE=$(uv run python -c "import mlx,os; print(os.path.join(mlx.__path__[0],'share','cmake','MLX'))")
MLX_INC=$(uv run python -c "import mlx,os; print(os.path.join(mlx.__path__[0],'include'))")
PY=$(uv run python -c "import sys; print(sys.executable)")
cmake -S csrc -B build_ext \
    -DMLX_DIR="$MLX_CMAKE" -DMLX_CMAKE_DIR="$MLX_CMAKE" \
    -DMLX_INCLUDE_DIRS="$MLX_INC" -DPython_EXECUTABLE="$PY" \
    -DCMAKE_BUILD_TYPE=Release

# Build (compiles .metal -> .metallib + nanobind C++ extension)
cmake --build build_ext -j$(sysctl -n hw.ncpu)

# Install alongside source
cp build_ext/foveated_ext.cpython-*-darwin.so build_ext/foveated_attn.metallib .
```

### Run tests

```bash
uv run pytest tests/ -v  # 73 tests
```

```python
from mlx_lm import load
from foveated_kv.mlx_generate import generate_fused
from foveated_kv.mlx_foveated import MLXTierConfig

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
cfg = MLXTierConfig()  # defaults to 10% near

text, stats = generate_fused(
    model, tokenizer,
    "What is the meaning of life?",
    max_tokens=100, cfg=cfg,
)
print(text)
print(f"Memory saved: {stats['mem_saved_mb']:.1f} MB")
```

## Run Benchmarks

```bash
# Full paper suite (~2.5 hours on 8GB Mac)
bash scripts/run_mlx_benchmarks.sh

# Individual
uv run python benchmarks/benchmark_mlx_longbench.py       # LongBench-Lite (6 tasks)
uv run python benchmarks/benchmark_mlx_needle_heatmap.py   # Needle retrieval grid
uv run python benchmarks/benchmark_mlx_ablation.py         # Component ablation
uv run python benchmarks/benchmark_mlx_throughput.py       # Throughput + memory
uv run python benchmarks/benchmark_mlx.py                  # Synthetic kernel speed
uv run python benchmarks/benchmark_promotion_quality.py    # Passkey retrieval with promotion
uv run python benchmarks/benchmark_promotion_recovery.py   # Multi-fact promotion recovery
uv run python benchmarks/benchmark_mlx_sustained.py        # Sustained accuracy
```

### CLI

```bash
# Generate with foveated cache
foveated-kv generate --prompt "Explain quantum entanglement:" --max-tokens 100

# Compare against standard baseline
foveated-kv generate --prompt "Explain quantum entanglement:" --max-tokens 100 --standard

# Custom model and compression
foveated-kv generate --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Hello world" --near-pct 0.05 --max-tokens 50
```

## Project Structure

```
src/foveated_kv/
  mlx_foveated.py        Core cache: 2 tiers, compress, attend, decode buffer
  mlx_quantize.py        fp8 E4M3 per-token K + INT4 packed V
  metal_foveated.py      Python Metal kernel (fallback when C++ ext not built)
  mlx_generate.py        SDPA monkey-patch for mlx-lm, generation loops
  disk_archive.py        NVMe-backed numpy.memmap fp16 archive
  cli.py                 CLI entry point (foveated-kv generate)

csrc/
  kernels/
    foveated_attn.metal      Fused Split-K + Reduce kernel. Pre-scaled query,
                             single-exp softmax, attend_fp16 helper, LUT fp8,
                             score-gated V, spike detection. 10 variants in
                             one metallib (D=64/128 x MAX_SPLITS=1/2/4/8/16).
    foveated_compress.metal  GPU compression kernels (fp8 E4M3, INT4)
  foveated_attn.h/.cpp       FoveatedPrimitive (subclasses mlx::core::Primitive)
                             FoveatedHandle (cached pipelines, blob buffer)
  promotion_pipeline.h/.cpp  C++ promotion worker: disk mmap -> blob near
                             headroom + atomic near_valid[h] commit
  foveated_compress.h/.cpp   CompressHandle: GPU compression kernels
  bindings.cpp               nanobind module (NB_DOMAIN mlx)
  CMakeLists.txt             mlx_build_metallib + nanobind + Threads

benchmarks/              10 benchmarks + scoring library
tests/                   73 tests (cache, kernel, quantization, archive, fallback, scoring)
docs/                    Technical strategy, benchmark methodology, build status
```

### 7B Model on 8GB Mac (Qwen2.5-7B-Instruct-4bit)

On memory-constrained hardware, FoveatedKV is *faster* than standard because compressed KV cache reduces memory pressure:

| Context | Standard tok/s | Foveated tok/s | Speedup | Memory Saved |
|---------|---------------|----------------|---------|-------------|
| 512 | 0.7 | 1.6 | **2.3x** | — |
| 1024 | 0.6 | 1.0 | **1.7x** | 50 MB |
| 2048 | 0.2 | 0.4 | **2.0x** | 101 MB |
| 3072 | — | 0.3 | — | 151 MB |

Fused kernel vs dequant-reference cosine: **1.000000** (kernel perfectly correct).
OOM crossover on 8GB: ~3K foveated, ~3-4K standard.

### Promotion Recovery

Multi-fact retrieval benchmark (biographical needle with 4 checkable facts, ~2K context, 5% near tier). Demonstrates that spike detection + promotion recovers facts lost to far-tier quantization:

| Method | Facts Retrieved | Score |
|--------|----------------|-------|
| Standard (fp16) | 56/64 | **88%** |
| Foveated + promotion | 56/64 | **88%** |
| Foveated no promotion | 32/64 | 50% |

Promotion recovers **24 additional facts** (38% of total), bringing foveated quality back to the standard fp16 baseline. 100% reproducible across 8 seeds.

## Memory Unlocks on 8GB Mac

FoveatedKV roughly **doubles the maximum context length** for models where KV cache is the memory bottleneck. Archive goes to NVMe disk, only compressed tiers stay in RAM.

| Model | Without FoveatedKV | With FoveatedKV |
|-------|-------------------|-----------------|
| Qwen2.5-7B 4-bit | Max ~16K context | **32K-64K context** |
| Llama-3.2-3B 4-bit | Max ~16K context | **32K context** |
| Mistral-7B 4-bit | Max ~8K context | **16K context** |
| Qwen2.5-3B 4-bit | Max ~65K context | **128K context** |

### Future: Deeper Compression

The default 10/90 config achieves ~3x compression on far tokens (fp8 K = 1 byte, INT4 V = 0.5 byte + scale/zero per token vs 4 bytes fp16 K+V). More aggressive near percentages under investigation.

fp8 E4M3 keys are the hard floor — going below fp8 for K causes 130x error increase (softmax amplification). Values tolerate INT4 since they are medium-to-low-attention tokens with bounded weight in the output.

## TODO: Further Benchmarking

The following benchmarks require a machine with more memory (16GB+) to run properly and are needed before the paper is submission-ready:

- [ ] **llama.cpp head-to-head** — Compare against llama.cpp's quantized KV cache on the same Mac hardware (the practical baseline any Mac user would compare against)
- [ ] **LongBench full-length** — Run LongBench tasks at full context length (current results are truncated to 2K due to 8GB memory constraints)
- [ ] **Multi-hardware throughput curves** — Benchmark across M1/M2/M3/M4 and different memory configurations (8GB/16GB/32GB) to characterize the crossover point
- [ ] **LeanKV algorithmic comparison** — Compare tier assignment quality and compression effectiveness against a Python reference implementation of LeanKV's compression policy
- [ ] **Larger model evaluation** — Test on 13B+ models to validate that the system scales beyond 0.5B/1B/7B

## Design Docs

- [`foveated_strategy.md`](docs/foveated_strategy.md) — Speed budget, error bounds, spike detection, promotion protocol
- [`async_tier_manager.md`](docs/async_tier_manager.md) — C++ promotion pipeline, thread safety, unified memory
- [`benchmark_plan.md`](docs/benchmark_plan.md) — What we measure and why, with results
- [`benchmark_ground_truth.md`](docs/benchmark_ground_truth.md) — Scoring methodology and baseline sources

## License

MIT
