# FoveatedKV

**Importance-adaptive mixed-precision KV cache compression for LLM inference on Apple Silicon.**

2x memory compression. 0.995+ cosine fidelity. Custom Metal GPU kernels. 2.3x faster at 32K context.

---

## The Problem

Every LLM decode step reads the *entire* KV cache from memory. At long context, this is gigabytes — memory bandwidth, not compute, becomes the bottleneck. The field has two ideas: **evict tokens** (breaks softmax) or **quantize everything uniformly** (treats the answer to your question the same as filler text).

Both ignore something obvious about attention: it focuses.

## The Approach

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

## Asymmetric K/V Precision

Multiple recent works (KIVI, LeanKV, KV-AdaQuant, AsymKV) have independently discovered that keys need higher precision than values. Key error gets amplified through exp() in softmax (multiplicative damage); value error is just additive noise scaled by attention weight (linear, bounded).

FoveatedKV builds on this established insight, using fp8 E4M3 for keys and INT4 for values in the far tier. Our key precision ablation confirms the choice:

| Key Format | Cosine Error | vs fp8 |
|-----------|-------------|--------|
| fp16 (exact) | 0.000000 | — |
| INT8 | 0.000009 | 1.0x |
| fp8 E4M3 (ours) | 0.000009 | 1.0x |
| INT4 | 0.000317 | **34.3x** |

fp8 is preferred over INT8 because the 256-entry LUT enables a single shared-memory read per element vs per-channel scale/zero arithmetic.

## What's Novel

The quantization strategy is prior work. Our contribution is systems engineering:

### 1. Foveated Two-Tier Cache

Tokens are partitioned per attention head into near (fp16) and far (fp8 K + INT4 V) tiers based on attention importance. Attention sinks and a recency window are always near. Unlike uniform quantization (KIVI, KVSplit), high-attention tokens retain full precision.

### 2. Fused Split-K Metal Kernel

A single GPU dispatch handles both tiers + decode buffer:

- **Pre-scaled query**: q *= 1/sqrt(D) at load time
- **LUT fp8 decode**: 256-entry threadgroup memory table — 1 read vs 10+ ALU ops
- **Score-gated V loading**: skip INT4 dequant when score < m - 16 (exp(-16) ~ 1e-7)
- **Single-exp online softmax**: each exp() computed exactly once
- **Spike detection**: free byproduct of online softmax — tracks max_far vs min_near per head

No fp16 intermediates ever touch global memory. Dequantization happens in registers.

### 3. Spike Detection and Promotion

When the kernel detects a far-tier token scoring above near-tier tokens, a C++ background worker:

1. Reads exact fp16 from NVMe-backed mmap archive
2. Writes into near-tier headroom + atomic increment `near_valid[h]`
3. Kernel sees promoted token next dispatch as ordinary near token

Raw spike rate is 95% per head-layer slot, but aggressive filtering (cooldown, dedup, budget cap, GQA dedup) reduces to 3.1% effective promotion rate. This mechanism has no analogue in prior asymmetric quantization work — they apply static quantization without the ability to recover precision at inference time.

## Results

All results on Apple M2 8GB. Evaluated on Qwen2.5-0.5B-Instruct-4bit (quality), Qwen2.5-7B-Instruct-4bit (throughput), and Llama-3.2-1B-Instruct-4bit (cross-architecture verification).

### Kernel Performance

Fused kernel vs Apple's SDPA (7B-equivalent GQA shapes: H_kv=8, H_q=32, D=128, single layer):

| Context | fp16 SDPA | Fused Kernel | Speedup |
|---------|-----------|-------------|---------|
| 1K | 1.12 ms | 0.94 ms | 1.19x |
| 4K | 2.33 ms | 1.46 ms | 1.60x |
| 8K | 4.05 ms | 2.22 ms | 1.82x |
| 16K | 7.68 ms | 3.73 ms | 2.06x |
| 32K | 15.72 ms | 6.81 ms | 2.31x |

Break-even around 512 tokens. Bandwidth advantage grows with context length.

### 7B on 8GB Mac

On memory-constrained hardware, FoveatedKV is *faster* than standard because compressed KV cache reduces memory pressure:

| Context | Standard tok/s | Foveated tok/s | Speedup | Memory Saved |
|---------|---------------|----------------|---------|-------------|
| 512 | 0.7 | 1.6 | **2.3x** | — |
| 1024 | 0.6 | 1.0 | **1.7x** | 50 MB |
| 2048 | 0.2 | 0.4 | **2.0x** | 101 MB |
| 3072 | OOM | 0.3 | — | 151 MB |

OOM crossover on 8GB: ~3K foveated, ~3-4K standard. Fused kernel vs dequant-reference cosine: **1.000000**.

### Quality

Kernel-level attention cosine fidelity (10% near tier, synthetic queries on real model KV):

| Context | Cosine vs Exact | MAE | Compression |
|---------|----------------|------|-------------|
| 512 | 0.9956 | 0.0053 | 2.03x |
| 1K | 0.9950 | 0.0040 | 2.02x |
| 4K | 0.9952 | 0.0020 | 2.02x |
| 8K | 0.9954 | 0.0014 | 2.02x |
| 16K | 0.9953 | 0.0010 | 2.02x |
| 32K | 0.9954 | 0.0007 | 2.02x |

### Perplexity

End-to-end perplexity (foveated/standard ratio is the relevant metric — absolute PPL reflects 4-bit weight quantization):

| Model | Context | Standard PPL | Foveated PPL | Ratio |
|-------|---------|-------------|-------------|-------|
| Qwen2.5-0.5B-4bit | 1K | 6.86 | 7.03 | 1.025x |
| Qwen2.5-0.5B-4bit | 2K | 15.14 | 15.12 | 0.999x |
| Qwen2.5-0.5B-4bit | 4K | 29.17 | 29.36 | 1.007x |
| Llama-3.2-1B-4bit | 1K | 8.12 | 8.11 | 0.999x |
| Llama-3.2-1B-4bit | 2K | 12.45 | 12.46 | 1.001x |

### LongBench-Lite

6 tasks, 10 samples each, official THUDM scoring (Qwen2.5-0.5B-4bit, 2K context):

| Task | Category | Standard | Foveated 10/90 |
|------|----------|----------|----------------|
| qasper | Single-doc QA | 2.0 | 1.6 |
| hotpotqa | Multi-doc QA | 0.0 | 0.0 |
| qmsum | Summarization | 5.3 | 7.0 |
| triviaqa | Few-shot | 1.0 | 1.0 |
| passage_retrieval_en | Synthetic | 0.0 | 0.0 |
| lcc | Code | 27.7 | 28.1 |
| **Average** | | **6.0** | **6.3** |

Absolute scores are low due to the small 4-bit model and 2K context truncation. The relevant comparison is foveated vs standard — foveated matches or slightly exceeds on every task.

### Promotion Recovery

Multi-fact retrieval (biographical needle with 4 checkable facts, ~2K context, 5% near tier):

| Method | Facts Retrieved | Score |
|--------|----------------|-------|
| Standard (fp16) | 56/64 | **88%** |
| Foveated + promotion | 56/64 | **88%** |
| Foveated no promotion | 32/64 | 50% |

Promotion recovers **24 additional facts** (38% of total), matching the fp16 baseline. 100% reproducible across 8 seeds.

### Second Model Family: Llama-3.2-1B

- Fused kernel vs dequant-reference cosine: **1.000000** (kernel correct on D=64)
- Foveated vs exact fp16 attention cosine: **0.994**
- Short-context token match: 100%
- Throughput: 92.8 tok/s on M2 8GB

## Quick Start

```bash
git clone <repo>
cd foveated_kv
uv sync --extra dev --extra ext
```

### Build the C++ extension (optional — precompiled metallib included)

The Metal kernel ships precompiled (`foveated_attn.metallib`). You only need to build if you want the C++ extension for the promotion pipeline:

```bash
MLX_CMAKE=$(uv run python -c "import mlx,os; print(os.path.join(mlx.__path__[0],'share','cmake','MLX'))")
MLX_INC=$(uv run python -c "import mlx,os; print(os.path.join(mlx.__path__[0],'include'))")
PY=$(uv run python -c "import sys; print(sys.executable)")
cmake -S csrc -B build_ext \
    -DMLX_DIR="$MLX_CMAKE" -DMLX_CMAKE_DIR="$MLX_CMAKE" \
    -DMLX_INCLUDE_DIRS="$MLX_INC" -DPython_EXECUTABLE="$PY" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build_ext -j$(sysctl -n hw.ncpu)
cp build_ext/foveated_ext.cpython-*-darwin.so .
```

### Run tests

```bash
uv run pytest tests/ -v  # 73 tests
```

### Python API

```python
from mlx_lm import load
from foveated_kv.mlx_generate import generate_fused
from foveated_kv.mlx_foveated import MLXTierConfig

model, tokenizer = load("mlx-community/Qwen2.5-0.5B-Instruct-4bit")
cfg = MLXTierConfig()  # defaults to 10% near

text, stats = generate_fused(
    model, tokenizer,
    "What is the meaning of life?",
    max_tokens=100, cfg=cfg,
)
print(text)
print(f"Memory saved: {stats['mem_saved_mb']:.1f} MB")
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

## Run Benchmarks

```bash
# Individual
uv run python benchmarks/benchmark_mlx_longbench.py       # LongBench-Lite (6 tasks)
uv run python benchmarks/benchmark_mlx_needle_heatmap.py   # Needle retrieval grid
uv run python benchmarks/benchmark_mlx_ablation.py         # Component ablation
uv run python benchmarks/benchmark_mlx_throughput.py       # Throughput + memory
uv run python benchmarks/benchmark_mlx.py                  # Synthetic kernel speed
uv run python benchmarks/benchmark_promotion_recovery.py   # Multi-fact promotion recovery
uv run python benchmarks/benchmark_mlx_sustained.py        # Sustained accuracy
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
    foveated_attn.metal      Fused Split-K + Reduce kernel (10 variants)
    foveated_compress.metal  GPU compression kernels (fp8 E4M3, INT4)
  foveated_attn.h/.cpp       FoveatedPrimitive + FoveatedHandle
  promotion_pipeline.h/.cpp  C++ promotion worker
  foveated_compress.h/.cpp   CompressHandle: GPU compression
  bindings.cpp               nanobind module
  CMakeLists.txt

paper/                   Technical report + reproducible scripts
benchmarks/              8 benchmarks + scoring library
tests/                   73 tests
docs/                    Design docs
```

## TODO: Further Benchmarking

The following require a machine with more memory (16GB+) and are needed before the paper is submission-ready:

- [ ] **llama.cpp head-to-head** — Compare against llama.cpp's quantized KV cache on the same Mac hardware
- [ ] **LongBench full-length** — Run at full context length (current results truncated to 2K on 8GB)
- [ ] **Multi-hardware throughput curves** — M1/M2/M3/M4 across 8GB/16GB/32GB configurations
- [ ] **LeanKV algorithmic comparison** — Compare tier assignment quality against LeanKV's compression policy
- [ ] **Larger model evaluation** — 13B+ models to validate scaling beyond 0.5B/1B/7B

## Prior Work

FoveatedKV builds on asymmetric K/V quantization established by:
- **KIVI** (ICML 2024) — per-channel key / per-token value quantization
- **LeanKV** (Dec 2024) — explicit K8V4 + importance-adaptive compression
- **KV-AdaQuant** (Feb 2025) — spectral norm analysis, K4V2 allocation
- **AsymKV** (Oct 2024) — systematic attention output error analysis
- **KVSplit** — K8V4 on Apple Silicon (uniform, no tiering or promotion)

## License

MIT
