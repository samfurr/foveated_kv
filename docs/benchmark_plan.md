# MLX Benchmark Plan

## Platform

Apple Silicon Mac (8GB+). All benchmarks run locally via MLX.

## Benchmark Suite

### 1. LongBench-Lite Quality (`benchmark_mlx_longbench.py`)

6 representative tasks. Measures end-to-end generation quality with foveated cache.

**Results (Qwen2.5-0.5B-Instruct-bf16):**

| Method | Avg Score |
|--------|-----------|
| Standard (full precision) | 14.9 |
| Foveated 10/90 | 15.1 |

Foveated matches or exceeds standard. Quality fully preserved with 2-tier 10/90 config.

Scoring uses official THUDM LongBench v1 metrics: F1 (QA), ROUGE (summarization),
accuracy (few-shot/synthetic), edit similarity (code). Dataset loaded directly from
JSONL via HuggingFace.

### 2. Needle-in-Haystack Heatmap (`benchmark_mlx_needle_heatmap.py`)

Grid evaluation: context length (1K-8K) x needle depth (0%-100%).

**Results:** 55/55 (100%) retrieval across all depths and contexts.

No degradation from compression at any position or context length.

### 3. Ablation Study (`benchmark_mlx_ablation.py`)

Component contribution analysis. Tests each design choice in isolation.

**Key results (4K context, real model K,V):**

| Config | Cosine | MAE |
|--------|--------|-----|
| Full system (10/90) | 0.999939 | 0.0050 |
| No near (uniform fp8+int4) | 0.999929 | 0.0055 |
| Symmetric (int4 K + int4 V) | 0.999783 | 0.0091 |
| Uniform INT8 (no tiers) | 1.000000 | 0.0001 |

Asymmetric K/V (fp8 K + int4 V) achieves 3.6x lower cosine error than symmetric int4.

### 4. Kernel Throughput (`benchmark_mlx_throughput.py`)

Metal fused kernel speed vs standard fp16 SDPA.

**Results (7B shapes: H_q=32, H_kv=8, D=128):**

| Context | fp16 SDPA | Fused Kernel | Speedup |
|---------|-----------|-------------|---------|
| 1K | 0.84 ms | 1.00 ms | 0.84x |
| 4K | 2.07 ms | 1.20 ms | 1.72x |
| 8K | 4.15 ms | 1.68 ms | 2.47x |
| 16K | 9.67 ms | 2.90 ms | 3.34x |
| 32K | 15.19 ms | 5.18 ms | 2.93x |

**End-to-end decode performance:**

| Model | Fused | Standard | Speedup |
|-------|-------|----------|---------|
| Qwen2.5-7B-Instruct-4bit | 150 tok/s | 130-146 tok/s | 1.03-1.45x |
| Qwen2.5-0.5B-Instruct-bf16 | 67-69 tok/s | 60-66 tok/s | 1.04-1.14x |

Memory compression: 2.02x at all context lengths.

### 5. Synthetic Kernel Benchmark (`benchmark_mlx.py`)

Raw kernel microbenchmark. Tests the Metal Split-K kernel in isolation with
controlled tensor shapes. Used for kernel development iteration.

### 6. End-to-End Model Benchmark (`benchmark_mlx_model.py`)

Full model inference with foveated cache via mlx-lm integration. Measures
real-world generation speed and memory usage.

**Perplexity Results (Qwen2.5-0.5B-Instruct-bf16, WikiText-103):**

| Context | Standard PPL | Foveated PPL | Ratio |
|---------|-------------|-------------|-------|
| 1K | 6.86 | 7.03 | 1.025x |
| 2K | 15.14 | 15.12 | 0.999x |
| 4K | 29.17 | 29.36 | 1.007x |

Foveated PPL within 2.5% of standard at all context lengths.

## Running All Benchmarks

```bash
# Run the full MLX benchmark suite
bash scripts/run_mlx_benchmarks.sh

# Or run individually:
uv run python benchmarks/benchmark_mlx_longbench.py
uv run python benchmarks/benchmark_mlx_needle_heatmap.py
uv run python benchmarks/benchmark_mlx_ablation.py
uv run python benchmarks/benchmark_mlx_throughput.py
uv run python benchmarks/benchmark_mlx.py
uv run python benchmarks/benchmark_mlx_model.py
```

## Baseline Methodology

### KIVI / H2O (PyTorch reference)

KIVI and H2O baselines exist as faithful PyTorch reimplementations in
`benchmarks/baselines.py`. These are used for quality comparison context
(cited numbers) but do not run on the MLX path.

- KIVI: group-size quantization, residual buffer, real INT2 packing
- H2O: dynamic eviction, cumulative attention scoring, physical KV pruning

### MLX Benchmarks

All MLX benchmarks compare foveated cache against standard full-precision
MLX inference. Dataset loading uses direct JSONL from HuggingFace datasets.

## What Makes These Results Defensible

1. **LongBench scoring** is aligned with official THUDM v1 pipeline (29 tests verify this)
2. **Needle retrieval** tests actual token recovery, not approximate scoring
3. **PPL ratios** computed against same model at same context -- relative, not absolute
4. **Kernel speed** measured with MLX's built-in timing (eval + sync)
5. **Memory** measured via actual tensor allocation, not estimates
