# MLX Benchmark Plan

## Platform

Apple Silicon Mac (8GB+). All benchmarks run locally via MLX.

## Benchmark Suite

### 1. LongBench-Lite Quality (`benchmark_mlx_longbench.py`)

6 representative tasks. Measures end-to-end generation quality with foveated cache.

**Results (Qwen2.5-0.5B-Instruct-bf16):**

| Method | Avg Score |
|--------|-----------|
| Standard (full precision) | 9.8 |
| Foveated 5/25/70 | 9.7 |
| Foveated 2/18/80 | 9.7 |

Within 0.1 points. Quality preserved across tier configurations.

Scoring uses official THUDM LongBench v1 metrics: F1 (QA), ROUGE (summarization),
accuracy (few-shot/synthetic), edit similarity (code). Dataset loaded directly from
JSONL via HuggingFace.

### 2. Needle-in-Haystack Heatmap (`benchmark_mlx_needle_heatmap.py`)

Grid evaluation: context length (2K-8K) x needle depth (0%-100%).

**Results:** 36/36 (100%) retrieval across all depths and contexts.

No degradation from compression at any position or context length.

### 3. Ablation Study (`benchmark_mlx_ablation.py`)

Component contribution analysis. Tests each design choice in isolation.

**Key results:**
- Asymmetric K/V is the critical component (130x error without it)
- Foveal tier provides 1.5x improvement over uniform INT8
- PPL ratios: 0.998x (1K), 0.993x (2K), 1.003x (4K) -- error does not accumulate

Configurations tested:
- Full system (all components)
- No asymmetric K/V (symmetric INT4 for far tier)
- No foveal tier (all tokens quantized uniformly)
- Different tier splits (5/25/70, 2/18/80, 10/30/60)

### 4. Kernel Throughput (`benchmark_mlx_throughput.py`)

Metal fused kernel speed vs standard fp16 SDPA.

**Results (synthetic, 7B shapes):**

| Context | Speedup vs fp16 SDPA |
|---------|---------------------|
| 4K | 1.49x |
| 8K | 1.46x |
| 32K | 3.28x (original shapes) |

Memory compression: 2.21x at all context lengths.

### 5. Synthetic Kernel Benchmark (`benchmark_mlx.py`)

Raw kernel microbenchmark. Tests the Metal Split-K kernel in isolation with
controlled tensor shapes. Used for kernel development iteration.

### 6. End-to-End Model Benchmark (`benchmark_mlx_model.py`)

Full model inference with foveated cache via mlx-lm integration. Measures
real-world generation speed and memory usage.

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

1. **LongBench scoring** is aligned with official THUDM v1 pipeline (29 scoring tests)
2. **Needle retrieval** tests actual token recovery, not approximate scoring
3. **PPL ratios** computed against same model at same context -- relative, not absolute
4. **Kernel speed** measured with MLX's built-in timing (eval + sync)
5. **Memory** measured via actual tensor allocation, not estimates
