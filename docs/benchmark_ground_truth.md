# Benchmark Ground Truth

Last updated: 2026-03-14

## Purpose

This document pins the external benchmark and baseline sources we treat as ground
truth. The goal is to be able to say, honestly, that our evaluations use official
benchmark scoring and faithful baseline implementations.

## Rules

1. Prefer the official paper scoring pipeline over reimplementing from scratch.
2. Keep prompt templates, generation lengths, scoring, and dataset splits aligned
   with the official code.
3. If we deviate, label the result as an approximation.
4. Do not call a baseline "KIVI" or "H2O" unless the behavior matches the
   official method closely enough to defend that naming.

## Official Sources

### LongBench v1

- Paper: `https://aclanthology.org/2024.acl-long.172`
- Official repo: `https://github.com/THUDM/LongBench`
- Dataset: `THUDM/LongBench`

Current repo status:

- `benchmarks/benchmark_mlx_longbench.py` runs LongBench-Lite (6 representative tasks)
- Scoring aligned with official THUDM v1 pipeline:
  - Official scoring functions: qa_f1_score, rouge_score, classification_score,
    code_sim_score, count_score, retrieval_score
  - Official post-processing (first-line extraction for few-shot tasks)
  - 29 unit tests verifying scoring correctness (`tests/test_longbench_scoring.py`)
- Dataset loaded directly from JSONL via HuggingFace
- MLX benchmarks use direct JSONL loading, not the full pred.py pipeline
- Label: `faithful-reimplementation` for scoring, `approximation` for data pipeline

### Needle In A Haystack

- Canonical public repo: `https://github.com/gkamradt/LLMTest_NeedleInAHaystack`

Current repo status:

- `benchmarks/benchmark_mlx_needle_heatmap.py` runs depth x context grid evaluation
- Result: 36/36 (100%) across 2K-8K, all depths
- Label: `approximation` (custom prompt generator, not the paper implementation)

### KIVI Baseline (PyTorch reference only)

- Paper: `https://arxiv.org/abs/2402.02750`
- Official repo: `https://github.com/jy-yuan/KIVI`

Current repo status:

- `benchmarks/baselines.py` is a `faithful-reimplementation`:
  - Per-channel K quantization with group_size along sequence dimension
  - Per-token V quantization with group_size along feature dimension
  - Residual buffer (recent tokens in fp16, periodically flushed and quantized)
  - Real INT2/INT4/INT8 bit-packing via pack_tensor/unpack_tensor
  - Default parameters: bits=2, group_size=32, residual_length=128
- This is PyTorch-only, used for quality comparison context (cited numbers)
- Not run on the MLX path
- Remaining gap: KIVI uses a custom kernel for dequant-free attention; we
  dequantize then use standard SDPA. Numerically equivalent for quality comparison.

### H2O Baseline (PyTorch reference only)

- Paper: NeurIPS 2023 H2O
- Official repo: `https://github.com/FMInference/H2O`

Current repo status:

- `benchmarks/baselines.py` is a `faithful-reimplementation`:
  - Cumulative attention scores accumulated across every decode step
  - Heavy-hitter budget + recent window with physical KV pruning
  - Per-layer independent eviction decisions
  - Default parameters: heavy_ratio=0.10, recent_ratio=0.10
- This is PyTorch-only, used for quality comparison context
- Not run on the MLX path

## MLX Benchmark Methodology

All MLX benchmarks (`benchmark_mlx_*.py`) follow this methodology:

1. **Data loading**: JSONL directly from HuggingFace datasets
2. **Model**: mlx-lm loaded models (e.g., Qwen2.5-0.5B-Instruct-bf16)
3. **Scoring**: Official THUDM scoring functions for LongBench tasks
4. **Timing**: MLX eval + sync for kernel benchmarks
5. **Memory**: Actual tensor allocation measurement
6. **Comparison**: Foveated vs standard full-precision on same model and data

## Allowed Labels

- `exact-official`: runs the official benchmark pipeline directly
- `faithful-reimplementation`: we reimplemented it, matched behavior, validated
- `approximation`: useful for local iteration, not for headline claims

## What We Can Defend

- LongBench-Lite scoring is faithful to THUDM v1 (29 tests verify this)
- KIVI and H2O baselines are faithful PyTorch reimplementations (cited for comparison)
- Needle retrieval is an approximation (custom prompts, not paper implementation)
- Kernel speed numbers are synthetic benchmarks with controlled shapes
- PPL and memory measurements are direct, not estimated
