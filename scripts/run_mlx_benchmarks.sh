#!/bin/bash
# Paper benchmark suite for FoveatedKV on Apple Silicon.
# Run serially (8GB Mac). Total: ~2.5 hours.
set -e

MODEL="mlx-community/Qwen2.5-0.5B-Instruct-bf16"
OUTDIR="results/paper"
mkdir -p "$OUTDIR"

echo "============================================"
echo "FoveatedKV Paper Benchmarks"
echo "Model: $MODEL"
echo "Output: $OUTDIR/"
echo "============================================"

# Phase 1: Fast validation (~10 min)
echo ""
echo ">>> Phase 1: Ablation (validates model + quality measurement)"
uv run python benchmarks/benchmark_mlx_ablation.py \
  --model "$MODEL" --output "$OUTDIR/ablation.json"

# Phase 1b: PPL scaling (~30 min)
echo ""
echo ">>> Phase 1b: Perplexity scaling"
uv run python benchmarks/benchmark_mlx_model.py \
  --model "$MODEL" --ppl \
  --context-len 1024 --eval-len 128 --output "$OUTDIR/ppl_1k.json"
uv run python benchmarks/benchmark_mlx_model.py \
  --model "$MODEL" --ppl \
  --context-len 2048 --eval-len 128 --output "$OUTDIR/ppl_2k.json"
uv run python benchmarks/benchmark_mlx_model.py \
  --model "$MODEL" --ppl \
  --context-len 4096 --eval-len 128 --output "$OUTDIR/ppl_4k.json"

# Phase 2: Medium benchmarks (~65 min)
echo ""
echo ">>> Phase 2: Throughput + Memory"
uv run python benchmarks/benchmark_mlx_throughput.py \
  --model "$MODEL" --output "$OUTDIR/throughput.json"

echo ""
echo ">>> Phase 2b: Needle Heatmap"
uv run python benchmarks/benchmark_mlx_needle_heatmap.py \
  --model "$MODEL" --output "$OUTDIR/needle_heatmap.json"

# Phase 3: Longest benchmark (~40 min)
echo ""
echo ">>> Phase 3: LongBench-Lite"
uv run python benchmarks/benchmark_mlx_longbench.py \
  --model "$MODEL" --output "$OUTDIR/longbench_lite.json"

echo ""
echo "============================================"
echo "All benchmarks complete. Results in $OUTDIR/"
ls -la "$OUTDIR/"
echo "============================================"
