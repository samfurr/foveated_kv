"""
LongBench-Lite quality evaluation on MLX for FoveatedKV paper.

6 representative tasks (one per LongBench category) with official THUDM scoring.
Runs on 8GB Apple Silicon with Qwen2.5-0.5B-bf16.

Usage:
  uv run python benchmarks/benchmark_mlx_longbench.py
  uv run python benchmarks/benchmark_mlx_longbench.py --max-samples 5 --tasks qasper,hotpotqa
"""

import argparse
import json
import os
import sys
import time

import mlx.core as mx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.benchmark_longbench import (
    DATASET2MAXLEN,
    DATASET2PROMPT,
    score_task,
)
from mipmap_kv.mlx_foveated import MLXTierConfig
from mipmap_kv.mlx_generate import generate_fused, _generate_short

# 6 tasks, one per LongBench category
DEFAULT_TASKS = [
    "qasper",                # single-doc QA (F1)
    "hotpotqa",              # multi-doc QA (F1)
    "qmsum",                 # summarization (ROUGE-L)
    "triviaqa",              # few-shot (F1)
    "passage_retrieval_en",  # synthetic (retrieval)
    "lcc",                   # code (edit similarity)
]

TASK_CATEGORY = {
    "qasper": "Single-doc QA",
    "hotpotqa": "Multi-doc QA",
    "qmsum": "Summarization",
    "triviaqa": "Few-shot",
    "passage_retrieval_en": "Synthetic",
    "lcc": "Code",
}


def load_longbench_task(task_name: str, max_samples: int = 20):
    """Load samples from THUDM/LongBench dataset."""
    data_dir = _ensure_longbench_data()
    path = os.path.join(data_dir, f"{task_name}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"LongBench task {task_name} not found at {path}")
    samples = []
    with open(path) as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            item = json.loads(line)
            samples.append({
                "context": item["context"],
                "input": item["input"],
                "answers": item["answers"] if isinstance(item["answers"], list) else [item["answers"]],
                "all_classes": item.get("all_classes", []),
            })
    return samples


_longbench_data_dir = None

def _ensure_longbench_data() -> str:
    """Download and extract THUDM/LongBench data.zip if needed."""
    global _longbench_data_dir
    if _longbench_data_dir and os.path.exists(_longbench_data_dir):
        return _longbench_data_dir

    from huggingface_hub import hf_hub_download
    import zipfile

    cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "longbench")
    data_dir = os.path.join(cache_dir, "data")
    if os.path.exists(data_dir) and any(f.endswith(".jsonl") for f in os.listdir(data_dir)):
        _longbench_data_dir = data_dir
        return data_dir

    os.makedirs(cache_dir, exist_ok=True)
    zip_path = hf_hub_download("THUDM/LongBench", "data.zip", repo_type="dataset")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_dir)
    _longbench_data_dir = data_dir
    return data_dir


def format_prompt(task_name: str, sample: dict) -> str:
    template = DATASET2PROMPT[task_name]
    return template.format(context=sample["context"], input=sample["input"])


def run_task(model, tokenizer, task_name: str, samples: list,
             method: str, cfg: MLXTierConfig, max_context: int) -> dict:
    """Run one task with one method, return scored results."""
    max_gen = DATASET2MAXLEN.get(task_name, 64)
    predictions = []
    references = []

    for i, sample in enumerate(samples):
        prompt = format_prompt(task_name, sample)
        # Truncate to max context
        tokens = tokenizer.encode(prompt)
        if len(tokens) > max_context:
            tokens = tokens[:max_context]
            prompt = tokenizer.decode(tokens)

        try:
            if method == "standard":
                text, _ = _generate_short(model, tokenizer, prompt, max_tokens=max_gen)
            else:
                text, _ = generate_fused(
                    model, tokenizer, prompt, max_tokens=max_gen, cfg=cfg,
                    enable_promotion=False,
                )
        except Exception as e:
            text = ""
            print(f"    Sample {i} error: {e}")

        predictions.append(text)
        references.append(sample["answers"])

    all_classes = samples[0].get("all_classes", []) if samples else []
    score = score_task(task_name, predictions, references, all_classes)
    return {"task": task_name, "method": method, "score": score, "n_samples": len(samples)}


def main():
    parser = argparse.ArgumentParser(description="LongBench-Lite MLX Evaluation")
    parser.add_argument("--model", default="mlx-community/Qwen2.5-0.5B-Instruct-bf16")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-context", type=int, default=6144)
    parser.add_argument("--foveal-pct", type=float, default=0.05)
    parser.add_argument("--periph-pct", type=float, default=0.25)
    parser.add_argument("--output", default="results/paper/longbench_lite.json")
    args = parser.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model)
    tasks = args.tasks.split(",")
    configs = [
        ("standard", None),
        ("foveated_5_25", MLXTierConfig(foveal_pct=0.05, periph_pct=0.25)),
        ("foveated_2_18", MLXTierConfig(foveal_pct=0.02, periph_pct=0.18)),
    ]

    all_results = {"model": args.model, "max_samples": args.max_samples, "tasks": {}}

    print(f"{'Task':<25} {'Category':<15} {'Standard':>10} {'5/25/70':>10} {'2/18/80':>10}")
    print("-" * 75)

    for task in tasks:
        print(f"Loading {task}...", end=" ", flush=True)
        samples = load_longbench_task(task, args.max_samples)
        print(f"{len(samples)} samples")

        task_results = {}
        for method_name, cfg in configs:
            t0 = time.perf_counter()
            result = run_task(model, tokenizer, task, samples, method_name, cfg, args.max_context)
            elapsed = time.perf_counter() - t0
            result["time_s"] = round(elapsed, 1)
            task_results[method_name] = result

        scores = [task_results[m]["score"] for m, _ in configs]
        cat = TASK_CATEGORY.get(task, "")
        print(f"{task:<25} {cat:<15} {scores[0]:>10.1f} {scores[1]:>10.1f} {scores[2]:>10.1f}")
        all_results["tasks"][task] = task_results

    # Summary averages
    print("\n" + "=" * 75)
    for method_name, _ in configs:
        avg = sum(all_results["tasks"][t][method_name]["score"] for t in tasks) / len(tasks)
        print(f"  {method_name:<20} avg: {avg:.1f}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
