"""
Generate paper figures from benchmark results.

Reads JSON results from results/ directory, produces publication-quality charts.
Run AFTER run_all_benchmarks.sh completes.

Usage: python scripts/generate_charts.py
"""

import json
import os
import sys

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError:
    print("pip install matplotlib numpy")
    sys.exit(1)

RESULTS_DIR = "results"
CHARTS_DIR = "results/charts"
os.makedirs(CHARTS_DIR, exist_ok=True)

# Paper-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox_inches': 'tight',
})


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  Skipping {filename} (not found)")
        return None
    with open(path) as f:
        return json.load(f)


def fig1_ruler_accuracy():
    """Figure 1: RULER accuracy vs context length (line plot)."""
    data = load_json("ruler.json")
    if not data:
        return

    fig, axes = plt.subplots(1, len(data), figsize=(4 * len(data), 4), sharey=True)
    if len(data) == 1:
        axes = [axes]

    colors = {'Standard': '#2196F3', 'Foveated 5/25': '#4CAF50', 'Foveated 2/18': '#FF9800'}

    for ax, (subtask, ctx_data) in zip(axes, data.items()):
        for method in list(list(ctx_data.values())[0].keys()):
            contexts = sorted([int(c) for c in ctx_data.keys()])
            accs = [ctx_data[str(c)].get(method, 0) for c in contexts]
            color = colors.get(method, '#9E9E9E')
            ax.plot(contexts, accs, 'o-', label=method, color=color, linewidth=2, markersize=6)
        ax.set_title(subtask)
        ax.set_xlabel('Context Length')
        ax.set_xscale('log', base=2)
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Accuracy (%)')
    axes[-1].legend(loc='lower left')
    fig.suptitle('RULER: Retrieval Accuracy vs Context Length', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'fig1_ruler.png'))
    plt.close()
    print("  Generated fig1_ruler.png")


def fig2_needle_heatmap():
    """Figure 2: Needle-in-Haystack heatmap (depth × context)."""
    data = load_json("needle_heatmap.json")
    if not data:
        return

    methods = list(data.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        entries = data[method]
        contexts = sorted(set(e['context'] for e in entries))
        depths = sorted(set(e['depth'] for e in entries))

        grid = np.zeros((len(depths), len(contexts)))
        for e in entries:
            ci = contexts.index(e['context'])
            di = depths.index(e['depth'])
            grid[di, ci] = 1.0 if e['found'] else 0.0

        cmap = mcolors.LinearSegmentedColormap.from_list('rg', ['#f44336', '#4CAF50'])
        ax.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(contexts)))
        ax.set_xticklabels([f'{c//1024}K' for c in contexts], rotation=45)
        ax.set_yticks(range(len(depths)))
        ax.set_yticklabels([f'{d:.0%}' for d in depths])
        ax.set_xlabel('Context Length')
        ax.set_ylabel('Needle Depth')
        ax.set_title(method)

    fig.suptitle('Needle-in-Haystack Retrieval', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHARTS_DIR, 'fig2_needle_heatmap.png'))
    plt.close()
    print("  Generated fig2_needle_heatmap.png")


def fig3_ablation():
    """Figure 3: Ablation study (bar chart)."""
    # Parse from text output since ablation doesn't save JSON consistently
    path = os.path.join(RESULTS_DIR, "ablation.txt")
    if not os.path.exists(path):
        print("  Skipping ablation chart (no results)")
        return

    # Simple placeholder — ablation data format varies
    print("  Ablation chart: parse results/ablation.txt manually for now")


def fig4_throughput():
    """Figure 4: Throughput comparison table (text-based, not a chart)."""
    path = os.path.join(RESULTS_DIR, "throughput.txt")
    if not os.path.exists(path):
        print("  Skipping throughput (no results)")
        return
    print("  Throughput table: see results/throughput.txt (copy to paper directly)")


def fig5_kernel_speed():
    """Figure 5: Kernel speed comparison."""
    path = os.path.join(RESULTS_DIR, "kernel_speed.txt")
    if not os.path.exists(path):
        print("  Skipping kernel speed chart (no results)")
        return
    print("  Kernel speed table: see results/kernel_speed.txt")


def main():
    print("=" * 60)
    print("  Generating paper figures from benchmark results")
    print(f"  Results dir: {RESULTS_DIR}")
    print(f"  Charts dir: {CHARTS_DIR}")
    print("=" * 60)

    fig1_ruler_accuracy()
    fig2_needle_heatmap()
    fig3_ablation()
    fig4_throughput()
    fig5_kernel_speed()

    print(f"\nCharts saved to {CHARTS_DIR}/")
    print("Tables (throughput, kernel speed, LongBench) are text — copy from results/*.txt")


if __name__ == "__main__":
    main()
