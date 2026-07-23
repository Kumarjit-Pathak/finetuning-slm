"""Visualization utilities: training curves, comparison charts, SVG export."""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for notebooks


def plot_training_curves(metrics: dict, title: str = "Training Loss", save_path: str = None):
    """Plot training loss curve from TrainingMetricsCallback data."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(metrics["steps"], metrics["loss"], label="Train Loss", color="#2563eb")
    if metrics.get("eval_loss"):
        eval_steps = metrics["steps"][: len(metrics["eval_loss"])]
        ax.plot(eval_steps, metrics["eval_loss"], label="Eval Loss", color="#dc2626", linestyle="--")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_technique_comparison_radar(data: dict, save_path: str = None):
    """Create radar chart comparing techniques across multiple dimensions."""
    import numpy as np

    categories = list(next(iter(data.values())).keys())
    N = len(categories)
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(range(len(data)))

    for i, (technique, values) in enumerate(data.items()):
        vals = list(values.values())
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, linestyle="solid", label=technique, color=colors[i])
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.title("Technique Comparison", pad=20)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_memory_comparison(data: dict, save_path: str = None):
    """Bar chart comparing peak memory usage across techniques."""
    fig, ax = plt.subplots(figsize=(10, 5))
    techniques = list(data.keys())
    memory_gb = list(data.values())

    bars = ax.bar(techniques, memory_gb, color="#3b82f6", edgecolor="#1d4ed8", linewidth=0.5)
    ax.set_ylabel("Peak Memory (GB)")
    ax.set_title("Memory Usage by Technique")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, val in zip(bars, memory_gb):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pareto_frontier(points: list[dict], x_key: str = "memory", y_key: str = "quality", save_path: str = None):
    """Plot Pareto frontier of memory vs quality trade-off."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for p in points:
        ax.scatter(p[x_key], p[y_key], s=100, zorder=5)
        ax.annotate(p["name"], (p[x_key], p[y_key]), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)

    ax.set_xlabel(f"{x_key.replace('_', ' ').title()}")
    ax.set_ylabel(f"{y_key.replace('_', ' ').title()}")
    ax.set_title("Memory vs Quality Trade-off")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_id_vs_ood(results: dict, save_path: str = None):
    """Plot in-domain vs out-of-domain performance comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    techniques = list(results.keys())
    id_scores = [r["in_domain"] for r in results.values()]
    ood_scores = [r["out_of_domain"] for r in results.values()]

    x = range(len(techniques))
    width = 0.35
    ax.bar([i - width / 2 for i in x], id_scores, width, label="In-Domain", color="#22c55e")
    ax.bar([i + width / 2 for i in x], ood_scores, width, label="Out-of-Domain", color="#f97316")

    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("In-Domain vs Out-of-Domain Performance")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_svg(fig, path: str):
    """Export a matplotlib figure as SVG."""
    fig.savefig(path, format="svg", bbox_inches="tight")
