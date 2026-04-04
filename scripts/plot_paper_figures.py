#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from eval JSON files")
    parser.add_argument("--glob-amht-fast", required=True, help="Glob for AMHT-Fast eval JSON files")
    parser.add_argument("--glob-amht-accurate", required=True, help="Glob for AMHT-Accurate eval JSON files")
    parser.add_argument("--glob-transformer", required=True, help="Glob for Transformer eval JSON files")
    parser.add_argument("--outdir", default="figures", help="Output directory for generated figures")
    return parser.parse_args()


def load_eval(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return {item["task"]: item for item in data}


def load_runs(pattern: str) -> list[dict[str, dict]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No results matched {pattern}")
    return [load_eval(path) for path in paths]


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("No values to aggregate")
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    return mean, float(arr.std(ddof=1))


def collect_metric(runs: list[dict[str, dict]], task: str, field: str) -> list[float]:
    values: list[float] = []
    for run in runs:
        item = run.get(task, {})
        value = item.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def collect_niah_depth(runs: list[dict[str, dict]], depth_index: int) -> list[float]:
    values: list[float] = []
    for run in runs:
        by_depth = run.get("niah", {}).get("accuracy_by_depth", [])
        if depth_index < len(by_depth):
            values.append(float(by_depth[depth_index]))
    return values


def collect_scaling(runs: list[dict[str, dict]], seq_len: int) -> list[float]:
    values: list[float] = []
    for run in runs:
        for item in run.get("scaling", {}).get("results", []):
            if int(item.get("seq_len", -1)) == seq_len:
                values.append(float(item["tokens_per_second"]))
    return values


def save_architecture_placeholder(outdir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")

    boxes = [
        (0.04, 0.38, 0.16, 0.24, "Input Tokens"),
        (0.24, 0.38, 0.16, 0.24, "Block Router"),
        (0.46, 0.63, 0.18, 0.18, "SSM on All Blocks"),
        (0.46, 0.19, 0.18, 0.18, "Sparse Attention\non Routed Blocks"),
        (0.72, 0.38, 0.18, 0.24, "Latent Memory\nRead / Write"),
    ]

    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#e9f2ff", edgecolor="#355c7d", linewidth=1.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.20, 0.50), (0.24, 0.50)),
        ((0.40, 0.50), (0.46, 0.72)),
        ((0.40, 0.50), (0.46, 0.28)),
        ((0.64, 0.72), (0.72, 0.50)),
        ((0.64, 0.28), (0.72, 0.50)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8, color="#355c7d"))

    ax.text(
        0.5,
        0.06,
        "AMHT V2: SSM processes all blocks, sparse attention is routed at block level, and latent memory provides global compressed communication.",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(outdir / "amht_v2_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_niah_by_depth(outdir: Path, amht_runs: list[dict[str, dict]], baseline_runs: list[dict[str, dict]]) -> None:
    depths = [float(x) for x in amht_runs[0]["niah"]["needle_depths"]]
    amht_mean, amht_std, baseline_mean, baseline_std = [], [], [], []

    for idx, _ in enumerate(depths):
        m, s = mean_std(collect_niah_depth(amht_runs, idx))
        amht_mean.append(m)
        amht_std.append(s)
        m, s = mean_std(collect_niah_depth(baseline_runs, idx))
        baseline_mean.append(m)
        baseline_std.append(s)

    x = np.arange(len(depths))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, amht_mean, width, yerr=amht_std, label="AMHT", color="#355c7d", capsize=4)
    ax.bar(x + width / 2, baseline_mean, width, yerr=baseline_std, label="Transformer", color="#f67280", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels([str(depth) for depth in depths])
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel("Needle depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Needle-in-a-Haystack Accuracy by Depth")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "niah_by_depth.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_scaling(outdir: Path, amht_runs: list[dict[str, dict]], baseline_runs: list[dict[str, dict]]) -> None:
    seq_lens = sorted(
        {
            int(item["seq_len"])
            for run in amht_runs + baseline_runs
            for item in run.get("scaling", {}).get("results", [])
        }
    )
    amht_mean, amht_std, baseline_mean, baseline_std = [], [], [], []
    for seq_len in seq_lens:
        m, s = mean_std(collect_scaling(amht_runs, seq_len))
        amht_mean.append(m)
        amht_std.append(s)
        m, s = mean_std(collect_scaling(baseline_runs, seq_len))
        baseline_mean.append(m)
        baseline_std.append(s)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.errorbar(seq_lens, amht_mean, yerr=amht_std, marker="o", linewidth=2, color="#355c7d", label="AMHT")
    ax.errorbar(seq_lens, baseline_mean, yerr=baseline_std, marker="s", linewidth=2, color="#f67280", label="Transformer")
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput Scaling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "throughput_scaling.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_frontier(
    outdir: Path,
    fast_runs: list[dict[str, dict]],
    accurate_runs: list[dict[str, dict]],
    baseline_runs: list[dict[str, dict]],
) -> None:
    def pair(runs: list[dict[str, dict]]) -> tuple[float, float]:
        tps_mean, _ = mean_std(collect_metric(runs, "throughput", "tokens_per_second"))
        acc_mean, _ = mean_std(collect_metric(runs, "niah", "mean_accuracy"))
        return tps_mean, acc_mean

    points = {
        "AMHT-Fast": pair(fast_runs),
        "AMHT-Accurate": pair(accurate_runs),
        "Transformer": pair(baseline_runs),
    }
    colors = {
        "AMHT-Fast": "#355c7d",
        "AMHT-Accurate": "#6c5b7b",
        "Transformer": "#f67280",
    }

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for label, (x, y) in points.items():
        ax.scatter(x, y, s=120, color=colors[label], label=label)
        ax.annotate(label, (x, y), xytext=(8, 6), textcoords="offset points", fontsize=10)

    ax.set_xlabel("Tokens / second")
    ax.set_ylabel("Mean NIAH accuracy")
    ax.set_title("Efficiency-Quality Frontier")
    ax.set_ylim(0.45, 1.05)
    fig.tight_layout()
    fig.savefig(outdir / "efficiency_quality_frontier.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fast_runs = load_runs(args.glob_amht_fast)
    accurate_runs = load_runs(args.glob_amht_accurate)
    baseline_runs = load_runs(args.glob_transformer)
    amht_runs = fast_runs + accurate_runs

    save_architecture_placeholder(outdir)
    plot_niah_by_depth(outdir, amht_runs, baseline_runs)
    plot_scaling(outdir, amht_runs, baseline_runs)
    plot_frontier(outdir, fast_runs, accurate_runs, baseline_runs)

    print(f"generated_figures={outdir}")


if __name__ == "__main__":
    main()
