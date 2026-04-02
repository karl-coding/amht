#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed benchmark outputs")
    parser.add_argument("--glob-amht", required=True, help="Glob for AMHT eval JSON files")
    parser.add_argument("--glob-baseline", required=True, help="Glob for baseline eval JSON files")
    parser.add_argument("--baseline-name", default="transformer", help="Baseline label")
    return parser.parse_args()


def load_results(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return {item["task"]: item for item in data}


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def fmt_pair(mean: float | None, std: float | None) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.4f} +- {std:.4f}"


def collect_metric(runs: list[dict[str, dict]], task: str, field: str) -> list[float]:
    values = []
    for run in runs:
        item = run.get(task, {})
        value = item.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def collect_scaling_metric(runs: list[dict[str, dict]], seq_len: int, field: str) -> list[float]:
    values = []
    for run in runs:
        scaling = run.get("scaling", {}).get("results", [])
        for item in scaling:
            if int(item.get("seq_len", -1)) == seq_len and isinstance(item.get(field), (int, float)):
                values.append(float(item[field]))
    return values


def collect_niah_depth_metric(runs: list[dict[str, dict]], depth_index: int) -> list[float]:
    values = []
    for run in runs:
        niah = run.get("niah", {})
        by_depth = niah.get("accuracy_by_depth", [])
        if depth_index < len(by_depth) and isinstance(by_depth[depth_index], (int, float)):
            values.append(float(by_depth[depth_index]))
    return values


def print_markdown_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    print()
    print(title)
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    args = parse_args()
    amht_runs = [load_results(path) for path in sorted(glob.glob(args.glob_amht))]
    baseline_runs = [load_results(path) for path in sorted(glob.glob(args.glob_baseline))]

    if not amht_runs:
        raise SystemExit(f"No AMHT results matched {args.glob_amht}")
    if not baseline_runs:
        raise SystemExit(f"No baseline results matched {args.glob_baseline}")

    rows = [
        ("throughput_tokens_per_second", collect_metric(amht_runs, "throughput", "tokens_per_second"), collect_metric(baseline_runs, "throughput", "tokens_per_second")),
        ("throughput_ms_per_step", collect_metric(amht_runs, "throughput", "milliseconds_per_step"), collect_metric(baseline_runs, "throughput", "milliseconds_per_step")),
        ("niah_mean_accuracy", collect_metric(amht_runs, "niah", "mean_accuracy"), collect_metric(baseline_runs, "niah", "mean_accuracy")),
    ]

    print(f"{'metric':32} {'amht mean+-std':>24} {args.baseline_name + ' mean+-std':>24}")
    print("-" * 84)
    for name, amht_values, baseline_values in rows:
        amht_pair = mean_std(amht_values)
        baseline_pair = mean_std(baseline_values)
        print(f"{name:32} {fmt_pair(*amht_pair):>24} {fmt_pair(*baseline_pair):>24}")

    first_amht_niah = amht_runs[0].get("niah", {})
    depths = list(first_amht_niah.get("needle_depths", []))
    if depths:
        print()
        print("Per-depth NIAH")
        print(f"{'depth':10} {'amht mean+-std':>24} {args.baseline_name + ' mean+-std':>24}")
        print("-" * 64)
        niah_rows = []
        for index, depth in enumerate(depths):
            amht_pair = mean_std(collect_niah_depth_metric(amht_runs, index))
            baseline_pair = mean_std(collect_niah_depth_metric(baseline_runs, index))
            print(f"{depth:<10} {fmt_pair(*amht_pair):>24} {fmt_pair(*baseline_pair):>24}")
            niah_rows.append([str(depth), fmt_pair(*amht_pair), fmt_pair(*baseline_pair)])
        print_markdown_table(
            "Markdown NIAH Table",
            ["Depth", "AMHT mean+-std", f"{args.baseline_name} mean+-std"],
            niah_rows,
        )

    scaling_lengths = sorted(
        {
            int(item.get("seq_len"))
            for run in amht_runs + baseline_runs
            for item in run.get("scaling", {}).get("results", [])
            if "seq_len" in item
        }
    )
    if scaling_lengths:
        print()
        print("Per-length Scaling Throughput")
        print(f"{'seq_len':10} {'amht mean+-std':>24} {args.baseline_name + ' mean+-std':>24}")
        print("-" * 64)
        scaling_rows = []
        for seq_len in scaling_lengths:
            amht_pair = mean_std(collect_scaling_metric(amht_runs, seq_len, "tokens_per_second"))
            baseline_pair = mean_std(collect_scaling_metric(baseline_runs, seq_len, "tokens_per_second"))
            print(f"{seq_len:<10} {fmt_pair(*amht_pair):>24} {fmt_pair(*baseline_pair):>24}")
            scaling_rows.append([str(seq_len), fmt_pair(*amht_pair), fmt_pair(*baseline_pair)])
        print_markdown_table(
            "Markdown Scaling Table",
            ["Seq Len", "AMHT mean+-std", f"{args.baseline_name} mean+-std"],
            scaling_rows,
        )


if __name__ == "__main__":
    main()
