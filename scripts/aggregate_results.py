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


if __name__ == "__main__":
    main()
