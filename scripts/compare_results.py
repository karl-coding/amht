#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two AMHT benchmark result files")
    parser.add_argument("--amht", required=True, help="Path to AMHT eval JSON")
    parser.add_argument("--baseline", required=True, help="Path to baseline eval JSON")
    parser.add_argument("--baseline-name", default="transformer", help="Baseline label")
    return parser.parse_args()


def load_results(path: str) -> dict[str, dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return {item["task"]: item for item in data}


def fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    amht = load_results(args.amht)
    baseline = load_results(args.baseline)

    rows = [
        (
            "throughput_tokens_per_second",
            amht.get("throughput", {}).get("tokens_per_second"),
            baseline.get("throughput", {}).get("tokens_per_second"),
        ),
        (
            "throughput_ms_per_step",
            amht.get("throughput", {}).get("milliseconds_per_step"),
            baseline.get("throughput", {}).get("milliseconds_per_step"),
        ),
        (
            "niah_mean_accuracy",
            amht.get("niah", {}).get("mean_accuracy"),
            baseline.get("niah", {}).get("mean_accuracy"),
        ),
    ]

    amht_scaling = {item["seq_len"]: item for item in amht.get("scaling", {}).get("results", [])}
    baseline_scaling = {item["seq_len"]: item for item in baseline.get("scaling", {}).get("results", [])}
    for seq_len in sorted(set(amht_scaling) | set(baseline_scaling)):
        rows.append(
            (
                f"scaling_tps_{seq_len}",
                amht_scaling.get(seq_len, {}).get("tokens_per_second"),
                baseline_scaling.get(seq_len, {}).get("tokens_per_second"),
            )
        )

    print(f"{'metric':32} {'amht':>14} {args.baseline_name:>14} {'delta':>14}")
    print("-" * 78)
    for name, amht_value, baseline_value in rows:
        delta = None
        if amht_value is not None and baseline_value is not None:
            delta = amht_value - baseline_value
        print(f"{name:32} {fmt(amht_value):>14} {fmt(baseline_value):>14} {fmt(delta):>14}")


if __name__ == "__main__":
    main()
