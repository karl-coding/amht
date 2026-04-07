#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:  # pragma: no cover
    plt = None
    np = None


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    config: str
    color: str
    marker: str


MODEL_SPECS = {
    "amht_v4_stage2_round7": ModelSpec(
        key="amht_v4_stage2_round7",
        label="AMHT-V4-Stage2-R7",
        config="train/config_amht_v4_stage2_round7.yaml",
        color="#fb923c",
        marker="v",
    ),
    "amht_v4_stage2_round6": ModelSpec(
        key="amht_v4_stage2_round6",
        label="AMHT-V4-Stage2-R6",
        config="train/config_amht_v4_stage2_round6.yaml",
        color="#f97316",
        marker=">",
    ),
    "amht_v4_stage2_round5": ModelSpec(
        key="amht_v4_stage2_round5",
        label="AMHT-V4-Stage2-R5",
        config="train/config_amht_v4_stage2_round5.yaml",
        color="#ea580c",
        marker="P",
    ),
    "amht_v4_stage2_round4": ModelSpec(
        key="amht_v4_stage2_round4",
        label="AMHT-V4-Stage2-R4",
        config="train/config_amht_v4_stage2_round4.yaml",
        color="#c2410c",
        marker="p",
    ),
    "amht_v4_stage2_round3": ModelSpec(
        key="amht_v4_stage2_round3",
        label="AMHT-V4-Stage2-R3",
        config="train/config_amht_v4_stage2_round3.yaml",
        color="#b45309",
        marker="8",
    ),
    "amht_v4_stage2_round2": ModelSpec(
        key="amht_v4_stage2_round2",
        label="AMHT-V4-Stage2-R2",
        config="train/config_amht_v4_stage2_round2.yaml",
        color="#9a3412",
        marker="H",
    ),
    "amht_v4_stage2_round1": ModelSpec(
        key="amht_v4_stage2_round1",
        label="AMHT-V4-Stage2-R1",
        config="train/config_amht_v4_stage2_round1.yaml",
        color="#7c2d12",
        marker="h",
    ),
    "amht_v4_stage1_round4_long": ModelSpec(
        key="amht_v4_stage1_round4_long",
        label="AMHT-V4-Stage1-R4-Long",
        config="train/config_amht_v4_stage1_round4.yaml",
        color="#14532d",
        marker="*",
    ),
    "amht_v4_stage1_round4": ModelSpec(
        key="amht_v4_stage1_round4",
        label="AMHT-V4-Stage1-R4",
        config="train/config_amht_v4_stage1_round4.yaml",
        color="#0b6e4f",
        marker="X",
    ),
    "amht_v4_stage1_round3": ModelSpec(
        key="amht_v4_stage1_round3",
        label="AMHT-V4-Stage1-R3",
        config="train/config_amht_v4_stage1_round3.yaml",
        color="#1d3557",
        marker="P",
    ),
    "amht_v4_stage1_tuned": ModelSpec(
        key="amht_v4_stage1_tuned",
        label="AMHT-V4-Stage1",
        config="train/config_amht_v4_stage1_tuned.yaml",
        color="#264653",
        marker="o",
    ),
    "amht_v4_fast": ModelSpec(
        key="amht_v4_fast",
        label="AMHT-V4-Fast",
        config="train/config_amht_v4_8k_fast.yaml",
        color="#355c7d",
        marker="o",
    ),
    "amht_v4_accurate": ModelSpec(
        key="amht_v4_accurate",
        label="AMHT-V4-Accurate",
        config="train/config_amht_v4_8k_accurate.yaml",
        color="#6c5b7b",
        marker="^",
    ),
    "transformer_v4_baseline": ModelSpec(
        key="transformer_v4_baseline",
        label="Transformer",
        config="train/config_transformer_v4_baseline.yaml",
        color="#f67280",
        marker="s",
    ),
    "transformer_v4_stage2_baseline": ModelSpec(
        key="transformer_v4_stage2_baseline",
        label="Transformer",
        config="train/config_transformer_v4_stage2_baseline.yaml",
        color="#f67280",
        marker="s",
    ),
    "transformer_v4_stage2_round4_baseline": ModelSpec(
        key="transformer_v4_stage2_round4_baseline",
        label="Transformer",
        config="train/config_transformer_v4_stage2_round4_baseline.yaml",
        color="#f67280",
        marker="s",
    ),
    "transformer_v4_stage2_round7_baseline": ModelSpec(
        key="transformer_v4_stage2_round7_baseline",
        label="Transformer",
        config="train/config_transformer_v4_stage2_round7_baseline.yaml",
        color="#f67280",
        marker="s",
    ),
    "mamba3_hybrid_baseline": ModelSpec(
        key="mamba3_hybrid_baseline",
        label="Mamba-3-Inspired Hybrid",
        config="train/config_mamba3_hybrid_v4_baseline.yaml",
        color="#2a9d8f",
        marker="D",
    ),
    "mamba3_hybrid_v4_stage2_baseline": ModelSpec(
        key="mamba3_hybrid_v4_stage2_baseline",
        label="Mamba-3-Inspired Hybrid",
        config="train/config_mamba3_hybrid_v4_stage2_baseline.yaml",
        color="#2a9d8f",
        marker="D",
    ),
    "mamba3_hybrid_v4_stage2_round4_baseline": ModelSpec(
        key="mamba3_hybrid_v4_stage2_round4_baseline",
        label="Mamba-3-Inspired Hybrid",
        config="train/config_mamba3_hybrid_v4_stage2_round4_baseline.yaml",
        color="#2a9d8f",
        marker="D",
    ),
    "mamba3_hybrid_v4_stage2_round7_baseline": ModelSpec(
        key="mamba3_hybrid_v4_stage2_round7_baseline",
        label="Mamba-3-Inspired Hybrid",
        config="train/config_mamba3_hybrid_v4_stage2_round7_baseline.yaml",
        color="#2a9d8f",
        marker="D",
    ),
}


PRESETS = {
    "stage2_round7_validate": {
        "models": [
            "amht_v4_stage2_round7",
            "transformer_v4_stage2_round7_baseline",
            "mamba3_hybrid_v4_stage2_round7_baseline",
        ],
        "seeds": [42, 43, 44],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round7": {
        "models": [
            "amht_v4_stage2_round7",
            "transformer_v4_stage2_round7_baseline",
            "mamba3_hybrid_v4_stage2_round7_baseline",
        ],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round6": {
        "models": [
            "amht_v4_stage2_round6",
            "transformer_v4_stage2_round4_baseline",
            "mamba3_hybrid_v4_stage2_round4_baseline",
        ],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round4_validate": {
        "models": [
            "amht_v4_stage2_round4",
            "transformer_v4_stage2_round4_baseline",
            "mamba3_hybrid_v4_stage2_round4_baseline",
        ],
        "seeds": [42, 43, 44],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round5": {
        "models": [
            "amht_v4_stage2_round5",
            "transformer_v4_stage2_round4_baseline",
            "mamba3_hybrid_v4_stage2_round4_baseline",
        ],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round4": {
        "models": [
            "amht_v4_stage2_round4",
            "transformer_v4_stage2_round4_baseline",
            "mamba3_hybrid_v4_stage2_round4_baseline",
        ],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round3": {
        "models": ["amht_v4_stage2_round3", "transformer_v4_stage2_baseline", "mamba3_hybrid_v4_stage2_baseline"],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round2": {
        "models": ["amht_v4_stage2_round2", "transformer_v4_stage2_baseline", "mamba3_hybrid_v4_stage2_baseline"],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage2_round1": {
        "models": ["amht_v4_stage2_round1", "transformer_v4_stage2_baseline", "mamba3_hybrid_v4_stage2_baseline"],
        "seeds": [42],
        "seq_len": 16384,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage1_round4_validate": {
        "models": ["amht_v4_stage1_round4_long", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42, 43, 44],
        "seq_len": 8192,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
        "niah_seq_len": 16384,
    },
    "stage1_round4_long": {
        "models": ["amht_v4_stage1_round4_long", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42],
        "seq_len": 8192,
        "steps_scale": 2.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
    },
    "stage1_round4": {
        "models": ["amht_v4_stage1_round4", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42],
        "seq_len": 8192,
        "steps_scale": 1.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
    },
    "stage1_round3": {
        "models": ["amht_v4_stage1_round3", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42],
        "seq_len": 8192,
        "steps_scale": 1.0,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
    },
    "stage1_tuning": {
        "models": ["amht_v4_stage1_tuned", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42],
        "seq_len": 8192,
        "steps_scale": 0.5,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "all",
    },
    "colab_quick": {
        "models": ["amht_v4_fast", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42],
        "seq_len": 8192,
        "steps_scale": 0.1,
        "warmup_steps": 1,
        "benchmark_steps": 2,
        "eval_task": "throughput",
    },
    "paper_v4": {
        "models": ["amht_v4_fast", "amht_v4_accurate", "transformer_v4_baseline", "mamba3_hybrid_baseline"],
        "seeds": [42, 43, 44],
        "seq_len": 8192,
        "steps_scale": 1.0,
        "warmup_steps": 1,
        "benchmark_steps": 4,
        "eval_task": "all",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Colab-friendly AMHT experiment pipeline: train, benchmark, aggregate, and optionally generate figures/tables."
    )
    parser.add_argument("--preset", default="stage1_round4_long", choices=sorted(PRESETS))
    parser.add_argument("--models", default=None, help="Comma-separated model keys to override the preset model list")
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds to override the preset seed list")
    parser.add_argument("--seq-len", type=int, default=None, help="Optional override for training and primary evaluation sequence length")
    parser.add_argument("--niah-seq-len", type=int, default=None, help="Optional override for NIAH evaluation sequence length")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--outdir", default=None, help="Output directory for checkpoints, raw runs, reports, and optional paper artifacts")
    parser.add_argument("--steps-scale", type=float, default=None, help="Scale configured train steps by this factor")
    parser.add_argument(
        "--steps-override",
        action="append",
        default=[],
        help="Per-model train step override in the form model_key=steps",
    )
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional override for evaluation warmup steps")
    parser.add_argument("--benchmark-steps", type=int, default=None, help="Optional override for evaluation benchmark steps")
    parser.add_argument(
        "--eval-task",
        default=None,
        choices=["throughput", "niah", "state_tracking", "scaling", "all"],
        help="Benchmark task override",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation and produce tables/summary only")
    parser.add_argument("--force", action="store_true", help="Re-run even when outputs already exist")
    parser.add_argument(
        "--paper-assets-dir",
        default=None,
        help="Optional directory to copy generated figures and tables into for paper builds",
    )
    return parser.parse_args()


def parse_int_list(raw: str | None, fallback: list[int]) -> list[int]:
    if not raw:
        return fallback
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_model_list(raw: str | None, fallback: list[str]) -> list[str]:
    if not raw:
        return fallback
    models = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in models if item not in MODEL_SPECS]
    if unknown:
        raise SystemExit(f"Unknown model keys: {', '.join(unknown)}")
    return models


def parse_steps_override(items: list[str]) -> dict[str, int]:
    overrides: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --steps-override: {item}. Expected model_key=steps")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if key not in MODEL_SPECS:
            raise SystemExit(f"Unknown model key in --steps-override: {key}")
        overrides[key] = max(1, int(raw_value))
    return overrides


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_steps(model_key: str, config_path: Path, scale: float, overrides: dict[str, int]) -> int:
    if model_key in overrides:
        return overrides[model_key]
    cfg = load_yaml(config_path)
    base_steps = int(cfg["training"]["steps"])
    return max(1, int(math.ceil(base_steps * scale)))


def run_command(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def warn(message: str) -> None:
    print(f"warning={message}", flush=True)


def load_eval(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = [payload]
    return {item["task"]: item for item in payload}


def load_train_log(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def fmt(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def fmt_plain(mean: float | None, std: float | None, digits: int = 4) -> str:
    if mean is None or std is None:
        return "-"
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def collect_eval_metric(model_runs: list[dict], task: str, field: str) -> list[float]:
    values: list[float] = []
    for run in model_runs:
        item = run.get("eval", {}).get(task, {})
        value = item.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def collect_scaling_metric(model_runs: list[dict], seq_len: int, field: str) -> list[float]:
    values: list[float] = []
    for run in model_runs:
        for item in run.get("eval", {}).get("scaling", {}).get("results", []):
            if int(item.get("seq_len", -1)) == seq_len and isinstance(item.get(field), (int, float)):
                values.append(float(item[field]))
    return values


def collect_niah_depth(model_runs: list[dict], depth_index: int) -> list[float]:
    values: list[float] = []
    for run in model_runs:
        by_depth = run.get("eval", {}).get("niah", {}).get("accuracy_by_depth", [])
        if depth_index < len(by_depth):
            values.append(float(by_depth[depth_index]))
    return values


def collect_state_tracking_seq_len(model_runs: list[dict], seq_len: int) -> list[float]:
    values: list[float] = []
    for run in model_runs:
        for item in run.get("eval", {}).get("state_tracking", {}).get("results", []):
            if int(item.get("seq_len", -1)) == seq_len and isinstance(item.get("accuracy"), (int, float)):
                values.append(float(item["accuracy"]))
    return values


def niah_run_counts(run: dict) -> tuple[int | None, int | None]:
    item = run.get("eval", {}).get("niah", {})
    batch_size = item.get("batch_size")
    repeats = item.get("repeats")
    accuracy_by_depth = item.get("accuracy_by_depth", [])
    if not isinstance(batch_size, int) or not isinstance(repeats, int) or not accuracy_by_depth:
        return None, None
    cases_per_depth = batch_size * repeats
    total_hits = 0
    for score in accuracy_by_depth:
        if not isinstance(score, (int, float)):
            return None, None
        total_hits += int(round(float(score) * cases_per_depth))
    total_cases = cases_per_depth * len(accuracy_by_depth)
    return total_hits, total_cases


def collect_train_metric(model_runs: list[dict], field: str) -> list[float]:
    values: list[float] = []
    for run in model_runs:
        final_train = run.get("train_final", {})
        value = final_train.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def latex_table(headers: list[str], rows: list[list[str]]) -> str:
    spec = "l" + "c" * (len(headers) - 1)
    lines = [
        "\\begin{tabular}{" + spec + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def build_summary(
    runs_by_model: dict[str, list[dict]],
    model_keys: list[str],
    *,
    seed_count: int,
    eval_task: str,
    warmup_steps: int,
    benchmark_steps: int,
) -> dict:
    summary: dict[str, dict] = {}
    for key in model_keys:
        model_runs = runs_by_model[key]
        niah_hits_by_seed: list[float] = []
        niah_cases_by_seed: list[int] = []
        state_tracking_seq_lens = sorted(
            {
                int(item["seq_len"])
                for run in model_runs
                for item in run.get("eval", {}).get("state_tracking", {}).get("results", [])
                if "seq_len" in item
            }
        )
        for run in model_runs:
            hits, cases = niah_run_counts(run)
            if hits is not None and cases is not None:
                niah_hits_by_seed.append(float(hits))
                niah_cases_by_seed.append(cases)
        niah_hits_mean, niah_hits_std = mean_std(niah_hits_by_seed)
        cases_per_seed = niah_cases_by_seed[0] if niah_cases_by_seed and all(cases == niah_cases_by_seed[0] for cases in niah_cases_by_seed) else None
        summary[key] = {
            "label": MODEL_SPECS[key].label,
            "train": {
                "final_total_loss": {
                    "mean": mean_std(collect_train_metric(model_runs, "total_loss"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "total_loss"))[1],
                },
                "final_main_loss": {
                    "mean": mean_std(collect_train_metric(model_runs, "main_loss"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "main_loss"))[1],
                },
                "final_router_loss": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_loss"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_loss"))[1],
                },
                "final_router_mean": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_mean"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_mean"))[1],
                },
                "final_router_selected_ratio": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_selected_ratio"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_selected_ratio"))[1],
                },
                "final_router_selected_score_mean": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_selected_score_mean"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_selected_score_mean"))[1],
                },
                "final_router_unselected_score_mean": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_unselected_score_mean"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_unselected_score_mean"))[1],
                },
                "final_router_score_gap": {
                    "mean": mean_std(collect_train_metric(model_runs, "router_score_gap"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "router_score_gap"))[1],
                },
                "tokens_per_second": {
                    "mean": mean_std(collect_train_metric(model_runs, "tokens_per_second"))[0],
                    "std": mean_std(collect_train_metric(model_runs, "tokens_per_second"))[1],
                },
            },
            "throughput": {
                "tokens_per_second": {
                    "mean": mean_std(collect_eval_metric(model_runs, "throughput", "tokens_per_second"))[0],
                    "std": mean_std(collect_eval_metric(model_runs, "throughput", "tokens_per_second"))[1],
                },
                "milliseconds_per_step": {
                    "mean": mean_std(collect_eval_metric(model_runs, "throughput", "milliseconds_per_step"))[0],
                    "std": mean_std(collect_eval_metric(model_runs, "throughput", "milliseconds_per_step"))[1],
                },
            },
            "niah": {
                "mean_accuracy": {
                    "mean": mean_std(collect_eval_metric(model_runs, "niah", "mean_accuracy"))[0],
                    "std": mean_std(collect_eval_metric(model_runs, "niah", "mean_accuracy"))[1],
                },
                "hits_per_seed": {
                    "mean": niah_hits_mean,
                    "std": niah_hits_std,
                },
                "cases_per_seed": cases_per_seed,
                "aggregate_hits": int(sum(int(hits) for hits in niah_hits_by_seed)),
                "aggregate_cases": int(sum(niah_cases_by_seed)),
            },
            "state_tracking": {
                "mean_accuracy": {
                    "mean": mean_std(collect_eval_metric(model_runs, "state_tracking", "mean_accuracy"))[0],
                    "std": mean_std(collect_eval_metric(model_runs, "state_tracking", "mean_accuracy"))[1],
                },
                "accuracy_by_seq_len": {
                    str(seq_len): {
                        "mean": mean_std(collect_state_tracking_seq_len(model_runs, seq_len))[0],
                        "std": mean_std(collect_state_tracking_seq_len(model_runs, seq_len))[1],
                    }
                    for seq_len in state_tracking_seq_lens
                },
            },
        }

    depths = []
    niah_seq_len = None
    primary_seq_len = None
    for key in model_keys:
        model_runs = runs_by_model[key]
        if model_runs:
            depths = list(model_runs[0].get("eval", {}).get("niah", {}).get("needle_depths", []))
            candidate_niah_seq_len = model_runs[0].get("eval", {}).get("niah", {}).get("seq_len")
            if isinstance(candidate_niah_seq_len, int):
                niah_seq_len = candidate_niah_seq_len
            candidate_primary_seq_len = model_runs[0].get("eval", {}).get("throughput", {}).get("seq_len")
            if isinstance(candidate_primary_seq_len, int):
                primary_seq_len = candidate_primary_seq_len
            if depths:
                break

    state_tracking_seq_lens = sorted(
        {
            int(seq_len)
            for key in model_keys
            for run in runs_by_model[key]
            for seq_len in run.get("eval", {}).get("state_tracking", {}).get("seq_lens", [])
        }
    )

    scaling_lengths = sorted(
        {
            int(item["seq_len"])
            for key in model_keys
            for run in runs_by_model[key]
            for item in run.get("eval", {}).get("scaling", {}).get("results", [])
            if "seq_len" in item
        }
    )

    return {
        "models": summary,
        "depths": depths,
        "seed_count": seed_count,
        "eval_task": eval_task,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "primary_seq_len": primary_seq_len,
        "niah_seq_len": niah_seq_len,
        "state_tracking_seq_lens": state_tracking_seq_lens,
        "scaling_lengths": scaling_lengths,
    }


def write_summary_markdown(
    out_path: Path,
    summary: dict,
    model_keys: list[str],
    run_dir: Path,
) -> None:
    primary_seq_len = summary.get("primary_seq_len")
    state_tracking_seq_lens = [int(seq_len) for seq_len in summary.get("state_tracking_seq_lens", [])]
    primary_throughput_label = "Primary throughput benchmark (tok/s)"
    primary_latency_label = "Primary throughput latency (ms/step)"
    if primary_seq_len is not None:
        primary_throughput_label = f"Primary throughput @ {primary_seq_len} (tok/s)"
        primary_latency_label = f"Primary latency @ {primary_seq_len} (ms/step)"
    lines = [
        "# AMHT Training Summary",
        "",
        f"- Generated: {datetime.now(timezone.utc).isoformat()}",
        f"- Run directory: `{run_dir}`",
        "",
        "## Main Comparison",
        "",
        "| Model | Final train loss | Router mean | Selected ratio | Score gap | "
        + primary_throughput_label
        + " | "
        + primary_latency_label
        + " | Mean NIAH accuracy | Mean state accuracy | NIAH hits |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    metadata_lines: list[str] = []
    metadata_lines.append(f"- Seeds: `{summary.get('seed_count', 0)}`")
    metadata_lines.append("- Tabulated metrics use `mean +- std` across seeds.")
    metadata_lines.append(
        f"- Primary throughput metric: the eval bundle's `throughput` section, using `{summary.get('warmup_steps', '-')}` warmup step(s) and `{summary.get('benchmark_steps', '-')}` timed benchmark step(s)."
    )
    if summary.get("primary_seq_len") is not None:
        metadata_lines.append(f"- Primary throughput seq_len: `{summary['primary_seq_len']}`")
    if summary.get("niah_seq_len") is not None:
        metadata_lines.append(f"- NIAH seq_len: `{summary['niah_seq_len']}`")
    metadata_lines.append("- NIAH hits aggregate exact correct retrievals across all depths and all seeds.")
    if state_tracking_seq_lens:
        metadata_lines.append(
            "- State-tracking metric: the eval bundle's `state_tracking` section, using modulo-sum final-token accuracy."
        )
        metadata_lines.append(f"- State-tracking seq_lens: `{state_tracking_seq_lens}`")
    if metadata_lines:
        lines[4:4] = metadata_lines + [""]
    for key in model_keys:
        model = summary["models"][key]
        lines.append(
            "| "
            + " | ".join(
                [
                    model["label"],
                    fmt_plain(model["train"]["final_total_loss"]["mean"], model["train"]["final_total_loss"]["std"]),
                    fmt_plain(model["train"]["final_router_mean"]["mean"], model["train"]["final_router_mean"]["std"]),
                    fmt_plain(model["train"]["final_router_selected_ratio"]["mean"], model["train"]["final_router_selected_ratio"]["std"]),
                    fmt_plain(model["train"]["final_router_score_gap"]["mean"], model["train"]["final_router_score_gap"]["std"]),
                    fmt_plain(model["throughput"]["tokens_per_second"]["mean"], model["throughput"]["tokens_per_second"]["std"]),
                    fmt_plain(model["throughput"]["milliseconds_per_step"]["mean"], model["throughput"]["milliseconds_per_step"]["std"]),
                    fmt_plain(model["niah"]["mean_accuracy"]["mean"], model["niah"]["mean_accuracy"]["std"]),
                    fmt_plain(model["state_tracking"]["mean_accuracy"]["mean"], model["state_tracking"]["mean_accuracy"]["std"]),
                    f"{model['niah']['aggregate_hits']} / {model['niah']['aggregate_cases']}",
                ]
            )
            + " |"
        )

    if state_tracking_seq_lens:
        lines.extend(
            [
                "",
                "## State Tracking By Sequence Length",
                "",
                "| Seq Len | " + " | ".join(summary["models"][key]["label"] for key in model_keys) + " |",
                "| --- | " + " | ".join("---" for _ in model_keys) + " |",
            ]
        )
        for seq_len in state_tracking_seq_lens:
            row = [str(seq_len)]
            for key in model_keys:
                pair = summary["models"][key]["state_tracking"]["accuracy_by_seq_len"].get(str(seq_len), {})
                row.append(fmt_plain(pair.get("mean"), pair.get("std")))
            lines.append("| " + " | ".join(row) + " |")

    lines.extend(
        [
            "",
            "## Run Assets",
            "",
            f"- Figures: `{run_dir / 'figures'}`",
            f"- LaTeX tables: `{run_dir / 'report' / 'paper_tables.tex'}`",
            f"- Raw per-seed runs: `{run_dir / 'runs'}`",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_tables(out_path: Path, summary: dict, runs_by_model: dict[str, list[dict]], model_keys: list[str]) -> None:
    main_rows = []
    primary_seq_len = summary.get("primary_seq_len")
    state_tracking_seq_lens = [int(seq_len) for seq_len in summary.get("state_tracking_seq_lens", [])]
    throughput_header = "Primary Throughput"
    latency_header = "Primary Latency"
    if primary_seq_len is not None:
        throughput_header = f"Primary Throughput @ {primary_seq_len}"
        latency_header = f"Primary Latency @ {primary_seq_len}"
    for key in model_keys:
        model = summary["models"][key]
        main_rows.append(
            [
                model["label"],
                fmt(model["train"]["final_total_loss"]["mean"], model["train"]["final_total_loss"]["std"]),
                fmt(model["throughput"]["tokens_per_second"]["mean"], model["throughput"]["tokens_per_second"]["std"]),
                fmt(model["throughput"]["milliseconds_per_step"]["mean"], model["throughput"]["milliseconds_per_step"]["std"]),
                fmt(model["niah"]["mean_accuracy"]["mean"], model["niah"]["mean_accuracy"]["std"]),
                fmt(model["state_tracking"]["mean_accuracy"]["mean"], model["state_tracking"]["mean_accuracy"]["std"]),
                f"{model['niah']['aggregate_hits']} / {model['niah']['aggregate_cases']}",
            ]
        )

    sections = [
        "% Auto-generated by scripts/run_colab_paper.py",
        "",
        "% Main comparison",
        latex_table(
            ["Model", "Train Loss", throughput_header, latency_header, "Mean NIAH", "Mean State", "NIAH Hits"],
            main_rows,
        ),
    ]

    depths = summary["depths"]
    if depths:
        depth_rows = []
        for depth_index, depth in enumerate(depths):
            row = [str(depth)]
            for key in model_keys:
                pair = mean_std(collect_niah_depth(runs_by_model[key], depth_index))
                row.append(fmt(*pair))
            depth_rows.append(row)
        sections.extend(
            [
                "",
                "% NIAH by depth",
                latex_table(["Depth"] + [MODEL_SPECS[key].label for key in model_keys], depth_rows),
            ]
        )

    scaling_lengths = summary["scaling_lengths"]
    if scaling_lengths:
        scaling_rows = []
        for seq_len in scaling_lengths:
            row = [str(seq_len)]
            for key in model_keys:
                pair = mean_std(collect_scaling_metric(runs_by_model[key], seq_len, "tokens_per_second"))
                row.append(fmt(*pair))
            scaling_rows.append(row)
        sections.extend(
            [
                "",
                "% Throughput scaling",
                latex_table(["Seq Len"] + [MODEL_SPECS[key].label for key in model_keys], scaling_rows),
            ]
        )

    if state_tracking_seq_lens:
        state_rows = []
        for seq_len in state_tracking_seq_lens:
            row = [str(seq_len)]
            for key in model_keys:
                pair = mean_std(collect_state_tracking_seq_len(runs_by_model[key], seq_len))
                row.append(fmt(*pair))
            state_rows.append(row)
        sections.extend(
            [
                "",
                "% State tracking by sequence length",
                latex_table(["Seq Len"] + [MODEL_SPECS[key].label for key in model_keys], state_rows),
            ]
        )

    out_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")


def aggregate_curve(model_runs: list[dict], field: str) -> tuple[list[int], list[float], list[float]]:
    by_step: dict[int, list[float]] = {}
    for run in model_runs:
        for record in run.get("train_log", []):
            if isinstance(record.get("step"), int) and isinstance(record.get(field), (int, float)):
                by_step.setdefault(int(record["step"]), []).append(float(record[field]))
    steps = sorted(by_step)
    means, stds = [], []
    for step in steps:
        mean, std = mean_std(by_step[step])
        means.append(0.0 if mean is None else mean)
        stds.append(0.0 if std is None else std)
    return steps, means, stds


def generate_figures(figures_dir: Path, runs_by_model: dict[str, list[dict]], model_keys: list[str]) -> None:
    if plt is None or np is None:
        raise SystemExit("matplotlib and numpy are required to generate paper figures")

    figures_dir.mkdir(parents=True, exist_ok=True)

    first_key = next((key for key in model_keys if runs_by_model[key]), None)
    if first_key is None:
        return
    depths = list(runs_by_model[first_key][0].get("eval", {}).get("niah", {}).get("needle_depths", []))
    if depths:
        x = np.arange(len(depths))
        width = 0.8 / max(len(model_keys), 1)
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        offsets = np.linspace(-(len(model_keys) - 1) / 2, (len(model_keys) - 1) / 2, len(model_keys))
        for offset, key in zip(offsets, model_keys):
            means, stds = [], []
            for idx, _ in enumerate(depths):
                mean, std = mean_std(collect_niah_depth(runs_by_model[key], idx))
                means.append(0.0 if mean is None else mean)
                stds.append(0.0 if std is None else std)
            ax.bar(x + offset * width, means, width=width, yerr=stds, capsize=4, color=MODEL_SPECS[key].color, label=MODEL_SPECS[key].label)
        ax.set_xticks(x)
        ax.set_xticklabels([str(depth) for depth in depths])
        ax.set_ylim(0.0, 1.08)
        ax.set_xlabel("Needle depth")
        ax.set_ylabel("Accuracy")
        ax.set_title("Needle-in-a-Haystack Accuracy by Depth")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "niah_by_depth.pdf", bbox_inches="tight")
        plt.close(fig)

    scaling_lengths = sorted(
        {
            int(item["seq_len"])
            for key in model_keys
            for run in runs_by_model[key]
            for item in run.get("eval", {}).get("scaling", {}).get("results", [])
            if "seq_len" in item
        }
    )
    if scaling_lengths:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for key in model_keys:
            means, stds = [], []
            for seq_len in scaling_lengths:
                mean, std = mean_std(collect_scaling_metric(runs_by_model[key], seq_len, "tokens_per_second"))
                means.append(np.nan if mean is None else mean)
                stds.append(0.0 if std is None else std)
            ax.errorbar(
                scaling_lengths,
                means,
                yerr=stds,
                marker=MODEL_SPECS[key].marker,
                linewidth=2,
                color=MODEL_SPECS[key].color,
                label=MODEL_SPECS[key].label,
            )
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Tokens / second")
        ax.set_title("Throughput Scaling")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "throughput_scaling.pdf", bbox_inches="tight")
        plt.close(fig)

    state_tracking_seq_lens = sorted(
        {
            int(item["seq_len"])
            for key in model_keys
            for run in runs_by_model[key]
            for item in run.get("eval", {}).get("state_tracking", {}).get("results", [])
            if "seq_len" in item
        }
    )
    if state_tracking_seq_lens:
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        for key in model_keys:
            means, stds = [], []
            for seq_len in state_tracking_seq_lens:
                mean, std = mean_std(collect_state_tracking_seq_len(runs_by_model[key], seq_len))
                means.append(np.nan if mean is None else mean)
                stds.append(0.0 if std is None else std)
            ax.errorbar(
                state_tracking_seq_lens,
                means,
                yerr=stds,
                marker=MODEL_SPECS[key].marker,
                linewidth=2,
                color=MODEL_SPECS[key].color,
                label=MODEL_SPECS[key].label,
            )
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Accuracy")
        ax.set_title("State Tracking Accuracy by Sequence Length")
        ax.set_ylim(0.0, 1.05)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "state_tracking_by_seq_len.pdf", bbox_inches="tight")
        plt.close(fig)

    if all(collect_eval_metric(runs_by_model[key], "throughput", "tokens_per_second") for key in model_keys):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for key in model_keys:
            tps_mean, _ = mean_std(collect_eval_metric(runs_by_model[key], "throughput", "tokens_per_second"))
            acc_mean, _ = mean_std(collect_eval_metric(runs_by_model[key], "niah", "mean_accuracy"))
            if tps_mean is None or acc_mean is None:
                continue
            ax.scatter(tps_mean, acc_mean, s=120, color=MODEL_SPECS[key].color, label=MODEL_SPECS[key].label)
            ax.annotate(MODEL_SPECS[key].label, (tps_mean, acc_mean), xytext=(8, 6), textcoords="offset points", fontsize=10)
        ax.set_xlabel("Tokens / second")
        ax.set_ylabel("Mean NIAH accuracy")
        ax.set_title("Efficiency-Quality Frontier")
        fig.tight_layout()
        fig.savefig(figures_dir / "efficiency_quality_frontier.pdf", bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    plotted = False
    for key in model_keys:
        steps, means, stds = aggregate_curve(runs_by_model[key], "main_loss")
        if not steps:
            continue
        plotted = True
        steps_arr = np.asarray(steps, dtype=float)
        means_arr = np.asarray(means, dtype=float)
        stds_arr = np.asarray(stds, dtype=float)
        ax.plot(steps_arr, means_arr, color=MODEL_SPECS[key].color, linewidth=2, label=MODEL_SPECS[key].label)
        ax.fill_between(steps_arr, means_arr - stds_arr, means_arr + stds_arr, color=MODEL_SPECS[key].color, alpha=0.18)
    if plotted:
        ax.set_xlabel("Training step")
        ax.set_ylabel("Main loss")
        ax.set_title("Training Curves")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def sync_paper_assets(run_dir: Path, paper_assets_dir: Path) -> None:
    paper_assets_dir.mkdir(parents=True, exist_ok=True)
    figures_out = paper_assets_dir / "figures"
    figures_out.mkdir(parents=True, exist_ok=True)
    for figure in (run_dir / "figures").glob("*.pdf"):
        shutil.copy2(figure, figures_out / figure.name)
    shutil.copy2(run_dir / "report" / "paper_tables.tex", paper_assets_dir / "paper_tables.tex")
    shutil.copy2(run_dir / "report" / "summary.md", paper_assets_dir / "summary.md")


def main() -> None:
    args = parse_args()
    preset = PRESETS[args.preset]
    model_keys = parse_model_list(args.models, list(preset["models"]))
    seeds = parse_int_list(args.seeds, list(preset["seeds"]))
    seq_len = int(args.seq_len if args.seq_len is not None else preset.get("seq_len", 8192))
    steps_scale = float(args.steps_scale if args.steps_scale is not None else preset["steps_scale"])
    step_overrides = parse_steps_override(args.steps_override)
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(preset["warmup_steps"])
    benchmark_steps = args.benchmark_steps if args.benchmark_steps is not None else int(preset["benchmark_steps"])
    eval_task = args.eval_task if args.eval_task is not None else str(preset["eval_task"])
    niah_seq_len = args.niah_seq_len if args.niah_seq_len is not None else preset.get("niah_seq_len")

    outdir = Path(args.outdir) if args.outdir else ROOT / "paper_runs" / args.preset
    runs_dir = outdir / "runs"
    figures_dir = outdir / "figures"
    report_dir = outdir / "report"
    runs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "preset": args.preset,
        "models": model_keys,
        "seeds": seeds,
        "seq_len": seq_len,
        "device": args.device,
        "steps_scale": steps_scale,
        "warmup_steps": warmup_steps,
        "benchmark_steps": benchmark_steps,
        "eval_task": eval_task,
        "niah_seq_len": niah_seq_len,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.executable,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    if not args.skip_train or not args.skip_eval:
        for model_key in model_keys:
            spec = MODEL_SPECS[model_key]
            config_path = ROOT / spec.config
            steps = resolve_steps(model_key, config_path, steps_scale, step_overrides)
            for seed in seeds:
                run_dir = runs_dir / model_key / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = run_dir / f"{model_key}_seed{seed}_seq{seq_len}.pt"
                train_log_path = run_dir / "train.jsonl"
                eval_json_path = run_dir / "eval.json"

                if not args.skip_train and (args.force or not (checkpoint_path.exists() and train_log_path.exists())):
                    train_cmd = [
                        sys.executable,
                        "train/train.py",
                        "--config",
                        spec.config,
                        "--seq-len",
                        str(seq_len),
                        "--steps",
                        str(steps),
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--log-jsonl",
                        str(train_log_path),
                        "--checkpoint-out",
                        str(checkpoint_path),
                    ]
                    run_command(train_cmd)

                if not args.skip_eval and (args.force or not eval_json_path.exists()):
                    eval_cmd = [
                        sys.executable,
                        "eval/benchmark.py",
                        "--config",
                        spec.config,
                        "--checkpoint",
                        str(checkpoint_path),
                        "--task",
                        eval_task,
                        "--seq-len",
                        str(seq_len),
                        "--device",
                        args.device,
                        "--seed",
                        str(seed),
                        "--save-json",
                        str(eval_json_path),
                        "--warmup-steps",
                        str(warmup_steps),
                        "--benchmark-steps",
                        str(benchmark_steps),
                    ]
                    if niah_seq_len is not None:
                        eval_cmd.extend(["--niah-seq-len", str(niah_seq_len)])
                    run_command(eval_cmd)

    if args.skip_report:
        return

    runs_by_model: dict[str, list[dict]] = {key: [] for key in model_keys}
    for model_key in model_keys:
        for seed in seeds:
            run_dir = runs_dir / model_key / f"seed{seed}"
            train_log_path = run_dir / "train.jsonl"
            eval_json_path = run_dir / "eval.json"
            if not eval_json_path.exists():
                if args.skip_eval:
                    warn(f"missing_eval_json={eval_json_path}")
                    continue
                raise SystemExit(f"Missing eval JSON: {eval_json_path}")
            train_log: list[dict] = []
            train_final: dict = {}
            if train_log_path.exists():
                train_log = load_train_log(train_log_path)
                if train_log:
                    train_final = train_log[-1]
                elif not args.skip_train:
                    raise SystemExit(f"Empty train log: {train_log_path}")
                else:
                    warn(f"empty_train_log={train_log_path}")
            elif not args.skip_train:
                raise SystemExit(f"Missing train log: {train_log_path}")
            else:
                warn(f"missing_train_log={train_log_path}")
            runs_by_model[model_key].append(
                {
                    "seed": seed,
                    "train_log": train_log,
                    "train_final": train_final,
                    "eval": load_eval(eval_json_path),
                    "run_dir": str(run_dir),
                }
            )

    if not any(runs_by_model.values()):
        raise SystemExit("No completed runs found for report generation.")

    summary = build_summary(
        runs_by_model,
        model_keys,
        seed_count=len(seeds),
        eval_task=eval_task,
        warmup_steps=warmup_steps,
        benchmark_steps=benchmark_steps,
    )
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_summary_markdown(report_dir / "summary.md", summary, model_keys, outdir)
    write_latex_tables(report_dir / "paper_tables.tex", summary, runs_by_model, model_keys)
    figures_status = "generated"
    if args.skip_figures:
        figures_status = "skipped_by_flag"
    elif plt is None or np is None:
        figures_status = "skipped_missing_matplotlib_or_numpy"
    else:
        generate_figures(figures_dir, runs_by_model, model_keys)

    if args.paper_assets_dir:
        sync_paper_assets(outdir, Path(args.paper_assets_dir))

    print(f"run_bundle={outdir}")
    print(f"summary_md={report_dir / 'summary.md'}")
    print(f"tables_tex={report_dir / 'paper_tables.tex'}")
    print(f"figures_dir={figures_dir}")
    print(f"figures_status={figures_status}")


if __name__ == "__main__":
    main()
