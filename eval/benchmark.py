#!/usr/bin/env python3
"""Run throughput, NIAH, and scaling benchmarks for AMHT."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run benchmarking. Install dependencies from requirements.txt."
    ) from exc

import yaml

from eval.niah import benchmark_niah
from eval.scaling import benchmark_scaling
from model.amht import AMHTModel, synthetic_batch
from model.mamba3_hybrid import Mamba3HybridModel
from model.transformer import LocalTransformerModel


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str | None = None) -> torch.device:
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(cfg: dict) -> torch.nn.Module:
    architecture = str(cfg["model"].get("architecture", "amht")).lower()
    if architecture == "amht":
        return AMHTModel(cfg)
    if architecture == "mamba3_hybrid":
        return Mamba3HybridModel(cfg)
    if architecture == "transformer":
        return LocalTransformerModel(cfg)
    raise SystemExit(f"Unsupported model.architecture={architecture}")


def resolve_checkpoint_path(checkpoint_arg: str | None, cfg: dict, seq_len: int) -> Path | None:
    checkpoint_root = Path(os.environ.get("AMHT_CHECKPOINT_DIR", cfg["training"]["checkpoint_dir"]))
    architecture = str(cfg["model"].get("architecture", "amht")).lower()
    inferred = checkpoint_root / f"{architecture}_seq{seq_len}.pt"

    if checkpoint_arg is None:
        return inferred

    candidate = Path(checkpoint_arg)
    if candidate.is_absolute() or candidate.exists():
        return candidate

    # If the caller passed an old relative path like checkpoints/amht_seq8192.pt while
    # AMHT_CHECKPOINT_DIR points elsewhere, prefer the current checkpoint root with the same filename.
    alternate = checkpoint_root / candidate.name
    if alternate.exists():
        return alternate
    return candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark AMHT")
    parser.add_argument("--config", default="train/config.yaml", help="Path to YAML config")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path. If omitted, load from AMHT_CHECKPOINT_DIR or training.checkpoint_dir.",
    )
    parser.add_argument("--task", default="throughput", choices=["throughput", "niah", "scaling", "all"])
    parser.add_argument("--seq-len", type=int, default=8192, help="Primary context length")
    parser.add_argument("--niah-seq-len", type=int, default=None, help="Optional override for evaluation.niah.seq_len")
    parser.add_argument("--niah-batch-size", type=int, default=None, help="Optional override for evaluation.niah.batch_size")
    parser.add_argument("--niah-repeats", type=int, default=None, help="Optional override for evaluation.niah.repeats")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override")
    parser.add_argument("--save-json", default=None, help="Optional path to save benchmark JSON output")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Optional override for evaluation.warmup_steps")
    parser.add_argument("--benchmark-steps", type=int, default=None, help="Optional override for evaluation.benchmark_steps")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_throughput(model: torch.nn.Module, cfg: dict, seq_len: int, device: torch.device) -> dict:
    eval_cfg = cfg["evaluation"]
    batch = synthetic_batch(
        batch_size=int(eval_cfg["batch_size"]),
        seq_len=seq_len,
        vocab_size=int(cfg["model"]["vocab_size"]),
        device=device,
    )
    for _ in range(int(eval_cfg["warmup_steps"])):
        model(batch)
        synchronize(device)

    started = time.perf_counter()
    for _ in range(int(eval_cfg["benchmark_steps"])):
        model(batch)
    synchronize(device)
    elapsed = time.perf_counter() - started

    total_tokens = int(eval_cfg["batch_size"]) * seq_len * int(eval_cfg["benchmark_steps"])
    return {
        "task": "throughput",
        "device": str(device),
        "seq_len": seq_len,
        "batch_size": int(eval_cfg["batch_size"]),
        "steps": int(eval_cfg["benchmark_steps"]),
        "tokens_per_second": total_tokens / max(elapsed, 1e-6),
        "milliseconds_per_step": (elapsed / int(eval_cfg["benchmark_steps"])) * 1000.0,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.warmup_steps is not None:
        cfg.setdefault("evaluation", {})["warmup_steps"] = int(args.warmup_steps)
    if args.benchmark_steps is not None:
        cfg.setdefault("evaluation", {})["benchmark_steps"] = int(args.benchmark_steps)
    if args.niah_seq_len is not None:
        cfg.setdefault("evaluation", {}).setdefault("niah", {})["seq_len"] = int(args.niah_seq_len)
    if args.niah_batch_size is not None:
        cfg.setdefault("evaluation", {}).setdefault("niah", {})["batch_size"] = int(args.niah_batch_size)
    if args.niah_repeats is not None:
        cfg.setdefault("evaluation", {}).setdefault("niah", {})["repeats"] = int(args.niah_repeats)
    if args.seq_len > cfg["model"]["max_seq_len"]:
        raise SystemExit(
            f"--seq-len {args.seq_len} exceeds config model.max_seq_len={cfg['model']['max_seq_len']}"
        )
    niah_seq_len = int(cfg.get("evaluation", {}).get("niah", {}).get("seq_len", args.seq_len))
    if niah_seq_len > cfg["model"]["max_seq_len"]:
        raise SystemExit(
            f"NIAH seq_len {niah_seq_len} exceeds config model.max_seq_len={cfg['model']['max_seq_len']}"
        )

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)
    model = build_model(cfg).to(device)
    checkpoint_path = resolve_checkpoint_path(args.checkpoint, cfg, args.seq_len)
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise SystemExit(
                f"Checkpoint not found: {checkpoint_path}. "
                "Pass --checkpoint explicitly or set AMHT_CHECKPOINT_DIR to the directory used during training."
            )
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        print(json.dumps({"loaded_checkpoint": str(checkpoint_path)}))
    model.eval()

    outputs = []
    if args.task in {"throughput", "all"}:
        outputs.append(benchmark_throughput(model, cfg, args.seq_len, device))
    if args.task in {"niah", "all"}:
        outputs.append(benchmark_niah(model, cfg, device))
    if args.task in {"scaling", "all"}:
        outputs.append(benchmark_scaling(model, cfg, device, benchmark_throughput))

    if len(outputs) == 1:
        payload: dict | list = outputs[0]
        if isinstance(payload, dict):
            payload["seed"] = seed
        rendered = json.dumps(payload, indent=2)
        print(rendered)
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(rendered + "\n", encoding="utf-8")
    else:
        for item in outputs:
            item["seed"] = seed
        rendered = json.dumps(outputs, indent=2)
        print(rendered)
        if args.save_json:
            Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.save_json).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
