#!/usr/bin/env python3
"""Train AMHT with synthetic data for bring-up and benchmarking."""

from __future__ import annotations

import argparse
import importlib.util
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
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run training. Install dependencies from requirements.txt."
    ) from exc

import yaml

from data.dataset import MixedDataset, RetrievalDataset, StateTrackingDataset, SyntheticDataset
from model.amht import AMHTModel, LossBreakdown, compute_loss
from model.mamba3_hybrid import Mamba3HybridModel
from model.transformer import LocalTransformerModel

_DISTRIBUTED_PATH = ROOT / "train" / "distributed.py"
_DISTRIBUTED_SPEC = importlib.util.spec_from_file_location("amht_local_distributed", _DISTRIBUTED_PATH)
if _DISTRIBUTED_SPEC is None or _DISTRIBUTED_SPEC.loader is None:
    raise SystemExit(f"Unable to load local distributed helper from {_DISTRIBUTED_PATH}")
_DISTRIBUTED_MODULE = importlib.util.module_from_spec(_DISTRIBUTED_SPEC)
_DISTRIBUTED_SPEC.loader.exec_module(_DISTRIBUTED_MODULE)
maybe_init_distributed = _DISTRIBUTED_MODULE.maybe_init_distributed


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


def build_model(cfg: dict) -> nn.Module:
    architecture = str(cfg["model"].get("architecture", "amht")).lower()
    if architecture == "amht":
        return AMHTModel(cfg)
    if architecture == "mamba3_hybrid":
        return Mamba3HybridModel(cfg)
    if architecture == "transformer":
        return LocalTransformerModel(cfg)
    raise SystemExit(f"Unsupported model.architecture={architecture}")


def model_name(cfg: dict) -> str:
    return str(cfg["model"].get("architecture", "amht")).lower()


def build_retrieval_dataset(
    cfg: dict,
    seq_len: int,
    total_samples: int,
    *,
    seed: int | None = None,
) -> RetrievalDataset:
    data_cfg = cfg.get("data", {})
    niah_cfg = cfg["evaluation"]["niah"]
    retrieval_cfg = dict(niah_cfg)
    retrieval_cfg.update(data_cfg.get("retrieval", {}))
    return RetrievalDataset(
        vocab_size=int(cfg["model"]["vocab_size"]),
        seq_len=seq_len,
        total_samples=total_samples,
        pad_token=int(retrieval_cfg["pad_token"]),
        key_start=int(retrieval_cfg["key_start"]),
        value_start=int(retrieval_cfg["value_start"]),
        num_pairs=int(retrieval_cfg["num_pairs"]),
        num_keys=int(retrieval_cfg.get("num_keys", retrieval_cfg["num_pairs"])),
        depth_choices=[
            float(depth)
            for depth in retrieval_cfg.get(
                "depth_choices",
                retrieval_cfg.get("needle_depths", [0.1, 0.3, 0.5, 0.7, 0.9]),
            )
        ],
        seed=seed,
    )


def build_state_tracking_dataset(
    cfg: dict,
    seq_len: int,
    total_samples: int,
    *,
    seed: int | None = None,
) -> StateTrackingDataset:
    state_cfg = cfg.get("data", {}).get("state_tracking", {})
    return StateTrackingDataset(
        vocab_size=int(cfg["model"]["vocab_size"]),
        seq_len=seq_len,
        total_samples=total_samples,
        task=str(state_cfg.get("task", "modsum")),
        modulus=int(state_cfg.get("modulus", 16)),
        digit_start=int(state_cfg.get("digit_start", 0)),
        seed=seed,
    )


def build_dataset(
    cfg: dict,
    seq_len: int,
    total_samples: int,
    *,
    seed: int,
):
    data_cfg = cfg.get("data", {})
    dataset_type = str(data_cfg.get("dataset_type", "synthetic")).lower()
    if dataset_type == "retrieval":
        # Keep retrieval-only training aligned with the historical stage-two setup.
        return build_retrieval_dataset(cfg, seq_len, total_samples)
    if dataset_type == "state_tracking":
        return build_state_tracking_dataset(cfg, seq_len, total_samples, seed=seed)
    if dataset_type == "mixed":
        mixture_cfg = data_cfg.get("mixture", {})
        retrieval_weight = float(mixture_cfg.get("retrieval_weight", 0.0))
        state_tracking_weight = float(mixture_cfg.get("state_tracking_weight", 0.0))
        datasets = {}
        weights = {}
        if retrieval_weight > 0.0:
            datasets["retrieval"] = build_retrieval_dataset(cfg, seq_len, total_samples, seed=seed)
            weights["retrieval"] = retrieval_weight
        if state_tracking_weight > 0.0:
            datasets["state_tracking"] = build_state_tracking_dataset(cfg, seq_len, total_samples, seed=seed + 10_000_000)
            weights["state_tracking"] = state_tracking_weight
        return MixedDataset(datasets, weights, total_samples=total_samples, seed=seed)
    if dataset_type == "synthetic":
        return SyntheticDataset(
            vocab_size=int(cfg["model"]["vocab_size"]),
            seq_len=seq_len,
            total_samples=total_samples,
            seed=seed,
        )
    raise SystemExit(f"Unsupported data.dataset_type={dataset_type}")


def dataset_type_name(cfg: dict) -> str:
    return str(cfg.get("data", {}).get("dataset_type", "synthetic")).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AMHT model")
    parser.add_argument("--config", default="train/config.yaml", help="Path to YAML config")
    parser.add_argument("--seq-len", type=int, default=8192, help="Training context length")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for training steps")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed override")
    parser.add_argument("--resume", default=None, help="Optional checkpoint path to resume from")
    parser.add_argument("--log-jsonl", default=None, help="Optional JSONL path for training metrics")
    parser.add_argument("--checkpoint-out", default=None, help="Optional checkpoint output path")
    return parser.parse_args()


def sample_batch_sources(dataset, start: int, stop: int) -> dict[str, int]:
    if not hasattr(dataset, "sample_source"):
        return {}
    counts: dict[str, int] = {}
    for index in range(start, stop):
        source = str(dataset.sample_source(index))
        counts[source] = counts.get(source, 0) + 1
    return counts


def losses_are_finite(losses: LossBreakdown) -> bool:
    values = (
        losses.total,
        losses.main,
        losses.router,
        losses.router_mean,
        losses.router_selected_ratio,
        losses.router_selected_score_mean,
        losses.router_unselected_score_mean,
        losses.router_score_gap,
    )
    return all(bool(torch.isfinite(value).all().item()) for value in values)


def parameters_are_finite(model: nn.Module) -> bool:
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            return False
    return True


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seq_len > cfg["model"]["max_seq_len"]:
        raise SystemExit(
            f"--seq-len {args.seq_len} exceeds config model.max_seq_len={cfg['model']['max_seq_len']}"
        )

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    device = choose_device(args.device)
    maybe_init_distributed()
    model = build_model(cfg).to(device)
    architecture = model_name(cfg)

    train_cfg = cfg["training"]
    loss_cfg = cfg["loss"]
    loss_mode = str(loss_cfg.get("mode", "next_token"))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    checkpoint_root = os.environ.get("AMHT_CHECKPOINT_DIR", train_cfg["checkpoint_dir"])
    checkpoint_dir = Path(checkpoint_root)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_jsonl) if args.log_jsonl else checkpoint_dir / f"train_seq{args.seq_len}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    total_steps = args.steps if args.steps is not None else int(train_cfg["steps"])
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        state_dict = checkpoint.get("model", checkpoint)
        model.load_state_dict(state_dict)
        optimizer_state = checkpoint.get("optimizer")
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        start_step = int(checkpoint.get("step", 0))
        print(f"resumed_from={args.resume}")
        print(f"resume_step={start_step}")

    total_samples = (start_step + total_steps) * int(train_cfg["batch_size"])
    dataset = build_dataset(cfg, args.seq_len, total_samples, seed=seed)
    dataset_type = dataset_type_name(cfg)

    model.train()
    started = time.perf_counter()
    for step in range(start_step + 1, start_step + total_steps + 1):
        start = (step - 1) * int(train_cfg["batch_size"])
        stop = step * int(train_cfg["batch_size"])
        batch_sources = sample_batch_sources(dataset, start, stop)
        batch = [dataset[index] for index in range(start, stop)]
        tokens = torch.stack(batch, dim=0).to(device)
        losses = compute_loss(
            model=model,
            tokens=tokens,
            main_weight=float(loss_cfg["main_weight"]),
            router_weight=float(loss_cfg["router_weight"]),
            loss_mode=loss_mode,
            router_mean_target=float(loss_cfg.get("router_mean_target", cfg["model"].get("router_ratio", 0.1))),
            router_mean_weight=float(loss_cfg.get("router_mean_weight", 1.0)),
            router_score_margin=float(loss_cfg.get("router_score_margin", 0.02)),
            router_score_weight=float(loss_cfg.get("router_score_weight", 0.0)),
        )
        elapsed = time.perf_counter() - started
        local_step = step - start_step
        seen_tokens = local_step * int(train_cfg["batch_size"]) * args.seq_len
        throughput = seen_tokens / max(elapsed, 1e-6)
        metrics = {
            "step": step,
            "total_loss": losses.total.item(),
            "main_loss": losses.main.item(),
            "router_loss": losses.router.item(),
            "router_mean": losses.router_mean.item(),
            "router_selected_ratio": losses.router_selected_ratio.item(),
            "router_selected_score_mean": losses.router_selected_score_mean.item(),
            "router_unselected_score_mean": losses.router_unselected_score_mean.item(),
            "router_score_gap": losses.router_score_gap.item(),
            "tokens_per_second": throughput,
            "device": str(device),
            "seq_len": args.seq_len,
            "config": args.config,
            "seed": seed,
            "dataset_type": dataset_type,
            "status": "ok",
        }
        if batch_sources:
            metrics["batch_sources"] = batch_sources
            if len(batch_sources) == 1:
                metrics["batch_source"] = next(iter(batch_sources))
        if not losses_are_finite(losses):
            metrics["status"] = "non_finite_loss"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            batch_source = metrics.get("batch_source", dataset_type)
            raise SystemExit(
                f"Non-finite training loss at step {step} for {args.config} "
                f"(seed={seed}, batch_source={batch_source})"
            )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        grad_norm_value = float(grad_norm)
        metrics["grad_norm"] = grad_norm_value
        if not torch.isfinite(torch.as_tensor(grad_norm)).all():
            metrics["status"] = "non_finite_grad_norm"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            batch_source = metrics.get("batch_source", dataset_type)
            raise SystemExit(
                f"Non-finite gradient norm at step {step} for {args.config} "
                f"(seed={seed}, batch_source={batch_source})"
            )
        optimizer.step()
        if not parameters_are_finite(model):
            metrics["status"] = "non_finite_parameters"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")
            batch_source = metrics.get("batch_source", dataset_type)
            raise SystemExit(
                f"Non-finite model parameters at step {step} for {args.config} "
                f"(seed={seed}, batch_source={batch_source})"
            )

        if step % int(train_cfg["log_every"]) == 0 or step in {1, total_steps}:
            print(
                f"step={step} total_loss={losses.total.item():.4f} "
                f"main_loss={losses.main.item():.4f} router_loss={losses.router.item():.4f} "
                f"router_mean={losses.router_mean.item():.4f} "
                f"router_selected_ratio={losses.router_selected_ratio.item():.4f} "
                f"router_score_gap={losses.router_score_gap.item():.4f} "
                f"tokens_per_second={throughput:.2f}"
            )
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics) + "\n")

    checkpoint_path = Path(args.checkpoint_out) if args.checkpoint_out else checkpoint_dir / f"{architecture}_seq{args.seq_len}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": cfg,
            "seq_len": args.seq_len,
            "step": start_step + total_steps,
        },
        checkpoint_path,
    )
    print(f"device={device}")
    print(f"seed={seed}")
    print(f"log_jsonl={log_path}")
    print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    train()
