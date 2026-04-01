#!/usr/bin/env python3
"""Train AMHT with synthetic data for bring-up and benchmarking."""

from __future__ import annotations

import argparse
import importlib.util
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

from data.dataset import RetrievalDataset, SyntheticDataset
from model.amht import AMHTModel, compute_loss

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an AMHT model")
    parser.add_argument("--config", default="train/config.yaml", help="Path to YAML config")
    parser.add_argument("--seq-len", type=int, default=8192, help="Training context length")
    parser.add_argument("--steps", type=int, default=None, help="Optional override for training steps")
    parser.add_argument("--device", default="auto", help="cpu, cuda, mps, or auto")
    return parser.parse_args()


def train() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seq_len > cfg["model"]["max_seq_len"]:
        raise SystemExit(
            f"--seq-len {args.seq_len} exceeds config model.max_seq_len={cfg['model']['max_seq_len']}"
        )

    set_seed(int(cfg.get("seed", 42)))
    device = choose_device(args.device)
    maybe_init_distributed()
    model = AMHTModel(cfg).to(device)

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

    total_steps = args.steps if args.steps is not None else int(train_cfg["steps"])
    data_cfg = cfg.get("data", {})
    dataset_type = data_cfg.get("dataset_type", "synthetic")
    total_samples = total_steps * int(train_cfg["batch_size"])
    if dataset_type == "retrieval":
        niah_cfg = cfg["evaluation"]["niah"]
        dataset = RetrievalDataset(
            vocab_size=int(cfg["model"]["vocab_size"]),
            seq_len=args.seq_len,
            total_samples=total_samples,
            pad_token=int(niah_cfg["pad_token"]),
            key_start=int(niah_cfg["key_start"]),
            value_start=int(niah_cfg["value_start"]),
            num_pairs=int(niah_cfg["num_pairs"]),
            num_keys=int(niah_cfg.get("num_keys", niah_cfg["num_pairs"])),
            depth_choices=[float(depth) for depth in niah_cfg.get("needle_depths", [0.1, 0.3, 0.5, 0.7, 0.9])],
        )
    else:
        dataset = SyntheticDataset(
            vocab_size=int(cfg["model"]["vocab_size"]),
            seq_len=args.seq_len,
            total_samples=total_samples,
        )

    model.train()
    started = time.perf_counter()
    for step in range(1, total_steps + 1):
        start = (step - 1) * int(train_cfg["batch_size"])
        stop = step * int(train_cfg["batch_size"])
        batch = [dataset[index] for index in range(start, stop)]
        tokens = torch.stack(batch, dim=0).to(device)
        losses = compute_loss(
            model=model,
            tokens=tokens,
            main_weight=float(loss_cfg["main_weight"]),
            router_weight=float(loss_cfg["router_weight"]),
            loss_mode=loss_mode,
        )
        optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
        optimizer.step()

        if step % int(train_cfg["log_every"]) == 0 or step in {1, total_steps}:
            elapsed = time.perf_counter() - started
            seen_tokens = step * int(train_cfg["batch_size"]) * args.seq_len
            throughput = seen_tokens / max(elapsed, 1e-6)
            print(
                f"step={step} total_loss={losses.total.item():.4f} "
                f"main_loss={losses.main.item():.4f} router_loss={losses.router.item():.4f} "
                f"router_mean={losses.router_mean.item():.4f} tokens_per_second={throughput:.2f}"
            )

    checkpoint_path = checkpoint_dir / f"amht_seq{args.seq_len}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": cfg,
            "seq_len": args.seq_len,
        },
        checkpoint_path,
    )
    print(f"device={device}")
    print(f"saved_checkpoint={checkpoint_path}")


if __name__ == "__main__":
    train()
