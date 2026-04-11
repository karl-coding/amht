from __future__ import annotations

from statistics import mean

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc

from data.dataset import build_retrieval_batch


def build_niah_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_token: int,
    key_start: int,
    value_start: int,
    num_pairs: int,
    num_keys: int,
    value_pool_size: int | None,
    random_value_mapping: bool,
    depth: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_retrieval_batch(
        batch_size=batch_size,
        vocab_size=vocab_size,
        seq_len=seq_len,
        pad_token=pad_token,
        key_start=key_start,
        value_start=value_start,
        num_pairs=num_pairs,
        num_keys=num_keys,
        value_pool_size=value_pool_size,
        random_value_mapping=random_value_mapping,
        target_depth=depth,
        device=device,
    )


def evaluate_niah_hits(logits: torch.Tensor, expected: torch.Tensor) -> float:
    pred = logits[:, -1].argmax(dim=-1)
    return float((pred == expected).float().mean().item())


@torch.no_grad()
def evaluate_niah_hits_chunked(
    model,
    tokens: torch.Tensor,
    expected: torch.Tensor,
    forward_batch_size: int,
) -> float:
    total_correct = 0
    total_items = 0
    for start in range(0, tokens.size(0), forward_batch_size):
        end = min(start + forward_batch_size, tokens.size(0))
        logits, _ = model(tokens[start:end, :-1])
        pred = logits[:, -1].argmax(dim=-1)
        total_correct += int((pred == expected[start:end]).sum().item())
        total_items += end - start
    return float(total_correct) / max(total_items, 1)


@torch.no_grad()
def benchmark_niah(model, cfg: dict, device: torch.device) -> dict:
    niah_cfg = cfg["evaluation"]["niah"]
    batch_size = int(niah_cfg.get("batch_size", cfg["evaluation"]["batch_size"]))
    forward_batch_size = int(niah_cfg.get("forward_batch_size", batch_size))
    repeats = int(niah_cfg.get("repeats", 1))
    scores = []
    for depth in niah_cfg["needle_depths"]:
        depth_scores = []
        for _ in range(repeats):
            tokens, answer_ids = build_niah_batch(
                batch_size=batch_size,
                seq_len=int(niah_cfg["seq_len"]),
                vocab_size=int(cfg["model"]["vocab_size"]),
                pad_token=int(niah_cfg["pad_token"]),
                key_start=int(niah_cfg["key_start"]),
                value_start=int(niah_cfg["value_start"]),
                num_pairs=int(niah_cfg["num_pairs"]),
                num_keys=int(niah_cfg.get("num_keys", niah_cfg["num_pairs"])),
                value_pool_size=(
                    None
                    if niah_cfg.get("value_pool_size") is None
                    else int(niah_cfg["value_pool_size"])
                ),
                random_value_mapping=bool(niah_cfg.get("random_value_mapping", False)),
                depth=float(depth),
                device=device,
            )
            depth_scores.append(
                evaluate_niah_hits_chunked(
                    model=model,
                    tokens=tokens,
                    expected=answer_ids,
                    forward_batch_size=forward_batch_size,
                )
            )
        scores.append(mean(depth_scores) if depth_scores else 0.0)
    return {
        "task": "niah",
        "device": str(device),
        "seq_len": int(niah_cfg["seq_len"]),
        "batch_size": batch_size,
        "forward_batch_size": forward_batch_size,
        "repeats": repeats,
        "needle_depths": list(niah_cfg["needle_depths"]),
        "num_pairs": int(niah_cfg["num_pairs"]),
        "num_keys": int(niah_cfg.get("num_keys", niah_cfg["num_pairs"])),
        "value_pool_size": (
            None
            if niah_cfg.get("value_pool_size") is None
            else int(niah_cfg["value_pool_size"])
        ),
        "random_value_mapping": bool(niah_cfg.get("random_value_mapping", False)),
        "accuracy_by_depth": scores,
        "mean_accuracy": mean(scores) if scores else 0.0,
    }
