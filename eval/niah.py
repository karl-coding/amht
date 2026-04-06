from __future__ import annotations

from statistics import mean

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


def build_niah_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    pad_token: int,
    key_start: int,
    value_start: int,
    num_pairs: int,
    num_keys: int,
    depth: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.randint(value_start + num_keys + 10, vocab_size, (batch_size, seq_len), device=device)
    tokens[:, 0] = pad_token
    expected = torch.empty((batch_size,), dtype=torch.long, device=device)

    base_positions = [
        min(seq_len - 4, max(1, int(seq_len * depth))),
        min(seq_len - 8, max(2, int(seq_len * 0.25))),
        min(seq_len - 12, max(3, int(seq_len * 0.75))),
        min(seq_len - 16, max(4, int(seq_len * 0.4))),
    ]
    while len(base_positions) < num_pairs:
        base_positions.append(min(seq_len - 20 - 2 * len(base_positions), seq_len - 4))

    for batch_idx in range(batch_size):
        permutation = torch.randperm(num_keys, device=device)[:num_pairs]
        target_idx = torch.randint(0, num_pairs, (1,), device=device).item()
        for pair_idx in range(num_pairs):
            pos = min(base_positions[pair_idx], seq_len - 3)
            key_id = permutation[pair_idx].item()
            tokens[batch_idx, pos] = key_start + key_id
            tokens[batch_idx, pos + 1] = value_start + key_id
        target_key = permutation[target_idx].item()
        tokens[batch_idx, -2] = key_start + target_key
        expected[batch_idx] = value_start + target_key
        tokens[batch_idx, -1] = expected[batch_idx]
    return tokens, expected


def evaluate_niah_hits(logits: torch.Tensor, expected: torch.Tensor) -> float:
    pred = logits[:, -1].argmax(dim=-1)
    return float((pred == expected).float().mean().item())


@torch.no_grad()
def benchmark_niah(model, cfg: dict, device: torch.device) -> dict:
    niah_cfg = cfg["evaluation"]["niah"]
    batch_size = int(niah_cfg.get("batch_size", cfg["evaluation"]["batch_size"]))
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
                depth=float(depth),
                device=device,
            )
            logits, _ = model(tokens[:, :-1])
            depth_scores.append(evaluate_niah_hits(logits=logits, expected=answer_ids))
        scores.append(mean(depth_scores) if depth_scores else 0.0)
    return {
        "task": "niah",
        "device": str(device),
        "seq_len": int(niah_cfg["seq_len"]),
        "batch_size": batch_size,
        "repeats": repeats,
        "needle_depths": list(niah_cfg["needle_depths"]),
        "accuracy_by_depth": scores,
        "mean_accuracy": mean(scores) if scores else 0.0,
    }
