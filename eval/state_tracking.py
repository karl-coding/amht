from __future__ import annotations

from statistics import mean

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


def build_state_tracking_batch(
    *,
    batch_size: int,
    seq_len: int,
    modulus: int,
    digit_start: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    digits = torch.randint(
        digit_start,
        digit_start + modulus,
        (batch_size, seq_len - 1),
        dtype=torch.long,
        device=device,
    )
    expected = ((digits - digit_start).sum(dim=1) % modulus) + digit_start
    tokens = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    tokens[:, :-1] = digits
    tokens[:, -1] = expected
    return tokens, expected


@torch.no_grad()
def evaluate_state_tracking_accuracy_chunked(
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
def benchmark_state_tracking(model, cfg: dict, device: torch.device) -> dict:
    tracking_cfg = cfg.get("evaluation", {}).get("state_tracking", {})
    task_name = str(tracking_cfg.get("task", "modsum"))
    if task_name != "modsum":
        raise SystemExit(f"Unsupported evaluation.state_tracking.task={task_name}")

    batch_size = int(tracking_cfg.get("batch_size", 1))
    forward_batch_size = int(tracking_cfg.get("forward_batch_size", batch_size))
    repeats = int(tracking_cfg.get("repeats", 8))
    default_seq_lens = [
        seq_len
        for seq_len in [1024, 4096, 8192, 16384, 32768]
        if seq_len <= int(cfg["model"]["max_seq_len"])
    ]
    seq_lens = [int(seq_len) for seq_len in tracking_cfg.get("seq_lens", default_seq_lens)]
    modulus = int(tracking_cfg.get("modulus", 16))
    digit_start = int(tracking_cfg.get("digit_start", 0))

    scores = []
    results = []
    for seq_len in seq_lens:
        seq_scores = []
        for _ in range(repeats):
            tokens, expected = build_state_tracking_batch(
                batch_size=batch_size,
                seq_len=seq_len,
                modulus=modulus,
                digit_start=digit_start,
                device=device,
            )
            seq_scores.append(
                evaluate_state_tracking_accuracy_chunked(
                    model=model,
                    tokens=tokens,
                    expected=expected,
                    forward_batch_size=forward_batch_size,
                )
            )
        accuracy = mean(seq_scores) if seq_scores else 0.0
        scores.append(accuracy)
        results.append({"seq_len": seq_len, "accuracy": accuracy})

    return {
        "task": "state_tracking",
        "device": str(device),
        "task_name": task_name,
        "batch_size": batch_size,
        "forward_batch_size": forward_batch_size,
        "repeats": repeats,
        "modulus": modulus,
        "digit_start": digit_start,
        "seq_lens": seq_lens,
        "results": results,
        "accuracy_by_seq_len": scores,
        "mean_accuracy": mean(scores) if scores else 0.0,
    }
