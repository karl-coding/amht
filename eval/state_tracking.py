from __future__ import annotations

from statistics import mean

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc

from data.dataset import build_state_tracking_batch
from model.amht import AMHTModel


def state_tracking_model_kwargs(model, tracking_cfg: dict) -> dict:
    if not isinstance(model, AMHTModel):
        return {}
    runtime_cfg = tracking_cfg.get("runtime", {})
    return {
        "router_straight_through_enabled": bool(runtime_cfg.get("router_straight_through_enabled", False)),
        "router_attention_enabled": bool(runtime_cfg.get("router_attention_enabled", False)),
        "memory_enabled": bool(runtime_cfg.get("memory_enabled", False)),
    }


@torch.no_grad()
def evaluate_state_tracking_accuracy_chunked(
    model,
    tokens: torch.Tensor,
    expected: torch.Tensor,
    forward_batch_size: int,
    tracking_cfg: dict,
) -> float:
    total_correct = 0
    total_items = 0
    for start in range(0, tokens.size(0), forward_batch_size):
        end = min(start + forward_batch_size, tokens.size(0))
        model_kwargs = state_tracking_model_kwargs(model, tracking_cfg)
        logits, _ = model(tokens[start:end, :-1], **model_kwargs)
        pred = logits[:, -1].argmax(dim=-1)
        total_correct += int((pred == expected[start:end]).sum().item())
        total_items += end - start
    return float(total_correct) / max(total_items, 1)


@torch.no_grad()
def benchmark_state_tracking(model, cfg: dict, device: torch.device) -> dict:
    tracking_cfg = cfg.get("evaluation", {}).get("state_tracking", {})
    task_name = str(tracking_cfg.get("task", "modsum"))

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
    num_slots = int(tracking_cfg.get("num_slots", 8))
    value_count = int(tracking_cfg.get("value_count", 2))
    slot_start = int(tracking_cfg.get("slot_start", 0))
    value_start = tracking_cfg.get("value_start")
    query_start = tracking_cfg.get("query_start")
    min_query_gap_tokens = int(tracking_cfg.get("min_query_gap_tokens", 4096))

    scores = []
    results = []
    for seq_len in seq_lens:
        seq_scores = []
        for _ in range(repeats):
            tokens, expected = build_state_tracking_batch(
                batch_size=batch_size,
                vocab_size=int(cfg["model"]["vocab_size"]),
                seq_len=seq_len,
                task=task_name,
                modulus=modulus,
                digit_start=digit_start,
                num_slots=num_slots,
                value_count=value_count,
                slot_start=slot_start,
                value_start=None if value_start is None else int(value_start),
                query_start=None if query_start is None else int(query_start),
                min_query_gap_tokens=min_query_gap_tokens,
                device=device,
            )
            seq_scores.append(
                evaluate_state_tracking_accuracy_chunked(
                    model=model,
                    tokens=tokens,
                    expected=expected,
                    forward_batch_size=forward_batch_size,
                    tracking_cfg=tracking_cfg,
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
        "num_slots": num_slots,
        "value_count": value_count,
        "slot_start": slot_start,
        "value_start": value_start,
        "query_start": query_start,
        "min_query_gap_tokens": min_query_gap_tokens,
        "seq_lens": seq_lens,
        "results": results,
        "accuracy_by_seq_len": scores,
        "mean_accuracy": mean(scores) if scores else 0.0,
    }
