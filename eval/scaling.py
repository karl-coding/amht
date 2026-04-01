from __future__ import annotations


def benchmark_scaling(model, cfg: dict, device, throughput_fn) -> dict:
    results = []
    for seq_len in cfg["evaluation"]["scaling_lengths"]:
        metrics = throughput_fn(model, cfg, int(seq_len), device)
        results.append(
            {
                "seq_len": int(seq_len),
                "tokens_per_second": metrics["tokens_per_second"],
                "milliseconds_per_step": metrics["milliseconds_per_step"],
            }
        )
    return {"task": "scaling", "device": str(device), "results": results}
