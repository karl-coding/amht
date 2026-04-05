#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Suggest V4 training and architecture adjustments from a stage-one summary."
    )
    parser.add_argument("--summary", required=True, help="Path to report/summary.json")
    parser.add_argument("--out", default=None, help="Optional path to save the adjustment note")
    return parser.parse_args()


def metric(summary: dict, model_key: str, section: str, field: str) -> float | None:
    value = (
        summary.get("models", {})
        .get(model_key, {})
        .get(section, {})
        .get(field, {})
        .get("mean")
    )
    if isinstance(value, (int, float)):
        return float(value)
    return None


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def pick_best_amht(summary: dict) -> str | None:
    candidates = [
        key
        for key in ("amht_v4_stage1_tuned", "amht_v4_fast", "amht_v4_accurate")
        if key in summary.get("models", {})
    ]
    if not candidates:
        return None

    def score(key: str) -> tuple[float, float]:
        niah = metric(summary, key, "niah", "mean_accuracy")
        tps = metric(summary, key, "throughput", "tokens_per_second")
        return (niah if niah is not None else -1.0, tps if tps is not None else -1.0)

    return max(candidates, key=score)


def build_note(summary: dict) -> str:
    models = summary.get("models", {})
    best_amht = pick_best_amht(summary)
    transformer = "transformer_v4_baseline" if "transformer_v4_baseline" in models else None
    mamba_ref = "mamba3_hybrid_baseline" if "mamba3_hybrid_baseline" in models else None

    lines: list[str] = [
        "# Stage 1 Adjustment Note",
        "",
        "Focus on training comparison first. Do not freeze the paper path until AMHT is consistently ahead of the baselines on the intended quality-efficiency tradeoff.",
        "",
    ]

    if best_amht is None:
        lines.append("No AMHT model found in summary.")
        return "\n".join(lines) + "\n"

    best_label = models[best_amht]["label"]
    best_acc = metric(summary, best_amht, "niah", "mean_accuracy")
    best_tps = metric(summary, best_amht, "throughput", "tokens_per_second")
    best_loss = metric(summary, best_amht, "train", "final_total_loss")

    lines.extend(
        [
            "## Current Best AMHT",
            "",
            f"- Model: `{best_label}`",
            f"- Mean NIAH accuracy: `{fmt(best_acc)}`",
            f"- Throughput (tok/s): `{fmt(best_tps, 2)}`",
            f"- Final train loss: `{fmt(best_loss)}`",
            "",
        ]
    )

    def compare_block(target_key: str, target_title: str) -> list[str]:
        if target_key not in models:
            return []
        target_acc = metric(summary, target_key, "niah", "mean_accuracy")
        target_tps = metric(summary, target_key, "throughput", "tokens_per_second")
        acc_gap = None if best_acc is None or target_acc is None else best_acc - target_acc
        tps_gap = None if best_tps is None or target_tps is None else best_tps - target_tps

        block = [
            f"## AMHT vs {target_title}",
            "",
            f"- Baseline mean NIAH accuracy: `{fmt(target_acc)}`",
            f"- Baseline throughput (tok/s): `{fmt(target_tps, 2)}`",
            f"- Accuracy gap (AMHT - baseline): `{fmt(acc_gap)}`",
            f"- Throughput gap (AMHT - baseline): `{fmt(tps_gap, 2)}`",
            "",
        ]

        if acc_gap is not None and acc_gap < -0.02:
            block.extend(
                [
                    "Recommendation:",
                    "- Backbone is the first suspect. Increase `ssm_state_size`, keep or enable `ssm_complex`, and test `ssm_conv_kernel=5` if not already used.",
                    "- Use the Mamba-3-inspired baseline as a reference for recurrence strength, not as a reproduction target.",
                    "- If the loss is also high, extend steps before widening the router search.",
                    "",
                ]
            )
        elif tps_gap is not None and tps_gap < 0.0 and (acc_gap is None or acc_gap <= 0.02):
            block.extend(
                [
                    "Recommendation:",
                    "- Throughput is losing without a clear quality win. Reduce selective-attention cost first: lower `router_neighbor_radius` or `router_neighbor_bonus`, or increase `block_size`.",
                    "- If memory is heavy, reduce `latent_tokens` before weakening the recurrent backbone.",
                    "",
                ]
            )
        elif acc_gap is not None and acc_gap > 0.0 and tps_gap is not None and tps_gap > 0.0:
            block.extend(
                [
                    "Recommendation:",
                    "- This comparison is already favorable. Keep the architecture stable and test longer steps, harder retrieval, or `16K/32K` before making new architectural changes.",
                    "",
                ]
            )
        else:
            block.extend(
                [
                    "Recommendation:",
                    "- Mixed result. Tune router-learning and memory path next: test `router_straight_through_temperature`, `router_straight_through_scale`, and `latent_tokens` while keeping the backbone fixed for this iteration.",
                    "",
                ]
            )
        return block

    if transformer is not None:
        lines.extend(compare_block(transformer, "Transformer"))
    if mamba_ref is not None:
        lines.extend(compare_block(mamba_ref, "Mamba-3-Inspired Hybrid"))

    lines.extend(
        [
            "## Mamba-3 Reference Axes",
            "",
            "- Use Mamba-3 ideas as parameter-search directions: stronger recurrent state, complex-state dynamics, and larger local mixing kernel.",
            "- For this repo, the most relevant knobs are `ssm_state_size`, `ssm_complex`, `ssm_conv_kernel`, and then router or memory settings after the backbone is stable.",
            "- Do not describe any resulting config as Mamba-3 itself unless the full architecture and training recipe are matched.",
            "",
            "## Next Iteration Order",
            "",
            "1. Stabilize or strengthen the recurrent backbone.",
            "2. Re-run the same baseline comparison.",
            "3. Tune router and latent memory only after backbone gains stop moving.",
            "4. Delay paper-focused freezing until AMHT is consistently ahead on the chosen quality-efficiency target.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    note = build_note(summary)
    print(note, end="")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(note, encoding="utf-8")


if __name__ == "__main__":
    main()
