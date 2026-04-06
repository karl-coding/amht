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
        for key in (
            "amht_v4_stage2_round5",
            "amht_v4_stage2_round4",
            "amht_v4_stage2_round3",
            "amht_v4_stage2_round2",
            "amht_v4_stage2_round1",
            "amht_v4_stage1_round4_long",
            "amht_v4_stage1_round4",
            "amht_v4_stage1_round3",
            "amht_v4_stage1_tuned",
            "amht_v4_fast",
            "amht_v4_accurate",
        )
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
    quality_tie_tolerance = 0.02

    if best_amht and "stage2" in best_amht:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round4_baseline",
                    "transformer_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        mamba_ref = next(
            (
                key
                for key in (
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        intro = "Focus on hybrid specialization next. Keep the recurrent backbone fixed unless harder retrieval breaks the quality-efficiency tradeoff badly."
        favorable_line = "- This comparison is favorable for stage two: same-or-better quality at higher throughput on the harder retrieval setting."
    else:
        stage_label = "1"
        transformer = "transformer_v4_baseline" if "transformer_v4_baseline" in models else None
        mamba_ref = "mamba3_hybrid_baseline" if "mamba3_hybrid_baseline" in models else None
        intro = "Focus on training comparison first. Do not freeze the paper path until AMHT is consistently ahead of the baselines on the intended quality-efficiency tradeoff."
        favorable_line = "- This comparison already matches the stage-one tradeoff target: same-or-better quality at higher throughput."

    lines: list[str] = [
        f"# Stage {stage_label} Adjustment Note",
        "",
        intro,
        "",
    ]

    if best_amht is None:
        lines.append("No AMHT model found in summary.")
        return "\n".join(lines) + "\n"

    best_label = models[best_amht]["label"]
    best_acc = metric(summary, best_amht, "niah", "mean_accuracy")
    best_tps = metric(summary, best_amht, "throughput", "tokens_per_second")
    best_loss = metric(summary, best_amht, "train", "final_total_loss")
    best_router_loss = metric(summary, best_amht, "train", "final_router_loss")
    best_router_mean = metric(summary, best_amht, "train", "final_router_mean")
    best_router_selected_ratio = metric(summary, best_amht, "train", "final_router_selected_ratio")
    best_router_selected_score_mean = metric(summary, best_amht, "train", "final_router_selected_score_mean")
    best_router_unselected_score_mean = metric(summary, best_amht, "train", "final_router_unselected_score_mean")
    best_router_score_gap = metric(summary, best_amht, "train", "final_router_score_gap")
    router_score_collapsed = best_router_score_gap is not None and best_router_score_gap < quality_tie_tolerance
    comparison_gaps: list[tuple[float | None, float | None]] = []

    lines.extend(
        [
            "## Current Best AMHT",
            "",
            f"- Model: `{best_label}`",
            f"- Mean NIAH accuracy: `{fmt(best_acc)}`",
            f"- Throughput (tok/s): `{fmt(best_tps, 2)}`",
            f"- Final train loss: `{fmt(best_loss)}`",
            f"- Final router loss: `{fmt(best_router_loss)}`",
            f"- Final router mean: `{fmt(best_router_mean)}`",
            f"- Final selected ratio: `{fmt(best_router_selected_ratio)}`",
            f"- Final selected score mean: `{fmt(best_router_selected_score_mean)}`",
            f"- Final unselected score mean: `{fmt(best_router_unselected_score_mean)}`",
            f"- Final router score gap: `{fmt(best_router_score_gap)}`",
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
        comparison_gaps.append((acc_gap, tps_gap))

        block = [
            f"## AMHT vs {target_title}",
            "",
            f"- Baseline mean NIAH accuracy: `{fmt(target_acc)}`",
            f"- Baseline throughput (tok/s): `{fmt(target_tps, 2)}`",
            f"- Accuracy gap (AMHT - baseline): `{fmt(acc_gap)}`",
            f"- Throughput gap (AMHT - baseline): `{fmt(tps_gap, 2)}`",
            "",
        ]

        if router_score_collapsed and acc_gap is not None and acc_gap >= 0.0 and tps_gap is not None and tps_gap > 0.0:
            block.extend(
                [
                    "Recommendation:",
                    *(
                        [
                            "- Quality and throughput are acceptable, and actual selected ratio is controlled by top-k. The issue is score calibration: selected and unselected blocks are still too close under the harder stage-two setting.",
                            "- Keep the current straight-through settings, then raise `router_score_margin` and `router_score_weight` so the score-separation loss stays meaningful near the margin.",
                            "- Increase NIAH sampling density with `evaluation.niah.batch_size` or `evaluation.niah.repeats` before reading small quality moves as real.",
                            "- Keep the stronger backbone fixed for this iteration so the next run isolates router and memory behavior.",
                        ]
                        if stage_label == "2"
                        else [
                            "- Quality and throughput are acceptable, and actual selected ratio is controlled by top-k. The issue is score calibration: selected and unselected blocks are no longer well separated.",
                            "- Tune router-learning next: lower `router_straight_through_temperature`, raise `router_straight_through_scale`, and consider increasing `router_weight` only if score separation stays weak.",
                            "- Keep the stronger backbone fixed for this iteration so the next run isolates router and memory behavior.",
                        ]
                    ),
                    "",
                ]
            )
        elif acc_gap is not None and acc_gap < -quality_tie_tolerance:
            block.extend(
                [
                    "Recommendation:",
                    *(
                        [
                            "- Hybrid specialization is the first suspect. Keep the round-four backbone fixed and tune `router_neighbor_radius`, `router_neighbor_bonus`, `latent_tokens`, or memory I/O first.",
                            "- Re-open backbone changes only if the harder retrieval setting still misses quality after at least one router or memory iteration.",
                        ]
                        if stage_label == "2"
                        else [
                            "- Backbone is the first suspect. Increase recurrent capacity first: raise `ssm_state_size`, or add one more SSM layer if state size is already high enough.",
                            "- Keep `ssm_complex` and `ssm_conv_kernel=5` if they are already enabled; only revisit them if the next capacity run still misses quality.",
                            "- Use the Mamba-3-inspired baseline as a reference for recurrence strength, not as a reproduction target.",
                            "- If the loss is also high, extend steps before widening attention or memory changes.",
                        ]
                    ),
                    "",
                ]
            )
        elif tps_gap is not None and tps_gap < 0.0 and (acc_gap is None or acc_gap <= quality_tie_tolerance):
            block.extend(
                [
                    "Recommendation:",
                    "- Throughput is losing without a clear quality win. Reduce selective-attention cost first: lower `router_neighbor_radius` or `router_neighbor_bonus`, or increase `block_size`.",
                    "- If memory is heavy, reduce `latent_tokens` before weakening the recurrent backbone.",
                    "",
                ]
            )
        elif acc_gap is not None and acc_gap >= -quality_tie_tolerance and tps_gap is not None and tps_gap > 0.0:
            block.extend(
                [
                    "Recommendation:",
                    favorable_line,
                    "- Keep the architecture stable and validate it next with extra seeds, harder retrieval, or `16K/32K` before making new architectural changes.",
                    "",
                ]
            )
        else:
            block.extend(
                [
                    "Recommendation:",
                    "- Mixed result. Keep the backbone fixed and test longer training or memory/context changes next: `latent_tokens`, `router_neighbor_radius`, `router_neighbor_bonus`, or memory I/O.",
                    "- Do not retune straight-through router settings unless `router_score_gap` falls back below the margin.",
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
        ]
    )
    quality_deficit = any(acc_gap is not None and acc_gap < -quality_tie_tolerance for acc_gap, _ in comparison_gaps)
    throughput_deficit = any(
        tps_gap is not None and tps_gap < 0.0 and (acc_gap is None or acc_gap <= quality_tie_tolerance)
        for acc_gap, tps_gap in comparison_gaps
    )
    favorable_tradeoff = bool(comparison_gaps) and all(
        acc_gap is not None and acc_gap >= -quality_tie_tolerance and tps_gap is not None and tps_gap > 0.0
        for acc_gap, tps_gap in comparison_gaps
    )
    if quality_deficit:
        lines.extend(
            [
                *(
                    [
                        "1. Keep the round-four backbone fixed and tune router or memory behavior first.",
                        "2. Re-run the same harder retrieval comparison.",
                        "3. Re-open backbone changes only if hybrid tuning fails to recover quality.",
                        "4. Delay paper-focused freezing until AMHT stays ahead on the harder retrieval setting.",
                    ]
                    if stage_label == "2"
                    else [
                        "1. Stabilize or strengthen the recurrent backbone.",
                        "2. Re-run the same baseline comparison.",
                        "3. Tune router and latent memory only after backbone gains stop moving.",
                        "4. Delay paper-focused freezing until AMHT is consistently ahead on the chosen quality-efficiency target.",
                    ]
                ),
                "",
            ]
        )
    elif throughput_deficit:
        lines.extend(
            [
                "1. Reduce selective-attention or memory overhead before changing the backbone again.",
                "2. Re-run the same baseline comparison at the same sequence length and steps.",
                "3. Tune router score separation only after throughput recovers or if quality still stalls.",
                "4. Return to backbone changes only if efficiency tuning fails to recover the tradeoff.",
                "",
            ]
        )
    elif router_score_collapsed:
        lines.extend(
            [
                *(
                    [
                        "1. Increase NIAH sampling density so stage-two quality changes are measurable without changing throughput benchmarking.",
                        "2. Improve router score separation while keeping the top-k compute budget fixed.",
                        "3. Tune latent memory only if score separation improves but quality stays flat.",
                        "4. Return to backbone changes only if stronger router supervision still fails to improve retrieval.",
                    ]
                    if stage_label == "2"
                    else [
                        "1. Improve router score separation while keeping the top-k compute budget fixed.",
                        "2. Re-run the same baseline comparison at the same sequence length and steps.",
                        "3. Tune latent memory only if score separation improves but quality stays flat.",
                        "4. Return to backbone changes only if router tuning fails to improve retrieval.",
                    ]
                ),
                "",
            ]
        )
    elif favorable_tradeoff:
        lines.extend(
            [
                *(
                    [
                        "1. Keep the round-four backbone fixed as the stage-two starting point.",
                        "2. Tune router or memory specialization only if harder retrieval still leaves quality headroom.",
                        "3. Validate at `16K/32K` and add distribution shift before making new architectural changes.",
                        "4. Re-open backbone tuning only if the harder retrieval setting breaks the quality-efficiency tradeoff.",
                    ]
                    if stage_label == "2"
                    else [
                        "1. Keep the round-four backbone fixed as the current stage-one winner.",
                        "2. Validate the same comparison with extra seeds and longer contexts.",
                        "3. Test harder retrieval or `16K/32K` before making new architectural changes.",
                        "4. Re-open backbone or router tuning only if validation breaks the quality-efficiency tradeoff.",
                    ]
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                *(
                    [
                        "1. Keep the round-four backbone fixed and tune router or memory specialization first.",
                        "2. Re-run the same harder retrieval comparison.",
                        "3. Add longer-context or held-out retrieval only after hybrid gains stop moving.",
                        "4. Re-open backbone changes only if stage-two tuning fails to improve the tradeoff.",
                    ]
                    if stage_label == "2"
                    else [
                        "1. Stabilize or strengthen the recurrent backbone.",
                        "2. Re-run the same baseline comparison.",
                        "3. Tune router and latent memory only after backbone gains stop moving.",
                        "4. Delay paper-focused freezing until AMHT is consistently ahead on the chosen quality-efficiency target.",
                    ]
                ),
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
