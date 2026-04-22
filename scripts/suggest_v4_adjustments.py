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


def spread(summary: dict, model_key: str, section: str, field: str) -> float | None:
    value = (
        summary.get("models", {})
        .get(model_key, {})
        .get(section, {})
        .get(field, {})
        .get("std")
    )
    if isinstance(value, (int, float)):
        return float(value)
    return None


def has_variation(summary: dict, model_key: str) -> bool:
    return any(
        (value := spread(summary, model_key, section, field)) is not None and value > 0.0
        for section, field in (
            ("niah", "mean_accuracy"),
            ("state_tracking", "mean_accuracy"),
            ("throughput", "tokens_per_second"),
        )
    )


def summary_seed_count(summary: dict) -> int | None:
    value = summary.get("seed_count")
    if isinstance(value, int) and value > 0:
        return value
    return None


def completed_runs(summary: dict, model_key: str | None) -> int | None:
    if not model_key:
        return None
    value = summary.get("models", {}).get(model_key, {}).get("completed_runs")
    if isinstance(value, int) and value >= 0:
        return value
    return None


def has_complete_seed_coverage(summary: dict, model_key: str | None, expected: int | None) -> bool:
    if expected is None or expected <= 0:
        return False
    covered = completed_runs(summary, model_key)
    return covered is not None and covered >= expected


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def amht_candidates(summary: dict) -> list[str]:
    models = summary.get("models", {})
    preferred_order = [
        key
        for key in (
            "amht_v4_stage2_round19_content_path_long_stability_retry",
            "amht_v4_stage2_round19_content_path",
            "amht_v4_stage2_round18_content_retrieval",
            "amht_v4_stage2_round17_state_memory_diag",
            "amht_v4_stage2_round16",
            "amht_v4_stage2_round15",
            "amht_v4_stage2_round14",
            "amht_v4_stage2_round13",
            "amht_v4_stage2_round12_retry",
            "amht_v4_stage2_round12",
            "amht_v4_stage2_round11_retry",
            "amht_v4_stage2_round11_state_tracking_diag",
            "amht_v4_stage2_round11",
            "amht_v4_stage2_round10",
            "amht_v4_stage2_round9",
            "amht_v4_stage2_round8",
            "amht_v4_stage2_round7_retry",
            "amht_v4_stage2_round7",
            "amht_v4_stage2_round7_state_tracking_diag",
            "amht_v4_stage2_round6",
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
        if key in models
    ]
    dynamic_keys = sorted(
        key
        for key in models
        if key.startswith("amht_v4_stage2_") or key.startswith("amht_v4_stage1_")
    )
    candidates = []
    seen: set[str] = set()
    for key in [*preferred_order, *dynamic_keys]:
        if key in seen:
            continue
        seen.add(key)
        candidates.append(key)
    return candidates


def pick_present_amht(summary: dict) -> str | None:
    candidates = amht_candidates(summary)
    return candidates[0] if candidates else None


def pick_best_amht(summary: dict) -> str | None:
    candidates = amht_candidates(summary)
    candidates = [
        key
        for key in candidates
        if any(
            metric(summary, key, section, field) is not None
            for section, field in (
                ("niah", "mean_accuracy"),
                ("state_tracking", "mean_accuracy"),
                ("throughput", "tokens_per_second"),
            )
        )
    ]
    if not candidates:
        return None

    def score(key: str) -> tuple[float, float]:
        state = metric(summary, key, "state_tracking", "mean_accuracy")
        niah = metric(summary, key, "niah", "mean_accuracy")
        tps = metric(summary, key, "throughput", "tokens_per_second")
        return (
            niah if niah is not None else -1.0,
            state if state is not None else -1.0,
            tps if tps is not None else -1.0,
        )

    return max(candidates, key=score)


def build_note(summary: dict) -> str:
    models = summary.get("models", {})
    best_amht = pick_best_amht(summary)
    present_amht = pick_present_amht(summary)
    active_amht = best_amht or present_amht
    expected_seed_count = summary_seed_count(summary)
    quality_tie_tolerance = 0.02
    round13_validation_complete = any(
        key in models and has_variation(summary, key)
        for key in (
            "amht_v4_stage2_round13",
            "transformer_v4_stage2_round13_baseline",
            "mamba3_hybrid_v4_stage2_round13_baseline",
        )
    )
    diagnostic_mode = active_amht is not None and (
        "state_tracking_diag" in active_amht or "state_memory_diag" in active_amht
    )
    content_path_mode = active_amht is not None and "stage2_round19_content_path" in active_amht
    content_path_retry_mode = (
        active_amht is not None and "stage2_round19_content_path_long_stability_retry" in active_amht
    )
    content_path_retry_validated = (
        content_path_retry_mode and active_amht is not None and has_variation(summary, active_amht)
    )
    content_retrieval_mode = active_amht is not None and any(
        tag in active_amht for tag in ("stage2_round18_content_retrieval", "stage2_round19_content_path")
    )
    state_memory_diag_mode = active_amht is not None and "state_memory_diag" in active_amht
    validated_stage2_mode = (
        active_amht is not None and "stage2_round13" in active_amht and round13_validation_complete
    )
    validation_mode = (
        active_amht is not None and "stage2_round13" in active_amht and not round13_validation_complete
    )
    post_budget_mode = active_amht is not None and "stage2_round10" in active_amht
    state_memory_mode = active_amht is not None and "stage2_round16" in active_amht
    stable_mixed_mode = active_amht is not None and any(
        tag in active_amht
        for tag in (
            "stage2_round16",
            "stage2_round13",
            "stage2_round15",
            "stage2_round14",
            "stage2_round12_retry",
            "stage2_round12",
            "stage2_round11_retry",
            "stage2_round11",
            "stage2_round10",
            "stage2_round7_retry",
            "stage2_round8",
            "stage2_round9",
        )
    )
    content_path_retry_validated = False
    content_path_retry_amht_reproduced_only = False
    incomplete_retry_baselines: list[str] = []

    if content_path_mode:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round19_content_path_baseline",
                    "transformer_v4_stage2_round18_content_retrieval_baseline",
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round19_content_path_baseline",
                    "mamba3_hybrid_v4_stage2_round18_content_retrieval_baseline",
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        if content_path_retry_mode:
            amht_seed_complete = has_complete_seed_coverage(summary, active_amht, expected_seed_count)
            transformer_seed_complete = has_complete_seed_coverage(summary, transformer, expected_seed_count) if transformer else True
            mamba_seed_complete = has_complete_seed_coverage(summary, mamba_ref, expected_seed_count) if mamba_ref else True
            if transformer and not transformer_seed_complete:
                incomplete_retry_baselines.append("Transformer")
            if mamba_ref and not mamba_seed_complete:
                incomplete_retry_baselines.append("Mamba-3-inspired hybrid")
            content_path_retry_validated = (
                amht_seed_complete
                and transformer_seed_complete
                and mamba_seed_complete
            )
            content_path_retry_amht_reproduced_only = amht_seed_complete and not content_path_retry_validated

        if content_path_retry_validated:
            intro = "Validation is complete. The long-budget stability-retry configuration is now reproducible, so round19 can serve as the validated AMHT reference before any recurrent-capacity sweep."
            favorable_line = "- The long-budget stability-retry configuration is now validated as competitive, so do not reopen the content path unless the multi-seed result breaks."
        elif content_path_retry_amht_reproduced_only:
            intro = "AMHT long-budget stability is now reproduced across seeds, but the baseline eval set is still incomplete, so round19 is not fully validated yet."
            favorable_line = "- The AMHT stability-retry result is reproducible, but you still need the missing baseline evals before freezing this comparison."
        elif content_path_retry_mode:
            intro = "Focus on validation next. The stability-retry configuration completed the long-budget run at `1600 steps`, so do not open another architecture axis until it survives extra seeds."
            favorable_line = "- This long-budget stability-retry run is promising, but it is still a single-seed result and should be validated before more tuning."
        else:
            intro = "This round keeps the corrected retrieval benchmark fixed and tunes only the AMHT content path. The intent is to import Transformer-style lookup discipline before reopening recurrent-capacity changes."
            favorable_line = "- This content-path sweep is favorable: AMHT improves leakage-free retrieval without giving away its throughput edge."
    elif content_retrieval_mode:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round19_content_path_baseline",
                    "transformer_v4_stage2_round18_content_retrieval_baseline",
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round19_content_path_baseline",
                    "mamba3_hybrid_v4_stage2_round18_content_retrieval_baseline",
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        intro = "This round fixes the old retrieval leakage. The query token no longer determines the answer by construction, so the reported `NIAH` score now measures real context retrieval instead of lexical translation."
        favorable_line = "- This corrected retrieval benchmark is favorable: AMHT stays competitive on leakage-free retrieval while preserving the throughput edge."
    elif diagnostic_mode:
        stage_label = "2 Diagnostic"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round17_state_memory_diag_baseline",
                    "transformer_v4_stage2_round11_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round7_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round17_state_memory_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round11_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round7_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        if state_memory_diag_mode:
            intro = "Focus on latent-memory diagnosis first. This round removes retrieval training and asks whether AMHT gains a real state-tracking edge once memory is the only AMHT-specific path left active."
            favorable_line = "- This diagnostic is favorable: AMHT shows a clear pure state-tracking edge under the memory-on-state setup, so the latent-memory path is earning its keep."
        else:
            intro = "Focus on recurrent-state diagnosis first. Mixed retrieval training should only be retried after the pure state-tracking path is numerically stable."
            favorable_line = "- This diagnostic is favorable: AMHT is stable and already shows a clear state-tracking edge before mixed training is reintroduced."
    elif state_memory_mode:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        intro = "This round pivots from arbitrary hard retrieval to state-memory specialization. Retrieval is now a guardrail, while the main question is whether AMHT benefits once latent memory is actually trained on the state-sensitive batches."
        favorable_line = "- This pivot is favorable: AMHT is finally being tested on a regime that matches its recurrent-plus-memory thesis instead of only on arbitrary final-query retrieval."
    elif validated_stage2_mode:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        intro = "Validation is complete. The long-budget stable mix is now reproducible, but the current benchmark still does not show a decisive AMHT win over the baselines."
        favorable_line = "- The architecture is now validated as competitive, but not yet differentiated enough to justify another architecture axis on this benchmark."
    elif validation_mode:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
                    "mamba3_hybrid_v4_stage2_round4_baseline",
                    "mamba3_hybrid_v4_stage2_baseline",
                )
                if key in models
            ),
            None,
        )
        intro = "Focus on validation next. The long-budget stable mix is now the reference, so do not open another architecture axis until the single-seed result survives extra seeds."
        favorable_line = "- This long-budget run is promising, but it is still a single-seed result and should be validated before more tuning."
    elif best_amht and "stage2" in best_amht:
        stage_label = "2"
        transformer = next(
            (
                key
                for key in (
                    "transformer_v4_stage2_round16_baseline",
                    "transformer_v4_stage2_round15_baseline",
                    "transformer_v4_stage2_round13_baseline",
                    "transformer_v4_stage2_round14_baseline",
                    "transformer_v4_stage2_round11_state_tracking_diag_baseline",
                    "transformer_v4_stage2_round11_retry_baseline",
                    "transformer_v4_stage2_round11_baseline",
                    "transformer_v4_stage2_round10_baseline",
                    "transformer_v4_stage2_round7_retry_baseline",
                    "transformer_v4_stage2_round7_baseline",
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
                    "mamba3_hybrid_v4_stage2_round16_baseline",
                    "mamba3_hybrid_v4_stage2_round15_baseline",
                    "mamba3_hybrid_v4_stage2_round13_baseline",
                    "mamba3_hybrid_v4_stage2_round14_baseline",
                    "mamba3_hybrid_v4_stage2_round11_state_tracking_diag_baseline",
                    "mamba3_hybrid_v4_stage2_round11_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round11_baseline",
                    "mamba3_hybrid_v4_stage2_round10_baseline",
                    "mamba3_hybrid_v4_stage2_round7_retry_baseline",
                    "mamba3_hybrid_v4_stage2_round7_baseline",
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
        if present_amht is None:
            lines.append("No AMHT model found in summary.")
            return "\n".join(lines) + "\n"

        present_label = models.get(present_amht, {}).get("label", present_amht)
        lines.extend(
            [
                "## AMHT Status",
                "",
                f"- Model: `{present_label}`",
                "- The AMHT run is present in the summary bundle, but it did not produce usable eval metrics.",
                "- Treat this as an AMHT run failure or missing `eval.json`, not as a valid quality comparison against the baselines.",
                "",
            ]
        )
        if content_path_mode:
            lines.extend(
                [
                    "Recommendation:",
                    "- For `stage2_round19_content_path`, read this outcome as a training or evaluation failure on the AMHT side, not as evidence that the baselines won the comparison.",
                    "- Keep the `800-step` reproducibility check separate from the `1600-step` long-stability run. If only the longer run fails, classify it as an optimization-stability problem.",
                    "- Fix the AMHT stability issue first, then rerun `stage2_round19_content_path_long_stability_retry` before updating any retrieval-quality claim.",
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Recommendation:",
                    "- Fix the missing or failed AMHT run first, then rerun the same preset before reading architecture conclusions from this report.",
                    "",
                ]
            )
        return "\n".join(lines) + "\n"

    best_label = models[best_amht]["label"]
    best_acc = metric(summary, best_amht, "niah", "mean_accuracy")
    best_state_acc = metric(summary, best_amht, "state_tracking", "mean_accuracy")
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
            f"- Mean state-tracking accuracy: `{fmt(best_state_acc)}`",
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
        target_state_acc = metric(summary, target_key, "state_tracking", "mean_accuracy")
        target_tps = metric(summary, target_key, "throughput", "tokens_per_second")
        acc_gap = None if best_acc is None or target_acc is None else best_acc - target_acc
        state_gap = None if best_state_acc is None or target_state_acc is None else best_state_acc - target_state_acc
        tps_gap = None if best_tps is None or target_tps is None else best_tps - target_tps
        comparison_gaps.append((acc_gap, tps_gap))

        block = [
            f"## AMHT vs {target_title}",
            "",
            f"- Baseline mean NIAH accuracy: `{fmt(target_acc)}`",
            f"- Baseline mean state-tracking accuracy: `{fmt(target_state_acc)}`",
            f"- Baseline throughput (tok/s): `{fmt(target_tps, 2)}`",
            f"- Accuracy gap (AMHT - baseline): `{fmt(acc_gap)}`",
            f"- State-tracking gap (AMHT - baseline): `{fmt(state_gap)}`",
            f"- Throughput gap (AMHT - baseline): `{fmt(tps_gap, 2)}`",
            "",
        ]

        if content_path_retry_validated:
            block.extend(
                [
                    "Recommendation:",
                    "- The long-budget stability-retry result is now reproduced across seeds and can replace the earlier `800-step` readout as the active AMHT reference.",
                    "- Freeze `stage2_round19_content_path_long_stability_retry` and only then open a Mamba-inspired recurrent sweep with `ssm_state_size` if an unexplained state or continuation gap still matters.",
                    "",
                ]
            )
        elif content_path_retry_amht_reproduced_only:
            missing_labels = ", ".join(incomplete_retry_baselines) if incomplete_retry_baselines else "baseline models"
            block.extend(
                [
                    "Recommendation:",
                    "- AMHT long-budget stability is now reproduced across seeds, but the baseline eval set is still incomplete.",
                    f"- Complete the missing `{missing_labels}` evals for this preset before freezing `stage2_round19_content_path_long_stability_retry` or opening any recurrent-capacity sweep.",
                    "",
                ]
            )
        elif content_path_retry_mode:
            block.extend(
                [
                    "Recommendation:",
                    "- The long-budget stability-retry run is now the active AMHT reference, but it is still only one seed.",
                    "- Run `stage2_round19_content_path_long_stability_retry_validate` next before reopening router, memory, or recurrent-capacity tuning.",
                    "",
                ]
            )
        elif content_retrieval_mode:
            block.extend(
                [
                    "Recommendation:",
                    *(
                        [
                            "- The corrected retrieval benchmark is finally measuring real context lookup, and AMHT is ahead on both quality and throughput.",
                            "- Freeze the architecture and validate this result across seeds before reopening router or memory tuning.",
                        ]
                        if acc_gap is not None
                        and acc_gap >= 0.03
                        and tps_gap is not None
                        and tps_gap > 0.0
                        else [
                            "- The corrected retrieval benchmark is informative, but this baseline still has the stronger overall tradeoff on leakage-free retrieval.",
                            "- Do not reopen latent-memory claims here. Keep `stage2_round13` only as the stable efficiency reference and use corrected retrieval as the main gate for further tuning.",
                        ]
                        if acc_gap is not None
                        and acc_gap < -quality_tie_tolerance
                        else [
                            "- The corrected retrieval benchmark is now meaningful, and AMHT is at least competitive on it.",
                            "- Validate this round before any new architecture change; only reopen router supervision if the corrected retrieval gap stays small but reproducible.",
                        ]
                    ),
                    "",
                ]
            )
        elif validated_stage2_mode:
            block.extend(
                [
                    "Recommendation:",
                    *(
                        [
                            "- Validation says AMHT is at least competitive with this baseline and faster on the current benchmark.",
                            "- Keep the architecture frozen and move to harder retrieval or longer-context evaluation before reopening tuning.",
                        ]
                        if acc_gap is not None
                        and acc_gap >= -quality_tie_tolerance
                        and state_gap is not None
                        and state_gap >= -0.01
                        and tps_gap is not None
                        and tps_gap > 0.0
                        else [
                            "- Validation says this baseline still holds the stronger overall tradeoff on the current benchmark.",
                            "- Do not reopen backbone, router, or memory tuning to chase a marginal gap here; move next to harder retrieval or longer-context evaluation where the hybrid design can either separate or be retired.",
                        ]
                        if tps_gap is not None
                        and tps_gap < 0.0
                        and (state_gap is None or state_gap <= 0.0)
                        and (acc_gap is None or acc_gap <= quality_tie_tolerance)
                        else [
                            "- Validation says AMHT is competitive with this baseline, but the margin is too small to justify more architecture churn on the current benchmark.",
                            "- Keep the architecture frozen and move to harder retrieval or longer-context evaluation before reopening tuning.",
                        ]
                    ),
                    "",
                ]
            )
        elif validation_mode:
            block.extend(
                [
                    "Recommendation:",
                    "- The long-budget run is informative, but it is still only one seed. Freeze the architecture and validate it before reopening backbone, router, or memory tuning.",
                    "- Run `stage2_round13_validate` next to check whether the retrieval and state-tracking pattern survives across seeds.",
                    "",
                ]
            )
        elif post_budget_mode:
            block.extend(
                [
                    "Recommendation:",
                    "- This longer-budget run is the clear readout: retrieval improved, but the state-tracking benchmark still leaves all models clustered near chance.",
                    "- Keep the stable retry architecture fixed and redesign or simplify the benchmark before reopening backbone, router, or memory tuning.",
                    "",
                ]
            )
        elif diagnostic_mode:
            if state_memory_diag_mode and state_gap is not None and state_gap >= 0.03:
                block.extend(
                    [
                        "Recommendation:",
                        "- The pure state-memory diagnostic is favorable: AMHT now shows the intended state-sensitive edge without retrieval training.",
                        "- Validate this diagnostic next and add an AMHT memory-off ablation before reintroducing mixed retrieval.",
                        "",
                    ]
                )
            elif state_memory_diag_mode:
                block.extend(
                    [
                        "Recommendation:",
                        "- This pure state-memory diagnostic does not yet show the intended AMHT advantage over the baselines.",
                        "- Do not reintroduce mixed retrieval yet; first decide whether the memory-on-state path needs a simpler diagnostic or should be retired.",
                        "",
                    ]
                )
            elif state_gap is not None and state_gap >= 0.03:
                block.extend(
                    [
                        "Recommendation:",
                        "- Pure state-tracking is now numerically stable and already shows the intended separation.",
                        "- Retry mixed `stage2_round7` next and use retrieval as the holdout guardrail.",
                        "",
                    ]
                )
            else:
                block.extend(
                    [
                        "Recommendation:",
                        "- The pure state-tracking path is stable now, but the benchmark is still near chance and does not separate the models yet.",
                        "- Retry mixed `stage2_round7` next to see whether retrieval-aligned training improves the recurrent signal.",
                        "- If mixed training is also stable but state-tracking stays near chance, increase task budget or simplify the benchmark before reopening router or memory tuning.",
                        "",
                    ]
                )
        elif (
            state_gap is not None
            and state_gap >= 0.03
            and acc_gap is not None
            and acc_gap >= -quality_tie_tolerance
            and tps_gap is not None
            and tps_gap > 0.0
        ):
            block.extend(
                [
                    "Recommendation:",
                    "- This is the intended AMHT outcome: a clear win on the state-sensitive task while retrieval stays competitive.",
                    "- Freeze router and memory knobs, validate with extra seeds, and only then move to `16K/32K` continuation.",
                    "",
                ]
            )
        elif router_score_collapsed and acc_gap is not None and acc_gap >= 0.0 and tps_gap is not None and tps_gap > 0.0:
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
                        "- Quality is still short of target. Freeze the stable retry architecture and increase training budget next instead of opening another backbone axis.",
                        "- If the longer-budget run still leaves all models clustered near chance on state-tracking, redesign or simplify the benchmark before more architecture tuning.",
                    ]
                        if stage_label == "2" and stable_mixed_mode
                        else [
                            "- Quality is still short of target. Keep the round-four router and memory settings fixed and use the state-tracking benchmark as the next gate.",
                            "- Re-open only backbone capacity if AMHT still cannot recover quality without breaking the mixed-task tradeoff.",
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
        elif (
            state_gap is not None
            and state_gap < 0.03
            and acc_gap is not None
            and acc_gap >= -quality_tie_tolerance
        ):
            block.extend(
                [
                    "Recommendation:",
                    "- Retrieval is holding, but AMHT still lacks the intended state-tracking separation. Keep router and memory knobs frozen.",
                    "- Re-open only backbone capacity next: test `ssm_state_size` first, then `ssm_conv_kernel` if needed.",
                    "",
                ]
            )
        elif tps_gap is not None and tps_gap < 0.0 and (acc_gap is None or acc_gap <= quality_tie_tolerance):
            block.extend(
                [
                    "Recommendation:",
                    "- Throughput is losing without a clear quality win. Do not reopen router-neighbor tuning; it does not reduce top-k compute in the current implementation.",
                    "- Keep the round-four/router settings fixed and either stop here or test a backbone-neutral efficiency change that preserves the routed budget.",
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
                    "- Mixed result. Keep the backbone fixed and use the state-tracking benchmark as the next decision gate.",
                    "- Do not retune router-neighbor, margin, or latent-token settings before the recurrent benchmark clearly separates.",
                    "",
                ]
            )
        return block

    if transformer is not None:
        lines.extend(compare_block(transformer, "Transformer"))
    if mamba_ref is not None:
        lines.extend(compare_block(mamba_ref, "Mamba-3-Inspired Hybrid"))

    quality_deficit = any(acc_gap is not None and acc_gap < -quality_tie_tolerance for acc_gap, _ in comparison_gaps)
    throughput_deficit = any(
        tps_gap is not None and tps_gap < 0.0 and (acc_gap is None or acc_gap <= quality_tie_tolerance)
        for acc_gap, tps_gap in comparison_gaps
    )
    favorable_tradeoff = bool(comparison_gaps) and all(
        acc_gap is not None and acc_gap >= -quality_tie_tolerance and tps_gap is not None and tps_gap > 0.0
        for acc_gap, tps_gap in comparison_gaps
    )

    def zh_comparison_line(target_key: str | None, target_title: str) -> list[str]:
        if target_key is None or target_key not in models:
            return []
        target_acc = metric(summary, target_key, "niah", "mean_accuracy")
        target_state_acc = metric(summary, target_key, "state_tracking", "mean_accuracy")
        target_tps = metric(summary, target_key, "throughput", "tokens_per_second")
        acc_gap = None if best_acc is None or target_acc is None else best_acc - target_acc
        state_gap = None if best_state_acc is None or target_state_acc is None else best_state_acc - target_state_acc
        tps_gap = None if best_tps is None or target_tps is None else best_tps - target_tps
        return [
            f"- 相对 {target_title} baseline：平均 NIAH 差距 `{fmt(acc_gap)}`，state-tracking 差距 `{fmt(state_gap)}`，吞吐差 `{fmt(tps_gap, 2)}` tok/s。"
        ]

    if content_retrieval_mode:
        goal_line = (
            "Beat the baselines on leakage-free retrieval quality while keeping the AMHT design sparse, memory-efficient, and near `router_ratio ~ 0.1`."
        )
        if quality_deficit:
            status_line = (
                "AMHT satisfies the sparse-efficiency constraints but misses the corrected retrieval quality target against the baselines."
            )
            design_lines = [
                "- Transformer-derived content axis: improve explicit lookup first with `block_size`, `latent_tokens`, `router_score_margin`, `router_score_weight`, and `router_feature_sources`.",
                "- Mamba-derived recurrent axis: only after retrieval is competitive, test `ssm_state_size` first and then `ssm_conv_kernel`, while keeping `ssm_complex: true`.",
                "- AMHT rule: do not mix content-path and recurrent-path changes in the same round; keep one path frozen so the result stays interpretable.",
            ]
        elif favorable_tradeoff:
            status_line = (
                "AMHT is aligned with the corrected retrieval goal: competitive-or-better quality with a throughput edge under sparse routing."
            )
            design_lines = [
                "- Transformer-derived content axis is good enough for now; freeze it and validate across seeds before reopening retrieval-path tuning.",
                "- Mamba-derived recurrent axis stays secondary until validation says there is still an unexplained state or continuation gap.",
                "- AMHT rule: spend the throughput headroom only after the quality win is reproducible.",
            ]
        else:
            status_line = (
                "AMHT is partially aligned: throughput is strong, but corrected retrieval quality is only competitive or still statistically weak."
            )
            design_lines = [
                "- Transformer-derived content axis comes first because corrected retrieval is the decision gate in this round.",
                "- Mamba-derived recurrent axis should stay frozen until the corrected retrieval result is validated and clearly informative.",
                "- AMHT rule: validate before opening a new axis, then tune only the losing path.",
            ]
    elif diagnostic_mode or state_memory_mode:
        goal_line = (
            "Show a real state-sensitive gain from the recurrent-plus-memory design without breaking sparse routing or drifting toward dense attention."
        )
        if favorable_tradeoff:
            status_line = (
                "AMHT is directionally aligned: the diagnostic is favorable and the hybrid path is earning its complexity."
            )
        else:
            status_line = (
                "AMHT is not yet clearly aligned: the diagnostic still needs to separate the recurrent-plus-memory path from the baselines."
            )
        design_lines = [
            "- Mamba-derived recurrent axis comes first here: use `ssm_state_size`, then `ssm_conv_kernel`, while keeping `ssm_complex: true`.",
            "- Transformer-derived content mixing stays a guardrail only; do not widen attention to compensate for a weak recurrent signal.",
            "- AMHT rule: reopen router or latent-memory tuning only after the recurrent diagnostic is informative.",
        ]
    else:
        goal_line = (
            "Beat or match the baselines on the active quality target while preserving sparse routed attention, memory efficiency, and `router_ratio ~ 0.1`."
        )
        if favorable_tradeoff:
            status_line = "AMHT is currently aligned with the goal on the measured tradeoff."
        elif quality_deficit and not throughput_deficit:
            status_line = "AMHT has throughput headroom but is still short on quality."
        elif throughput_deficit and not quality_deficit:
            status_line = "AMHT is spending compute without a clear quality return."
        else:
            status_line = "AMHT is in a mixed state: the benchmark is not yet giving a clean architecture decision."
        design_lines = [
            "- Use Transformer as the reference for explicit content lookup and Mamba-3-inspired hybrid as the reference for recurrent efficiency.",
            "- If retrieval or lookup quality is short, tune the AMHT content path first with `block_size`, `latent_tokens`, `router_score_margin`, `router_score_weight`, and `router_feature_sources`.",
            "- If state-sensitive behavior is short, tune the AMHT recurrent path first with `ssm_state_size`, `ssm_conv_kernel`, and `ssm_complex`.",
        ]

    lines.extend(
        [
            "## Goal Alignment",
            "",
            f"- Primary goal: {goal_line}",
            f"- Current status: {status_line}",
            "- Design constraint: keep the model meaningfully AMHT. Do not widen into dense full-sequence attention just to imitate the baselines.",
            "",
            "## Harness Engineering Research Circle",
            "",
            "- Keep AMHT, Transformer, and Mamba-3-inspired baselines in every decision round on the same benchmark and budget.",
            "- Treat Transformer as the content-lookup reference and Mamba-3-inspired hybrid as the recurrent-efficiency reference.",
            "- Every AMHT experiment should name one borrowed advantage and one protected constraint before the run starts.",
            "- Reject changes that break `router_selected_ratio ~ 0.1`, materially raise dense attention cost, or mix multiple architecture axes in one unexplained jump.",
            "",
            "## Baseline-Informed Design Axes",
            "",
            *design_lines,
            "",
        ]
    )

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
    if content_path_retry_validated:
        lines.extend(
            [
                "1. Freeze `stage2_round19_content_path_long_stability_retry` as the validated AMHT reference for corrected retrieval.",
                "2. Keep the content path fixed and only open a Mamba-inspired recurrent sweep if an unexplained state or continuation gap still matters.",
                "3. If you open that recurrent sweep, test `ssm_state_size` first and then `ssm_conv_kernel`, while keeping sparse routing and dense-attention cost fixed.",
                "4. If the recurrent sweep still cannot create a meaningful state-sensitive edge, keep the retrieval claim narrow and revisit the paper framing.",
                "",
            ]
        )
    elif content_path_retry_amht_reproduced_only:
        missing_labels = ", ".join(incomplete_retry_baselines) if incomplete_retry_baselines else "baseline models"
        lines.extend(
            [
                f"1. Complete the missing `{missing_labels}` evals for `stage2_round19_content_path_long_stability_retry_validate` so AMHT and both baselines cover the same seed set.",
                "2. Rebuild the report; only if the full comparison still holds should you freeze `stage2_round19_content_path_long_stability_retry` as the validated AMHT reference.",
                "3. Keep the content path fixed until that full baseline comparison is complete.",
                "4. Only after the full comparison is complete should you consider a Mamba-inspired recurrent sweep with `ssm_state_size` and then `ssm_conv_kernel`.",
                "5. If the completed comparison weakens the quality edge, keep the retrieval claim narrow or drop the retrieval-specific AMHT advantage claim.",
                "",
            ]
        )
    elif content_path_retry_mode:
        lines.extend(
            [
                "1. Run `stage2_round19_content_path_long_stability_retry_validate` next before making any new architectural change.",
                "2. If multi-seed retry validation holds, freeze `stage2_round19_content_path_long_stability_retry` as the validated AMHT reference.",
                "3. Only after that validation is clean should you open a Mamba-inspired recurrent sweep with `ssm_state_size` and then `ssm_conv_kernel`.",
                "4. If retry validation does not reproduce, keep the SSM path fixed and treat this as a reproducibility or stability problem, not a new architecture opportunity.",
                "5. If the validated quality edge disappears, keep the claim narrow or drop the retrieval-specific AMHT advantage claim.",
                "",
            ]
        )
    elif content_path_mode:
        lines.extend(
            [
                "1. Run `stage2_round19_content_path_validate` first as the same-budget `800-step` reproducibility check before making any new architectural change.",
                "2. If `800-step` parity holds, run `stage2_round19_content_path_long_stability` next to test whether the same configuration stays stable at `1600 steps`.",
                "3. If long-stability fails, treat it as an optimization-stability problem and run `stage2_round19_content_path_long_stability_retry` before reopening any architecture axis.",
                "4. If AMHT is still behind on corrected retrieval at `800 steps`, keep the SSM path fixed and continue only the content-path sweep: `block_size`, `latent_tokens`, `router_score_margin`, `router_score_weight`, and `router_feature_sources`.",
                "5. Only after corrected retrieval reaches parity and long-stability is clean should you open a Mamba-inspired recurrent sweep with `ssm_state_size` and then `ssm_conv_kernel`.",
                "6. If the content-path sweep still loses on leakage-free retrieval, stop making a retrieval-specific AMHT claim and revisit the paper framing.",
                "",
            ]
        )
    elif content_retrieval_mode:
        lines.extend(
            [
                "1. Run `stage2_round18_content_retrieval_validate` before making any new architectural change.",
                "2. If the corrected retrieval gap still favors the baselines, run a Transformer-inspired AMHT content-path sweep first: `block_size`, `latent_tokens`, `router_score_margin`, `router_score_weight`, and `router_feature_sources`.",
                "3. Only after retrieval is competitive, run a Mamba-inspired recurrent sweep: `ssm_state_size` first, then `ssm_conv_kernel`, while keeping `ssm_complex: true`.",
                "4. Use corrected retrieval as the main decision gate; if AMHT still cannot beat the baselines there, stop claiming a retrieval-specific AMHT advantage and revisit the paper framing.",
                "",
            ]
        )
    elif diagnostic_mode:
        lines.extend(
            (
                [
                    "1. Run `stage2_round17_state_memory_diag_validate` before any mixed-task retry.",
                    "2. If AMHT shows a clear edge there, add an AMHT memory-off ablation to verify that latent memory causes the gain.",
                    "3. Reintroduce retrieval only after the pure memory diagnostic is favorable.",
                    "4. If AMHT still cannot beat the baselines here, stop memory-axis churn and revisit the benchmark or paper framing.",
                    "",
                ]
                if state_memory_diag_mode
                else [
                    "1. Retry mixed `stage2_round7` now that pure state-tracking is numerically stable.",
                    "2. If mixed training is stable, compare whether retrieval-aligned training lifts state-tracking above chance.",
                    "3. If all three models remain near chance, increase training budget or simplify the benchmark before reading architecture conclusions from it.",
                    "4. Re-open backbone capacity only after the task is stable and informative.",
                    "",
                ]
            )
        )
    elif validated_stage2_mode:
        lines.extend(
            [
                "1. Freeze `stage2_round13` as the validated AMHT reference.",
                "2. Stop reading the current `NIAH` setting as the main optimization target; it is saturated across the baselines.",
                "3. Move next to harder retrieval or longer-context evaluation with the same architecture.",
                "4. If AMHT still does not show a clear state-sensitive edge there, stop architecture churn and revisit the benchmark or paper framing.",
                "",
            ]
        )
    elif validation_mode:
        lines.extend(
            [
                "1. Run `stage2_round13_validate` to verify the long-budget result across seeds.",
                "2. Keep the `stage2_round13` architecture frozen while validating; do not reopen backbone, router, or memory tuning yet.",
                "3. If the state-tracking pattern does not hold across seeds, stop architecture churn and revisit the training recipe or benchmark.",
                "4. If the result is reproducible, then decide between efficiency work and longer-context validation.",
                "",
            ]
        )
    elif post_budget_mode:
        lines.extend(
            [
                "1. Keep the stable retry architecture fixed as the last reliable mixed-training setup.",
                "2. Stop opening new backbone, router, or memory axes on the current `modsum` benchmark.",
                "3. Redesign or simplify the state-tracking benchmark so it separates the models above chance at the same budget.",
                "4. Resume architecture tuning only after the benchmark is informative.",
                "",
            ]
        )
    elif state_memory_mode and quality_deficit:
        lines.extend(
            [
                "1. Keep the memory-on-state policy fixed; do not revert to the old state-batch path where latent memory is disabled.",
                "2. Adjust the retrieval/state mixture before adding more steps or opening a new backbone axis.",
                "3. Treat retrieval as a guardrail rather than the main target until a state-sensitive advantage appears.",
                "4. Re-open router changes only if the state-memory pivot still fails while retrieval remains competitive.",
                "",
            ]
        )
    elif state_memory_mode:
        lines.extend(
            [
                "1. Validate the memory-on-state pivot with extra seeds before making new architectural changes.",
                "2. Keep retrieval in the loop as a guardrail, but read state-tracking as the main decision gate.",
                "3. Tune the retrieval/state mixture before touching backbone capacity.",
                "4. Re-open router changes only if the state-memory pivot plateaus with acceptable retrieval quality.",
                "",
            ]
        )
    elif quality_deficit:
        lines.extend(
            [
                *(
                    [
                        "1. Keep the stable retry mix and freeze router, memory, and backbone settings.",
                        "2. Increase training budget only with the same architecture before opening new axes.",
                        "3. If all models still cluster near chance on state-tracking, redesign or simplify the benchmark before more architecture work.",
                        "4. Re-open backbone tuning only after the longer-budget run proves the benchmark is informative.",
                    ]
                    if stage_label == "2" and stable_mixed_mode
                    else [
                        "1. Run a pure state-tracking diagnostic before any new mixed-task retry.",
                        "2. Keep the round-four router and memory settings fixed; do not reopen latent or router-neighbor tuning.",
                        "3. Re-open only backbone capacity after state-tracking is numerically stable.",
                        "4. Retry mixed retrieval plus state-tracking smoke only after the diagnostic run is clean.",
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
                "1. Keep the round-four router and memory settings frozen.",
                "2. Re-run the same comparison with the new state-tracking benchmark enabled.",
                "3. Change the backbone only if AMHT still lacks a clear state-sensitive advantage.",
                "4. Delay efficiency tuning until the recurrent-thesis benchmark is proven.",
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
                        "1. Keep the round-four router and memory settings fixed.",
                        "2. Use mixed retrieval plus state-tracking training as the next gate.",
                        "3. Re-open only backbone capacity if AMHT fails to clear a state-tracking win.",
                        "4. Move to longer-context freezing only after state-tracking and retrieval both hold.",
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

    if content_retrieval_mode:
        if content_path_retry_validated:
            zh_round_status = (
                "当前 round19 stability-retry 已经完成跨 seed 验证。AMHT 在 `1600-step` 长预算下不再只是单次跑通，而是已经成为可复现的 corrected retrieval 参考结果。"
            )
            zh_goal_status = (
                "AMHT 已经从“长程训练会崩”推进到“长预算结果可复现”，因此下一步可以在冻结 content path 的前提下，谨慎评估是否还需要打开 recurrent 方向。"
            )
        elif content_path_retry_amht_reproduced_only:
            zh_round_status = (
                "当前 round19 stability-retry 已经证明 AMHT 自身的 `1600-step` 长预算稳定性可以跨 seed 复现，但 baseline eval 仍未补齐，因此这还不是完整的对比验证。"
            )
            zh_goal_status = (
                "AMHT 已经从“长程训练会崩”推进到“AMHT 自身可稳定复现”，但在 baseline 对比未补齐前，还不能把这一轮直接冻结为完整验证通过。"
            )
        elif content_path_retry_mode:
            zh_round_status = (
                "当前 round19 stability-retry 已经完成 `1600-step` 单 seed 稳定跑通。AMHT 已从“long-budget 会崩”推进到“long-budget 单次可稳定”，但还没有完成跨 seed 复现。"
            )
            zh_goal_status = (
                "AMHT 当前已经基本恢复到与 corrected retrieval 目标一致的状态，但证据仍停留在单 seed 长预算成功，不能直接当作最终定论。"
            )
        elif quality_deficit:
            zh_round_status = (
                "当前 corrected retrieval 轮次已经修复了旧 benchmark 的泄漏问题，但 AMHT 仍未稳定超过基线。现阶段证据只支持“稀疏效率优势”，不支持“retrieval 质量优势”结论。"
            )
            zh_goal_status = (
                "AMHT 仍符合稀疏、内存高效、近 `router_ratio ~ 0.1` 的设计约束，但还没有完成 leakage-free retrieval 的主目标。"
            )
        elif favorable_tradeoff:
            zh_round_status = (
                "当前 corrected retrieval 轮次已经达到可继续推进的状态。AMHT 在无泄漏 retrieval benchmark 上至少与基线持平，同时保留了明显吞吐优势。"
            )
            zh_goal_status = (
                "AMHT 与当前目标一致：在保持稀疏路由和内存效率约束的同时，retrieval 质量已经达到 competitive 或 parity 水平。"
            )
        else:
            zh_round_status = (
                "当前 corrected retrieval 轮次已经变得有意义，但证据仍然偏弱。AMHT 展现出吞吐优势，不过 retrieval 质量仍需要验证才能下结论。"
            )
            zh_goal_status = (
                "AMHT 目前只部分满足目标：效率优势明确，但 corrected retrieval 上的质量优势还不够稳定。"
            )

        zh_intro_title = "### Round19 结论" if content_path_mode else "### 修正检索轮次结论"
        zh_next_steps = (
            [
                "1. 先把 `stage2_round19_content_path_long_stability_retry` 冻结为 corrected retrieval 的已验证 AMHT 参考配置。",
                "2. 保持 content path 不变，只有在 state 或 continuation 方向仍有未解释差距时，才开启 Mamba-inspired recurrent sweep。",
                "3. 如果开启 recurrent sweep，优先顺序是 `ssm_state_size`，然后才是 `ssm_conv_kernel`，同时保持稀疏路由和 dense attention 成本不变。",
                "4. 如果 recurrent sweep 仍不能带来有意义的 state-sensitive 优势，就把 retrieval claim 保持在更窄的范围内，并回到更保守的 paper framing。",
            ]
            if content_path_retry_validated
            else [
                f"1. 先补齐 `stage2_round19_content_path_long_stability_retry_validate` 中缺失的 `{', '.join(incomplete_retry_baselines) if incomplete_retry_baselines else 'baseline'}` eval，确保 AMHT 和基线使用相同的 seed 集合。",
                "2. 重新生成 report；只有在完整对比后优势仍成立，才把 `stage2_round19_content_path_long_stability_retry` 冻结为已验证 AMHT 参考配置。",
                "3. 在这个完整 baseline 对比补齐之前，保持 content path 不变，不要提前开启 recurrent sweep。",
                "4. 只有在完整对比补齐之后，才考虑 Mamba-inspired recurrent sweep，优先顺序是 `ssm_state_size`，然后才是 `ssm_conv_kernel`。",
                "5. 如果补齐后的完整对比削弱了优势，就把 retrieval-specific AMHT claim 收窄，必要时直接放弃该 claim。",
            ]
            if content_path_retry_amht_reproduced_only
            else [
                "1. 先运行 `stage2_round19_content_path_long_stability_retry_validate`，在不改架构的前提下确认 retry 的 `1600-step` 结果是否能跨 seed 复现。",
                "2. 如果 multi-seed retry validation 成立，就把 `stage2_round19_content_path_long_stability_retry` 冻结为已验证的 AMHT 参考配置。",
                "3. 只有在这个验证通过后，才开启 Mamba-inspired recurrent sweep，优先顺序是 `ssm_state_size`，然后才是 `ssm_conv_kernel`。",
                "4. 如果 retry validation 不能复现，就把问题继续归类为稳定性或可复现性问题，不要立刻重开新的架构轴。",
                "5. 如果验证后质量优势消失，就把 retrieval-specific AMHT claim 收窄，必要时直接放弃该 claim。",
            ]
            if content_path_retry_mode
            else [
                "1. 先运行 `stage2_round19_content_path_validate`，把它作为同预算的 `800-step` 复现验证，在不改架构的前提下确认 parity 是否可复现。",
                "2. 如果 `800-step` parity 成立，再运行 `stage2_round19_content_path_long_stability`，单独检查同一配置在 `1600 steps` 下是否仍然稳定。",
                "3. 如果 long-stability 失败，把问题归类为训练稳定性问题，并先运行 `stage2_round19_content_path_long_stability_retry`，不要立刻重开新的架构轴。",
                "4. 如果 corrected retrieval 在 `800 steps` 下仍落后，只继续做 content-path 单轴搜索：`block_size`、`latent_tokens`、`router_score_margin`、`router_score_weight`、`router_feature_sources`。",
                "5. 只有在 retrieval 达到 parity 且 long-stability 也通过后，才开启 Mamba-inspired recurrent sweep，优先顺序是 `ssm_state_size`，然后才是 `ssm_conv_kernel`。",
                "6. 如果 content-path 方向最终仍无法稳定赢过 baseline，就停止 retrieval-specific AMHT claim，回到更保守的 paper framing。",
            ]
            if content_path_mode
            else [
                "1. 先运行 `stage2_round18_content_retrieval_validate`，在不改架构的前提下确认 corrected retrieval 结果是否可复现。",
                "2. 如果 corrected retrieval 仍落后，只继续做 Transformer-inspired content-path 单轴搜索：`block_size`、`latent_tokens`、`router_score_margin`、`router_score_weight`、`router_feature_sources`。",
                "3. 只有在 retrieval 达到 competitive 之后，才开启 Mamba-inspired recurrent sweep，优先顺序是 `ssm_state_size`，然后才是 `ssm_conv_kernel`。",
                "4. 如果 corrected retrieval 仍不能超过 baseline，就停止 retrieval-specific AMHT claim，并回到更保守的论文表述。",
            ]
        )

        lines.extend(
            [
                "## 中文结论（自动生成）",
                "",
                zh_intro_title,
                "",
                zh_round_status,
                "状态跟踪结果仍未提供更强记忆能力的区分证据，因此当前轮次主要支持“retrieval 质量与吞吐折中”的结论，而不支持更广义的 latent-memory 或通用 long-context memory 声明。",
                "",
                *zh_comparison_line(transformer, "Transformer"),
                *zh_comparison_line(mamba_ref, "Mamba-3-inspired hybrid"),
                "",
                "### 目标对齐",
                "",
                "- 主目标：在 leakage-free retrieval 上超过或至少追平基线，同时保持 AMHT 的稀疏、内存高效、近 `router_ratio ~ 0.1` 的设计约束。",
                f"- 当前状态：{zh_goal_status}",
                "- 设计约束：保持模型“有意义地属于 AMHT”，不要为了追基线而退化成 dense full-sequence attention。",
                "",
                "### 下一步顺序",
                "",
                *zh_next_steps,
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
