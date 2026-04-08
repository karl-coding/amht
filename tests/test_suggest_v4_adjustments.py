from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "suggest_v4_adjustments.py"
SPEC = importlib.util.spec_from_file_location("suggest_v4_adjustments", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load module from {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class SuggestV4AdjustmentsTests(unittest.TestCase):
    def test_picks_stage2_diag_amht_and_labels_stage2(self) -> None:
        summary = {
            "models": {
                "amht_v4_stage2_round7_state_tracking_diag": {
                    "label": "AMHT-V4-Stage2-R7-Diag",
                    "state_tracking": {"mean_accuracy": {"mean": 0.0625, "std": 0.0}},
                },
                "transformer_v4_stage2_round7_state_tracking_diag_baseline": {
                    "label": "Transformer-V4-Stage2-R7-Diag",
                    "state_tracking": {"mean_accuracy": {"mean": 0.05625, "std": 0.0}},
                },
                "mamba3_hybrid_v4_stage2_round7_state_tracking_diag_baseline": {
                    "label": "Mamba-V4-Stage2-R7-Diag",
                    "state_tracking": {"mean_accuracy": {"mean": 0.05, "std": 0.0}},
                },
            }
        }

        best = MODULE.pick_best_amht(summary)
        note = MODULE.build_note(summary)

        self.assertEqual(best, "amht_v4_stage2_round7_state_tracking_diag")
        self.assertIn("# Stage 2 Diagnostic Adjustment Note", note)
        self.assertNotIn("No AMHT model found in summary.", note)
        self.assertIn("Retry mixed `stage2_round7` now that pure state-tracking is numerically stable.", note)

    def test_stable_mixed_stage2_quality_deficit_points_to_backbone_only_followup(self) -> None:
        summary = {
            "models": {
                "amht_v4_stage2_round8": {
                    "label": "AMHT-V4-Stage2-R8",
                    "niah": {"mean_accuracy": {"mean": 0.4732, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.075, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 251377.94, "std": 0.0}},
                },
                "transformer_v4_stage2_round7_retry_baseline": {
                    "label": "Transformer",
                    "niah": {"mean_accuracy": {"mean": 0.5446, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.05, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 334267.74, "std": 0.0}},
                },
                "mamba3_hybrid_v4_stage2_round7_retry_baseline": {
                    "label": "Mamba",
                    "niah": {"mean_accuracy": {"mean": 0.5, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.075, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 241548.13, "std": 0.0}},
                },
            }
        }

        note = MODULE.build_note(summary)

        self.assertIn("# Stage 2 Adjustment Note", note)
        self.assertIn("Freeze the stable retry architecture and increase training budget next instead of opening another backbone axis.", note)
        self.assertIn("Increase training budget only with the same architecture before opening new axes.", note)
        self.assertNotIn("Run a pure state-tracking diagnostic before any new mixed-task retry.", note)

    def test_post_budget_stage2_points_to_benchmark_redesign(self) -> None:
        summary = {
            "models": {
                "amht_v4_stage2_round10": {
                    "label": "AMHT-V4-Stage2-R10",
                    "niah": {"mean_accuracy": {"mean": 0.7321, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.0625, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 302254.56, "std": 0.0}},
                },
                "transformer_v4_stage2_round10_baseline": {
                    "label": "Transformer",
                    "niah": {"mean_accuracy": {"mean": 0.7946, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.05, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 334244.38, "std": 0.0}},
                },
                "mamba3_hybrid_v4_stage2_round10_baseline": {
                    "label": "Mamba",
                    "niah": {"mean_accuracy": {"mean": 0.7411, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.0375, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 241537.31, "std": 0.0}},
                },
            }
        }

        note = MODULE.build_note(summary)

        self.assertIn("This longer-budget run is the clear readout: retrieval improved, but the state-tracking benchmark still leaves all models clustered near chance.", note)
        self.assertIn("Redesign or simplify the state-tracking benchmark so it separates the models above chance at the same budget.", note)
        self.assertNotIn("Re-open only backbone capacity next", note)

    def test_round13_note_includes_baseline_comparisons_and_validation_followup(self) -> None:
        summary = {
            "models": {
                "amht_v4_stage2_round13": {
                    "label": "AMHT-V4-Stage2-R13",
                    "niah": {"mean_accuracy": {"mean": 0.9911, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.5250, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 296610.64, "std": 0.0}},
                },
                "transformer_v4_stage2_round13_baseline": {
                    "label": "Transformer",
                    "niah": {"mean_accuracy": {"mean": 0.9850, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.5125, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 320271.74, "std": 0.0}},
                },
                "mamba3_hybrid_v4_stage2_round13_baseline": {
                    "label": "Mamba",
                    "niah": {"mean_accuracy": {"mean": 0.9700, "std": 0.0}},
                    "state_tracking": {"mean_accuracy": {"mean": 0.5375, "std": 0.0}},
                    "throughput": {"tokens_per_second": {"mean": 235149.04, "std": 0.0}},
                },
            }
        }

        note = MODULE.build_note(summary)

        self.assertIn("## AMHT vs Transformer", note)
        self.assertIn("## AMHT vs Mamba-3-Inspired Hybrid", note)
        self.assertIn("Run `stage2_round13_validate` next", note)
        self.assertIn("Run `stage2_round13_validate` to verify the long-budget result across seeds.", note)


if __name__ == "__main__":
    unittest.main()
