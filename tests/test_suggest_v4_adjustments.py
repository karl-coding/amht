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


if __name__ == "__main__":
    unittest.main()
