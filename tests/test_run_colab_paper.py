import sys
import unittest

from scripts.run_colab_paper import MODEL_SPECS, PRESETS, build_summary, deep_merge_dict, run_command


class RunColabPaperTests(unittest.TestCase):
    def test_round19_content_path_presets_and_models_are_registered(self) -> None:
        self.assertIn("amht_v4_stage2_round19_content_path_long_stability_retry", MODEL_SPECS)
        self.assertIn("amht_v4_stage2_round19_content_path", MODEL_SPECS)
        self.assertIn("stage2_round19_content_path", PRESETS)
        self.assertIn("stage2_round19_content_path_t4", PRESETS)
        self.assertIn("stage2_round19_content_path_validate", PRESETS)
        self.assertIn("stage2_round19_content_path_long_stability", PRESETS)
        self.assertIn("stage2_round19_content_path_long_stability_validate", PRESETS)
        self.assertIn("stage2_round19_content_path_long_stability_retry", PRESETS)
        self.assertIn("stage2_round19_content_path_long_stability_retry_validate", PRESETS)
        self.assertIn("stage2_round19_content_path_t4_long_stability", PRESETS)
        self.assertIn("stage2_round19_content_path_t4_long_stability_validate", PRESETS)
        self.assertEqual(PRESETS["stage2_round19_content_path"]["steps_scale"], 4.0)
        self.assertEqual(PRESETS["stage2_round19_content_path_validate"]["steps_scale"], 4.0)
        self.assertEqual(PRESETS["stage2_round19_content_path_long_stability"]["steps_scale"], 8.0)
        self.assertEqual(PRESETS["stage2_round19_content_path_long_stability_validate"]["steps_scale"], 8.0)
        self.assertEqual(PRESETS["stage2_round19_content_path_long_stability_retry"]["steps_scale"], 8.0)
        self.assertEqual(PRESETS["stage2_round19_content_path_long_stability_retry_validate"]["steps_scale"], 8.0)

    def test_round18_content_retrieval_presets_and_models_are_registered(self) -> None:
        self.assertIn("amht_v4_stage2_round18_content_retrieval", MODEL_SPECS)
        self.assertIn("transformer_v4_stage2_round18_content_retrieval_baseline", MODEL_SPECS)
        self.assertIn("mamba3_hybrid_v4_stage2_round18_content_retrieval_baseline", MODEL_SPECS)
        self.assertIn("stage2_round18_content_retrieval", PRESETS)
        self.assertIn("stage2_round18_content_retrieval_t4", PRESETS)

    def test_round17_state_memory_diag_presets_and_models_are_registered(self) -> None:
        self.assertIn("amht_v4_stage2_round17_state_memory_diag", MODEL_SPECS)
        self.assertIn("transformer_v4_stage2_round17_state_memory_diag_baseline", MODEL_SPECS)
        self.assertIn("mamba3_hybrid_v4_stage2_round17_state_memory_diag_baseline", MODEL_SPECS)
        self.assertIn("stage2_round17_state_memory_diag", PRESETS)
        self.assertIn("stage2_round17_state_memory_diag_t4", PRESETS)

    def test_round16_presets_and_models_are_registered(self) -> None:
        self.assertIn("amht_v4_stage2_round16", MODEL_SPECS)
        self.assertIn("transformer_v4_stage2_round16_baseline", MODEL_SPECS)
        self.assertIn("mamba3_hybrid_v4_stage2_round16_baseline", MODEL_SPECS)
        self.assertIn("stage2_round16", PRESETS)
        self.assertIn("stage2_round16_t4", PRESETS)

    def test_deep_merge_dict_overrides_nested_eval_values_only(self) -> None:
        base = {
            "training": {"steps": 200},
            "evaluation": {
                "benchmark_steps": 2,
                "niah": {"batch_size": 2, "seq_len": 32768},
            },
        }
        overrides = {
            "evaluation": {
                "benchmark_steps": 1,
                "niah": {"batch_size": 1},
            },
        }

        merged = deep_merge_dict(base, overrides)

        self.assertEqual(merged["training"]["steps"], 200)
        self.assertEqual(merged["evaluation"]["benchmark_steps"], 1)
        self.assertEqual(merged["evaluation"]["niah"]["batch_size"], 1)
        self.assertEqual(merged["evaluation"]["niah"]["seq_len"], 32768)

    def test_run_command_returns_false_when_continue_on_error_is_enabled(self) -> None:
        ok = run_command(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            continue_on_error=True,
        )
        self.assertFalse(ok)

    def test_build_summary_records_completed_runs_per_model(self) -> None:
        model_key = "amht_v4_stage2_round19_content_path_long_stability_retry"
        runs_by_model = {
            model_key: [
                {
                    "seed": 42,
                    "train_log": [],
                    "train_final": {
                        "total_loss": 1.0,
                        "main_loss": 1.0,
                        "router_loss": 0.0,
                        "router_mean": 0.1,
                        "router_selected_ratio": 0.1016,
                        "router_selected_score_mean": 0.2,
                        "router_unselected_score_mean": 0.1,
                        "router_score_gap": 0.1,
                        "tokens_per_second": 1000.0,
                    },
                    "eval": {
                        "throughput": {
                            "tokens_per_second": 2000.0,
                            "milliseconds_per_step": 10.0,
                            "seq_len": 16384,
                        },
                        "niah": {
                            "mean_accuracy": 0.1,
                            "batch_size": 1,
                            "repeats": 1,
                            "accuracy_by_depth": [0.0],
                            "needle_depths": [0.5],
                            "seq_len": 32768,
                        },
                        "state_tracking": {
                            "mean_accuracy": 0.0,
                            "results": [],
                            "seq_lens": [],
                        },
                    },
                    "run_dir": "/tmp/fake",
                }
            ]
        }

        summary = build_summary(
            runs_by_model,
            [model_key],
            seed_count=3,
            eval_task="all",
            warmup_steps=1,
            benchmark_steps=2,
        )

        self.assertEqual(summary["models"][model_key]["completed_runs"], 1)


if __name__ == "__main__":
    unittest.main()
