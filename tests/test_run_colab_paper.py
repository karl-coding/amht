import unittest

from scripts.run_colab_paper import MODEL_SPECS, PRESETS, deep_merge_dict


class RunColabPaperTests(unittest.TestCase):
    def test_round19_content_path_presets_and_models_are_registered(self) -> None:
        self.assertIn("amht_v4_stage2_round19_content_path", MODEL_SPECS)
        self.assertIn("stage2_round19_content_path", PRESETS)
        self.assertIn("stage2_round19_content_path_t4", PRESETS)

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


if __name__ == "__main__":
    unittest.main()
