import unittest

from scripts.run_colab_paper import deep_merge_dict


class RunColabPaperTests(unittest.TestCase):
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
