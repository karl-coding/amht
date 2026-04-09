import unittest

from train.train import should_skip_non_finite_batch


class TrainNonFiniteTests(unittest.TestCase):
    def test_default_budget_does_not_skip(self) -> None:
        self.assertFalse(should_skip_non_finite_batch({}, 0))

    def test_positive_budget_skips_until_exhausted(self) -> None:
        cfg = {"max_non_finite_batches": 2}
        self.assertTrue(should_skip_non_finite_batch(cfg, 0))
        self.assertTrue(should_skip_non_finite_batch(cfg, 1))
        self.assertFalse(should_skip_non_finite_batch(cfg, 2))


if __name__ == "__main__":
    unittest.main()
