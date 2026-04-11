from __future__ import annotations

import random
import unittest

import torch

from data.dataset import build_retrieval_batch
from eval.niah import build_niah_batch


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_default_retrieval_mapping_still_matches_query_offset(self) -> None:
        tokens, expected = build_retrieval_batch(
            batch_size=8,
            vocab_size=4096,
            seq_len=128,
            pad_token=0,
            key_start=100,
            value_start=1000,
            num_pairs=4,
            num_keys=16,
            target_depth=0.5,
            rng=random.Random(7),
        )

        direct_answer = 1000 + (tokens[:, -2] - 100)
        self.assertTrue(torch.equal(expected.cpu(), direct_answer.cpu()))
        self.assertTrue(torch.equal(tokens[:, -1].cpu(), direct_answer.cpu()))

    def test_random_value_mapping_breaks_direct_query_offset_rule(self) -> None:
        tokens, expected = build_retrieval_batch(
            batch_size=16,
            vocab_size=4096,
            seq_len=128,
            pad_token=0,
            key_start=100,
            value_start=1000,
            num_pairs=4,
            num_keys=16,
            value_pool_size=32,
            random_value_mapping=True,
            target_depth=0.5,
            rng=random.Random(7),
        )

        direct_answer = 1000 + (tokens[:, -2] - 100)
        self.assertTrue(bool((expected.cpu() != direct_answer.cpu()).any().item()))

    def test_niah_batch_can_use_random_value_mapping(self) -> None:
        random.seed(11)
        tokens, expected = build_niah_batch(
            batch_size=16,
            seq_len=256,
            vocab_size=4096,
            pad_token=0,
            key_start=100,
            value_start=1000,
            num_pairs=4,
            num_keys=16,
            value_pool_size=32,
            random_value_mapping=True,
            depth=0.8,
            device=torch.device("cpu"),
        )

        direct_answer = 1000 + (tokens[:, -2] - 100)
        self.assertTrue(bool((expected.cpu() != direct_answer.cpu()).any().item()))


if __name__ == "__main__":
    unittest.main()
