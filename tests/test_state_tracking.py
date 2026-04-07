from __future__ import annotations

import unittest

import torch

from data.dataset import MixedDataset, StateTrackingDataset
from model.amht import AMHTModel


class DummyDataset:
    def __init__(self, value: int, total_samples: int = 1024) -> None:
        self.value = value
        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor([self.value, index], dtype=torch.long)


class StateTrackingDatasetTests(unittest.TestCase):
    def test_modsum_target_matches_prefix_sum(self) -> None:
        dataset = StateTrackingDataset(
            vocab_size=64,
            seq_len=33,
            total_samples=4,
            modulus=16,
            seed=123,
        )

        sample = dataset[0]
        expected = int(sample[:-1].sum().item() % 16)
        self.assertEqual(int(sample[-1].item()), expected)

    def test_samples_are_deterministic_for_seed_and_index(self) -> None:
        left = StateTrackingDataset(vocab_size=64, seq_len=17, total_samples=8, modulus=16, seed=99)
        right = StateTrackingDataset(vocab_size=64, seq_len=17, total_samples=8, modulus=16, seed=99)
        other = StateTrackingDataset(vocab_size=64, seq_len=17, total_samples=8, modulus=16, seed=100)

        self.assertTrue(torch.equal(left[3], right[3]))
        self.assertFalse(torch.equal(left[3], other[3]))

    def test_flipflop_target_matches_last_write_before_query_gap(self) -> None:
        dataset = StateTrackingDataset(
            vocab_size=64,
            seq_len=10,
            total_samples=2,
            task="flipflop",
            num_slots=3,
            value_count=2,
            slot_start=0,
            value_start=8,
            query_start=16,
            min_query_gap_tokens=4,
            seed=7,
        )

        sample = dataset[0]
        query_slot = int(sample[-2].item()) - 16
        answer = int(sample[-1].item())
        updates = sample[:-2].view(-1, 2)

        decoded_slots = [int(pair[0].item()) for pair in updates]
        decoded_values = [int(pair[1].item()) for pair in updates]
        self.assertNotIn(query_slot, decoded_slots[-2:])

        last_match = max(index for index, slot in enumerate(decoded_slots) if slot == query_slot)
        expected = decoded_values[last_match]
        self.assertEqual(answer, expected)


class MixedDatasetTests(unittest.TestCase):
    def test_source_sampling_tracks_configured_weights(self) -> None:
        mixed = MixedDataset(
            datasets={
                "retrieval": DummyDataset(1),
                "state_tracking": DummyDataset(2),
            },
            weights={
                "retrieval": 0.75,
                "state_tracking": 0.25,
            },
            total_samples=2000,
            seed=42,
        )

        retrieval_count = sum(1 for index in range(len(mixed)) if mixed.sample_source(index) == "retrieval")
        retrieval_ratio = retrieval_count / len(mixed)

        self.assertGreater(retrieval_ratio, 0.70)
        self.assertLess(retrieval_ratio, 0.80)

    def test_returns_component_sample_from_selected_source(self) -> None:
        mixed = MixedDataset(
            datasets={
                "retrieval": DummyDataset(1),
                "state_tracking": DummyDataset(2),
            },
            weights={
                "retrieval": 1.0,
                "state_tracking": 0.0,
            },
            total_samples=8,
            seed=7,
        )

        sample = mixed[3]
        self.assertEqual(int(sample[0].item()), 1)
        self.assertEqual(int(sample[1].item()), 3)


class AMHTStateTrackingBypassTests(unittest.TestCase):
    def test_can_disable_router_and_memory_paths_for_state_tracking(self) -> None:
        cfg = {
            "model": {
                "architecture": "amht",
                "vocab_size": 64,
                "dim": 16,
                "hidden_dim": 32,
                "layers": 2,
                "heads": 2,
                "latent_tokens": 4,
                "memory_per_layer_io": True,
                "router_ratio": 0.1,
                "ssm_state_size": 8,
                "max_seq_len": 16,
                "ssm_impl": "surrogate",
            }
        }
        model = AMHTModel(cfg)
        tokens = torch.randint(0, 16, (2, 8), dtype=torch.long)

        logits, stats = model(
            tokens,
            router_straight_through_enabled=False,
            router_attention_enabled=False,
            memory_enabled=False,
        )

        self.assertEqual(tuple(logits.shape), (2, 8, 64))
        self.assertEqual(float(stats["router_mean"].item()), 0.0)
        self.assertEqual(float(stats["router_selected_ratio"].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
