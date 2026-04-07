from __future__ import annotations

from bisect import bisect_right
import random
from collections.abc import Mapping, Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class SyntheticDataset:
    """Synthetic token dataset used until a real corpus is wired in."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        total_samples: int = 1024,
        seed: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.seed is None:
            return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long, generator=generator)


class RetrievalDataset:
    """Key-value retrieval task aligned with harder NIAH-style training."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        total_samples: int = 1024,
        pad_token: int = 0,
        key_start: int = 100,
        value_start: int = 1000,
        num_pairs: int = 4,
        num_keys: int = 16,
        depth_choices: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples
        self.pad_token = pad_token
        self.key_start = key_start
        self.value_start = value_start
        self.num_pairs = num_pairs
        self.num_keys = max(num_keys, num_pairs)
        self.depth_choices = depth_choices or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if self.seed is None:
            rng = random
            generator = None
        else:
            rng = random.Random(self.seed + index)
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)

        randint_kwargs = {"dtype": torch.long}
        if generator is not None:
            randint_kwargs["generator"] = generator
        tokens = torch.randint(
            self.value_start + self.num_keys + 10,
            self.vocab_size,
            (self.seq_len,),
            **randint_kwargs,
        )
        tokens[0] = self.pad_token
        selected_keys = rng.sample(range(self.num_keys), self.num_pairs)
        target_idx = rng.randrange(self.num_pairs)
        target_depth = rng.choice(self.depth_choices)
        target_position = min(self.seq_len - 4, max(1, int(self.seq_len * target_depth)))

        occupied = {target_position, target_position + 1, self.seq_len - 2, self.seq_len - 1}
        available_positions = list(range(1, self.seq_len - 3))
        rng.shuffle(available_positions)

        pair_positions: list[int | None] = [None] * self.num_pairs
        pair_positions[target_idx] = target_position
        distractor_positions: list[int] = []
        for pos in available_positions:
            if len(distractor_positions) >= self.num_pairs - 1:
                break
            if pos in occupied or (pos + 1) in occupied:
                continue
            distractor_positions.append(pos)
            occupied.add(pos)
            occupied.add(pos + 1)

        distractor_iter = iter(sorted(distractor_positions))
        for pair_idx in range(self.num_pairs):
            if pair_positions[pair_idx] is None:
                pair_positions[pair_idx] = next(distractor_iter)

        for pair_idx, pos in enumerate(pair_positions):
            key_id = selected_keys[pair_idx]
            key_token = self.key_start + key_id
            value_token = self.value_start + key_id
            tokens[pos] = key_token
            tokens[pos + 1] = value_token

        target_key = selected_keys[target_idx]
        query_key = self.key_start + target_key
        answer_value = self.value_start + target_key
        tokens[-2] = query_key
        tokens[-1] = answer_value
        return tokens


class StateTrackingDataset:
    """State-sensitive final-token task used to probe recurrent tracking."""

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        total_samples: int = 1024,
        *,
        task: str = "modsum",
        modulus: int = 16,
        digit_start: int = 0,
        seed: int | None = None,
    ) -> None:
        if task != "modsum":
            raise ValueError(f"Unsupported state-tracking task: {task}")
        if seq_len < 2:
            raise ValueError("StateTrackingDataset requires seq_len >= 2")
        if modulus <= 0:
            raise ValueError("StateTrackingDataset requires modulus > 0")
        if digit_start < 0 or digit_start + modulus > vocab_size:
            raise ValueError("StateTrackingDataset digits must fit inside the configured vocabulary")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples
        self.task = task
        self.modulus = modulus
        self.digit_start = digit_start
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)
        digit_kwargs = {"dtype": torch.long}
        if generator is not None:
            digit_kwargs["generator"] = generator
        digits = torch.randint(
            self.digit_start,
            self.digit_start + self.modulus,
            (self.seq_len - 1,),
            **digit_kwargs,
        )
        target = ((digits - self.digit_start).sum() % self.modulus) + self.digit_start
        tokens = torch.empty((self.seq_len,), dtype=torch.long)
        tokens[:-1] = digits
        tokens[-1] = target
        return tokens


class MixedDataset:
    """Sample from multiple token datasets using fixed per-task weights."""

    def __init__(
        self,
        datasets: Mapping[str, Sequence[torch.Tensor]],
        weights: Mapping[str, float],
        *,
        total_samples: int | None = None,
        seed: int | None = None,
    ) -> None:
        if not datasets:
            raise ValueError("MixedDataset requires at least one component dataset")
        self.datasets = dict(datasets)
        self.names = list(self.datasets)
        if set(weights) != set(self.datasets):
            missing = sorted(set(self.datasets) - set(weights))
            extra = sorted(set(weights) - set(self.datasets))
            details = []
            if missing:
                details.append(f"missing weights for {', '.join(missing)}")
            if extra:
                details.append(f"unknown weight entries {', '.join(extra)}")
            raise ValueError("MixedDataset weight mismatch: " + "; ".join(details))

        normalized = [max(float(weights[name]), 0.0) for name in self.names]
        weight_sum = sum(normalized)
        if weight_sum <= 0.0:
            raise ValueError("MixedDataset requires at least one positive sampling weight")
        cumulative = []
        running = 0.0
        for weight in normalized:
            running += weight / weight_sum
            cumulative.append(running)
        cumulative[-1] = 1.0

        self.cumulative = cumulative
        self.total_samples = (
            total_samples
            if total_samples is not None
            else max(len(dataset) for dataset in self.datasets.values())
        )
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def sample_source(self, index: int) -> str:
        if self.seed is None:
            draw = random.random()
        else:
            draw = random.Random(self.seed + index).random()
        source_index = min(bisect_right(self.cumulative, draw), len(self.names) - 1)
        return self.names[source_index]

    def __getitem__(self, index: int) -> torch.Tensor:
        source = self.sample_source(index)
        dataset = self.datasets[source]
        dataset_index = index % len(dataset)
        return dataset[dataset_index]
