from __future__ import annotations

import random

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class SyntheticDataset:
    """Synthetic token dataset used until a real corpus is wired in."""

    def __init__(self, vocab_size: int, seq_len: int, total_samples: int = 1024) -> None:
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)


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

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        tokens = torch.randint(
            self.value_start + self.num_keys + 10,
            self.vocab_size,
            (self.seq_len,),
            dtype=torch.long,
        )
        tokens[0] = self.pad_token
        selected_keys = random.sample(range(self.num_keys), self.num_pairs)
        target_idx = random.randrange(self.num_pairs)
        target_depth = random.choice(self.depth_choices)
        target_position = min(self.seq_len - 4, max(1, int(self.seq_len * target_depth)))

        occupied = {target_position, target_position + 1, self.seq_len - 2, self.seq_len - 1}
        available_positions = list(range(1, self.seq_len - 3))
        random.shuffle(available_positions)

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
