from __future__ import annotations

from bisect import bisect_right
import random
from collections.abc import Mapping, Sequence
from typing import Any

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
        value_pool_size: int | None = None,
        random_value_mapping: bool = False,
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
        self.value_pool_size = (
            max(int(value_pool_size), num_pairs)
            if value_pool_size is not None
            else self.num_keys
        )
        self.random_value_mapping = bool(random_value_mapping)
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
        tokens, _ = build_retrieval_batch(
            batch_size=1,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            pad_token=self.pad_token,
            key_start=self.key_start,
            value_start=self.value_start,
            num_pairs=self.num_pairs,
            num_keys=self.num_keys,
            value_pool_size=self.value_pool_size,
            random_value_mapping=self.random_value_mapping,
            depth_choices=self.depth_choices,
            rng=rng,
            generator=generator,
        )
        return tokens[0]


def build_retrieval_batch(
    *,
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    pad_token: int = 0,
    key_start: int = 100,
    value_start: int = 1000,
    num_pairs: int = 4,
    num_keys: int = 16,
    value_pool_size: int | None = None,
    random_value_mapping: bool = False,
    depth_choices: Sequence[float] | None = None,
    target_depth: float | None = None,
    rng: random.Random | Any | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("Retrieval batch_size must be > 0")
    if seq_len < 4:
        raise ValueError("Retrieval requires seq_len >= 4")
    if num_pairs <= 0:
        raise ValueError("Retrieval requires num_pairs > 0")
    resolved_num_keys = max(int(num_keys), int(num_pairs))
    resolved_value_pool_size = (
        max(int(value_pool_size), int(num_pairs))
        if value_pool_size is not None
        else resolved_num_keys
    )
    if key_start < 0 or value_start < 0:
        raise ValueError("Retrieval token ranges must start at non-negative ids")
    required_vocab = max(
        key_start + resolved_num_keys,
        value_start + max(resolved_num_keys, resolved_value_pool_size),
    )
    if required_vocab >= vocab_size:
        raise ValueError("Retrieval token ranges must fit inside the configured vocabulary")

    depth_pool = [float(depth) for depth in (depth_choices or [0.1, 0.3, 0.5, 0.7, 0.9])]
    if not depth_pool and target_depth is None:
        raise ValueError("Retrieval requires at least one target depth choice")
    local_rng = rng if rng is not None else random

    background_low = required_vocab + 10
    tokens = _torch_randint(
        background_low,
        vocab_size,
        (batch_size, seq_len),
        generator=generator,
        device=device,
    )
    tokens[:, 0] = pad_token
    expected = torch.empty((batch_size,), dtype=torch.long, device=device)

    for batch_index in range(batch_size):
        selected_keys = local_rng.sample(range(resolved_num_keys), num_pairs)
        if random_value_mapping:
            selected_values = local_rng.sample(range(resolved_value_pool_size), num_pairs)
        else:
            selected_values = list(selected_keys)

        query_pair_index = int(local_rng.randrange(num_pairs))
        depth = float(target_depth) if target_depth is not None else float(local_rng.choice(depth_pool))
        query_position = min(seq_len - 4, max(1, int(seq_len * depth)))

        occupied = {query_position, query_position + 1, seq_len - 2, seq_len - 1}
        available_positions = list(range(1, seq_len - 3))
        local_rng.shuffle(available_positions)

        pair_positions: list[int | None] = [None] * num_pairs
        pair_positions[query_pair_index] = query_position
        distractor_positions: list[int] = []
        for pos in available_positions:
            if len(distractor_positions) >= num_pairs - 1:
                break
            if pos in occupied or (pos + 1) in occupied:
                continue
            distractor_positions.append(pos)
            occupied.add(pos)
            occupied.add(pos + 1)

        if len(distractor_positions) != num_pairs - 1:
            raise ValueError("Retrieval sequence is too short to place all key-value pairs without overlap")

        distractor_iter = iter(sorted(distractor_positions))
        for pair_index in range(num_pairs):
            if pair_positions[pair_index] is None:
                pair_positions[pair_index] = next(distractor_iter)

        for pair_index, pos in enumerate(pair_positions):
            key_token = key_start + selected_keys[pair_index]
            value_token = value_start + selected_values[pair_index]
            tokens[batch_index, pos] = key_token
            tokens[batch_index, pos + 1] = value_token

        query_key_token = key_start + selected_keys[query_pair_index]
        answer_value_token = value_start + selected_values[query_pair_index]
        tokens[batch_index, -2] = query_key_token
        tokens[batch_index, -1] = answer_value_token
        expected[batch_index] = answer_value_token

    return tokens, expected


def _torch_randint(
    low: int,
    high: int,
    size: tuple[int, ...],
    *,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    kwargs = {"dtype": torch.long}
    if generator is not None:
        kwargs["generator"] = generator
    if device is not None:
        kwargs["device"] = device
    return torch.randint(low, high, size, **kwargs)


def _resolve_flipflop_layout(
    *,
    vocab_size: int,
    num_slots: int,
    value_count: int,
    slot_start: int,
    value_start: int | None,
    query_start: int | None,
) -> tuple[int, int, int]:
    if num_slots < 2:
        raise ValueError("Flipflop state-tracking requires num_slots >= 2")
    if value_count < 2:
        raise ValueError("Flipflop state-tracking requires value_count >= 2")
    resolved_value_start = slot_start + num_slots if value_start is None else value_start
    resolved_query_start = resolved_value_start + value_count if query_start is None else query_start
    if slot_start < 0:
        raise ValueError("Flipflop slot_start must be >= 0")
    if resolved_value_start < slot_start + num_slots:
        raise ValueError("Flipflop value_start must come after the slot token range")
    if resolved_query_start < resolved_value_start + value_count:
        raise ValueError("Flipflop query_start must come after the value token range")
    if resolved_query_start + num_slots > vocab_size:
        raise ValueError("Flipflop token ranges must fit inside the configured vocabulary")
    return slot_start, resolved_value_start, resolved_query_start


def build_state_tracking_batch(
    *,
    batch_size: int,
    vocab_size: int,
    seq_len: int,
    task: str = "modsum",
    modulus: int = 16,
    digit_start: int = 0,
    num_slots: int = 8,
    value_count: int = 2,
    slot_start: int = 0,
    value_start: int | None = None,
    query_start: int | None = None,
    min_query_gap_tokens: int = 4096,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size <= 0:
        raise ValueError("State-tracking batch_size must be > 0")
    if seq_len < 2:
        raise ValueError("State-tracking requires seq_len >= 2")

    if task == "modsum":
        if modulus <= 0:
            raise ValueError("StateTrackingDataset requires modulus > 0")
        if digit_start < 0 or digit_start + modulus > vocab_size:
            raise ValueError("StateTrackingDataset digits must fit inside the configured vocabulary")
        digits = _torch_randint(
            digit_start,
            digit_start + modulus,
            (batch_size, seq_len - 1),
            generator=generator,
            device=device,
        )
        expected = ((digits - digit_start).sum(dim=1) % modulus) + digit_start
        tokens = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
        tokens[:, :-1] = digits
        tokens[:, -1] = expected
        return tokens, expected

    if task != "flipflop":
        raise ValueError(f"Unsupported state-tracking task: {task}")
    if (seq_len - 2) % 2 != 0:
        raise ValueError("Flipflop state-tracking requires seq_len - 2 to be divisible by 2")

    slot_start, value_start, query_start = _resolve_flipflop_layout(
        vocab_size=vocab_size,
        num_slots=num_slots,
        value_count=value_count,
        slot_start=slot_start,
        value_start=value_start,
        query_start=query_start,
    )
    operation_count = (seq_len - 2) // 2
    if operation_count < 2:
        raise ValueError("Flipflop state-tracking requires at least two operations before the query")
    effective_gap_tokens = max(int(min_query_gap_tokens), 2)
    effective_gap_pairs = min(max(effective_gap_tokens // 2, 1), operation_count - 1)
    max_last_query_index = max(operation_count - effective_gap_pairs - 1, 0)

    tokens = torch.empty((batch_size, seq_len), dtype=torch.long, device=device)
    expected = torch.empty((batch_size,), dtype=torch.long, device=device)
    for batch_index in range(batch_size):
        query_slot = int(_torch_randint(0, num_slots, (), generator=generator, device=device).item())
        last_query_index = int(
            _torch_randint(0, max_last_query_index + 1, (), generator=generator, device=device).item()
        )
        answer_value = int(_torch_randint(0, value_count, (), generator=generator, device=device).item())

        for op_index in range(operation_count):
            if op_index == last_query_index:
                slot = query_slot
                value = answer_value
            else:
                if op_index > last_query_index:
                    draw = int(_torch_randint(0, num_slots - 1, (), generator=generator, device=device).item())
                    slot = draw if draw < query_slot else draw + 1
                else:
                    slot = int(_torch_randint(0, num_slots, (), generator=generator, device=device).item())
                value = int(_torch_randint(0, value_count, (), generator=generator, device=device).item())
            tokens[batch_index, 2 * op_index] = slot_start + slot
            tokens[batch_index, 2 * op_index + 1] = value_start + value

        tokens[batch_index, -2] = query_start + query_slot
        tokens[batch_index, -1] = value_start + answer_value
        expected[batch_index] = value_start + answer_value
    return tokens, expected


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
        num_slots: int = 8,
        value_count: int = 2,
        slot_start: int = 0,
        value_start: int | None = None,
        query_start: int | None = None,
        min_query_gap_tokens: int = 4096,
        seed: int | None = None,
    ) -> None:
        if seq_len < 2:
            raise ValueError("StateTrackingDataset requires seq_len >= 2")
        if task == "modsum":
            if modulus <= 0:
                raise ValueError("StateTrackingDataset requires modulus > 0")
            if digit_start < 0 or digit_start + modulus > vocab_size:
                raise ValueError("StateTrackingDataset digits must fit inside the configured vocabulary")
        elif task == "flipflop":
            if (seq_len - 2) % 2 != 0:
                raise ValueError("Flipflop state-tracking requires seq_len - 2 to be divisible by 2")
            _resolve_flipflop_layout(
                vocab_size=vocab_size,
                num_slots=num_slots,
                value_count=value_count,
                slot_start=slot_start,
                value_start=value_start,
                query_start=query_start,
            )
        else:
            raise ValueError(f"Unsupported state-tracking task: {task}")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.total_samples = total_samples
        self.task = task
        self.modulus = modulus
        self.digit_start = digit_start
        self.num_slots = num_slots
        self.value_count = value_count
        self.slot_start = slot_start
        self.value_start = value_start
        self.query_start = query_start
        self.min_query_gap_tokens = min_query_gap_tokens
        self.seed = seed

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(self.seed + index)
        tokens, _ = build_state_tracking_batch(
            batch_size=1,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            task=self.task,
            modulus=self.modulus,
            digit_start=self.digit_start,
            num_slots=self.num_slots,
            value_count=self.value_count,
            slot_start=self.slot_start,
            value_start=self.value_start,
            query_start=self.query_start,
            min_query_gap_tokens=self.min_query_gap_tokens,
            generator=generator,
        )
        return tokens[0]


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
