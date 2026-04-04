from __future__ import annotations

import math

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class SparseRouter(nn.Module):
    """Block router plus chunked sparse local attention on routed blocks only."""

    def __init__(
        self,
        dim: int,
        heads: int,
        router_ratio: float,
        chunk_size: int = 256,
        block_size: int = 128,
        neighbor_radius: int = 0,
        neighbor_bonus: float = 0.0,
        feature_sources: int = 2,
        expand_mode: str = "bonus",
        straight_through_scores: bool = False,
        straight_through_temperature: float = 0.1,
        straight_through_scale: float = 0.1,
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.feature_sources = max(2, int(feature_sources))
        self.router = nn.Sequential(
            nn.Linear(dim * self.feature_sources, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.router_ratio = router_ratio
        self.chunk_size = chunk_size
        self.block_size = block_size
        self.neighbor_radius = max(0, int(neighbor_radius))
        self.neighbor_bonus = max(0.0, float(neighbor_bonus))
        self.expand_mode = str(expand_mode).lower()
        self.straight_through_scores = bool(straight_through_scores)
        self.straight_through_temperature = max(1e-3, float(straight_through_temperature))
        self.straight_through_scale = float(straight_through_scale)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        target = min(max(self.router_ratio, 1e-4), 1 - 1e-4)
        first, _, second = self.router
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(second.weight)
        nn.init.constant_(second.bias, math.log(target / (1.0 - target)))

    def _pad_to_blocks(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch, seq_len, dim = x.shape
        block_size = min(self.block_size, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len
        if pad > 0:
            x = torch.cat([x, x.new_zeros(batch, pad, dim)], dim=1)
        return x, num_blocks, block_size

    def _summarize_source(
        self,
        source: torch.Tensor | None,
        num_blocks: int,
        block_size: int,
        seq_len: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if source is None:
            return torch.zeros_like(reference)
        if source.size(1) == 1:
            return source.expand(-1, num_blocks, -1)
        if source.size(1) != seq_len:
            pooled = source.mean(dim=1, keepdim=True)
            return pooled.expand(-1, num_blocks, -1)
        padded, _, _ = self._pad_to_blocks(source)
        blocks = padded.view(source.size(0), num_blocks, block_size, source.size(-1))
        return blocks.mean(dim=2)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(x)
        features = [x, zeros]
        while len(features) < self.feature_sources:
            features.append(zeros)
        return torch.sigmoid(self.router(torch.cat(features, dim=-1))).squeeze(-1)

    def _local_context(self, block_summary: torch.Tensor) -> torch.Tensor:
        num_blocks = block_summary.size(1)
        local_context = torch.zeros_like(block_summary)
        if num_blocks > 1:
            local_context[:, 1:] += block_summary[:, :-1]
            local_context[:, :-1] += block_summary[:, 1:]
            neighbor_count = torch.zeros_like(block_summary[..., :1])
            neighbor_count[:, 1:] += 1
            neighbor_count[:, :-1] += 1
            local_context = local_context / neighbor_count
        return local_context

    def _bonus_adjusted_scores(self, block_scores: torch.Tensor, topk: int) -> torch.Tensor:
        adjusted_scores = block_scores
        num_blocks = block_scores.size(1)
        if self.neighbor_radius > 0 and self.neighbor_bonus > 0.0 and num_blocks > 1:
            seed_indices = torch.topk(block_scores, k=topk, dim=-1).indices
            seed_mask = torch.zeros_like(block_scores, dtype=torch.bool)
            seed_mask.scatter_(1, seed_indices, True)
            locality_bonus = torch.zeros_like(block_scores)
            for offset in range(1, self.neighbor_radius + 1):
                decay = float(self.neighbor_radius - offset + 1) / float(self.neighbor_radius + 1)
                left_bonus = seed_mask[:, :-offset].to(block_scores.dtype) * decay
                right_bonus = seed_mask[:, offset:].to(block_scores.dtype) * decay
                locality_bonus[:, offset:] = torch.maximum(locality_bonus[:, offset:], left_bonus)
                locality_bonus[:, :-offset] = torch.maximum(locality_bonus[:, :-offset], right_bonus)
            adjusted_scores = block_scores + self.neighbor_bonus * locality_bonus
        return adjusted_scores

    def _select_with_expansion(self, block_scores: torch.Tensor, topk: int) -> torch.Tensor:
        batch, num_blocks = block_scores.shape
        selected = torch.zeros_like(block_scores, dtype=torch.bool)
        if self.neighbor_radius <= 0 or num_blocks <= 1:
            selected.scatter_(1, torch.topk(block_scores, k=topk, dim=-1).indices, True)
            return selected

        span = max(1, 1 + 2 * self.neighbor_radius)
        seed_k = max(1, min(num_blocks, int(math.ceil(topk / span))))
        for batch_idx in range(batch):
            scores = block_scores[batch_idx]
            seed_indices = torch.topk(scores, k=seed_k, dim=-1).indices.tolist()
            chosen: set[int] = set(seed_indices)
            candidate_scores: dict[int, float] = {idx: float(scores[idx].item()) for idx in seed_indices}

            for seed_idx in seed_indices:
                for offset in range(1, self.neighbor_radius + 1):
                    decay = float(self.neighbor_radius - offset + 1) / float(self.neighbor_radius + 1)
                    for neighbor in (seed_idx - offset, seed_idx + offset):
                        if 0 <= neighbor < num_blocks:
                            boosted = float(scores[neighbor].item()) + self.neighbor_bonus * decay
                            best = candidate_scores.get(neighbor)
                            if best is None or boosted > best:
                                candidate_scores[neighbor] = boosted

            ranked = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
            for index, _ in ranked:
                chosen.add(index)
                if len(chosen) >= topk:
                    break

            if len(chosen) < topk:
                fallback = torch.topk(scores, k=topk, dim=-1).indices.tolist()
                for index in fallback:
                    chosen.add(index)
                    if len(chosen) >= topk:
                        break

            selected[batch_idx, list(chosen)[:topk]] = True
        return selected

    def block_gate(
        self,
        x: torch.Tensor,
        recurrent_context: torch.Tensor | None = None,
        latent_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        padded_x, num_blocks, block_size = self._pad_to_blocks(x)
        blocks = padded_x.view(batch, num_blocks, block_size, x.size(-1))
        block_summary = blocks.mean(dim=2)
        local_context = self._local_context(block_summary)
        recurrent_summary = self._summarize_source(
            recurrent_context,
            num_blocks=num_blocks,
            block_size=block_size,
            seq_len=seq_len,
            reference=block_summary,
        )
        latent_summary = self._summarize_source(
            latent_context,
            num_blocks=num_blocks,
            block_size=block_size,
            seq_len=seq_len,
            reference=block_summary,
        )

        features = [block_summary, local_context]
        optional_features = [recurrent_summary, latent_summary]
        for feature_index in range(self.feature_sources - 2):
            features.append(optional_features[feature_index] if feature_index < len(optional_features) else torch.zeros_like(block_summary))

        router_input = torch.cat(features, dim=-1)
        block_scores = torch.sigmoid(self.router(router_input)).squeeze(-1)

        target_blocks = max(1, int(math.ceil(num_blocks * self.router_ratio)))
        topk = min(target_blocks, num_blocks)
        if self.expand_mode == "expand":
            selected = self._select_with_expansion(block_scores, topk)
        else:
            adjusted_scores = self._bonus_adjusted_scores(block_scores, topk)
            topk_indices = torch.topk(adjusted_scores, k=topk, dim=-1).indices
            selected = torch.zeros_like(block_scores, dtype=torch.bool)
            selected.scatter_(1, topk_indices, True)

        expanded = selected.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch, num_blocks * block_size)
        token_mask = expanded[:, :seq_len]

        selection_gate = selected.float()
        if self.straight_through_scores:
            threshold = torch.topk(block_scores, k=topk, dim=-1).values[:, -1:].detach()
            soft_selection = torch.sigmoid((block_scores - threshold) / self.straight_through_temperature)
            selection_gate = selection_gate + soft_selection - soft_selection.detach()

        return block_scores, token_mask, selected, selection_gate

    def sparse_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        context = max(1, int(seq_len * self.router_ratio))
        chunk_size = min(self.chunk_size, seq_len)
        outputs = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            ctx_start = max(0, end - context)
            q_chunk = q[:, :, start:end, :]
            k_ctx = k[:, :, ctx_start:end, :]
            v_ctx = v[:, :, ctx_start:end, :]
            scores = torch.matmul(q_chunk, k_ctx.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(attn, v_ctx))

        mixed = torch.cat(outputs, dim=2)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(mixed)

    def routed_sparse_attention(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        selected_blocks: torch.Tensor,
        selection_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        block_size = min(self.block_size, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len
        window_blocks = max(1, int(math.ceil(num_blocks * self.router_ratio)))

        if pad > 0:
            padded_x = torch.cat([x, x.new_zeros(batch, pad, dim)], dim=1)
            padded_mask = torch.cat([token_mask, token_mask.new_zeros(batch, pad)], dim=1)
        else:
            padded_x = x
            padded_mask = token_mask

        blocks = padded_x.view(batch, num_blocks, block_size, dim)
        outputs = torch.zeros_like(blocks)

        routed_per_batch = int(selected_blocks.sum(dim=1).max().item())
        if routed_per_batch == 0:
            return x.new_zeros(batch, seq_len, dim)

        batch_indices = torch.arange(batch, device=x.device)[:, None]
        block_positions = torch.arange(num_blocks, device=x.device).expand(batch, -1)
        selected_positions = torch.where(selected_blocks, block_positions, num_blocks)
        selected_indices = torch.sort(selected_positions, dim=1).values[:, :routed_per_batch]
        q_blocks = blocks[batch_indices, selected_indices]

        q_tokens = q_blocks.reshape(batch * routed_per_batch, block_size, dim)
        q = self.q_proj(q_tokens).view(batch, routed_per_batch, block_size, self.heads, self.head_dim)
        q = q.permute(0, 1, 3, 2, 4)

        kv_tokens = q_blocks.reshape(batch, routed_per_batch * block_size, dim)
        k = self.k_proj(kv_tokens).view(batch, routed_per_batch * block_size, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_tokens).view(batch, routed_per_batch * block_size, self.heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k[:, None].transpose(-2, -1)) / (self.head_dim ** 0.5)

        query_positions = selected_indices[:, :, None]
        context_positions = selected_indices[:, None, :]
        valid_context = (context_positions <= query_positions) & (
            context_positions >= (query_positions - window_blocks + 1)
        )
        valid_tokens = valid_context.unsqueeze(-1).expand(-1, -1, -1, block_size).reshape(
            batch, routed_per_batch, routed_per_batch * block_size
        )
        scores = scores.masked_fill(~valid_tokens[:, :, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        mixed = torch.matmul(attn, v[:, None])
        mixed = mixed.permute(0, 1, 3, 2, 4).contiguous().view(batch, routed_per_batch, block_size, dim)
        mixed = self.out_proj(mixed.view(batch * routed_per_batch, block_size, dim)).view(
            batch,
            routed_per_batch,
            block_size,
            dim,
        )

        outputs.scatter_(
            1,
            selected_indices[:, :, None, None].expand(-1, -1, block_size, dim),
            mixed,
        )

        output_tokens = outputs.view(batch, num_blocks * block_size, dim)[:, :seq_len]
        output_tokens = output_tokens * padded_mask[:, :seq_len].unsqueeze(-1).to(output_tokens.dtype)

        if self.straight_through_scores and selection_gate is not None:
            gate_tokens = selection_gate.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch, num_blocks * block_size)
            gate_tokens = gate_tokens[:, :seq_len].unsqueeze(-1).to(output_tokens.dtype)
            # Forward pass stays unchanged; backward pass exposes a task-driven signal to router scores.
            output_tokens = output_tokens + self.straight_through_scale * x * (gate_tokens - gate_tokens.detach())

        return output_tokens
