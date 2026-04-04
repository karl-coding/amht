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
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.router = nn.Sequential(
            nn.Linear(dim * 2, dim),
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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        target = min(max(self.router_ratio, 1e-4), 1 - 1e-4)
        first, _, second = self.router
        nn.init.xavier_uniform_(first.weight)
        nn.init.zeros_(first.bias)
        nn.init.zeros_(second.weight)
        nn.init.constant_(second.bias, math.log(target / (1.0 - target)))

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(x)
        return torch.sigmoid(self.router(torch.cat([x, zeros], dim=-1))).squeeze(-1)

    def block_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        block_size = min(self.block_size, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len
        if pad > 0:
            x = torch.cat([x, x.new_zeros(batch, pad, x.size(-1))], dim=1)
        blocks = x.view(batch, num_blocks, block_size, x.size(-1))
        block_summary = blocks.mean(dim=2)

        # Add a coarse local-context summary so routing depends on both block content
        # and nearby sequence state, not only the block's own mean-pooled representation.
        local_context = torch.zeros_like(block_summary)
        if num_blocks > 1:
            local_context[:, 1:] += block_summary[:, :-1]
            local_context[:, :-1] += block_summary[:, 1:]
            neighbor_count = torch.zeros_like(block_summary[..., :1])
            neighbor_count[:, 1:] += 1
            neighbor_count[:, :-1] += 1
            local_context = local_context / neighbor_count
        router_input = torch.cat([block_summary, local_context], dim=-1)
        block_scores = torch.sigmoid(self.router(router_input)).squeeze(-1)

        target_blocks = max(1, int(math.ceil(num_blocks * self.router_ratio)))
        topk = min(target_blocks, num_blocks)

        adjusted_scores = block_scores
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

        topk_indices = torch.topk(adjusted_scores, k=topk, dim=-1).indices
        selected = torch.zeros_like(block_scores, dtype=torch.bool)
        selected.scatter_(1, topk_indices, True)

        expanded = selected.unsqueeze(-1).expand(-1, -1, block_size).reshape(batch, num_blocks * block_size)
        token_mask = expanded[:, :seq_len]
        return block_scores, token_mask, selected

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
        mixed = self.out_proj(mixed.view(batch * routed_per_batch, block_size, dim)).view(batch, routed_per_batch, block_size, dim)

        outputs.scatter_(
            1,
            selected_indices[:, :, None, None].expand(-1, -1, block_size, dim),
            mixed,
        )

        output_tokens = outputs.view(batch, num_blocks * block_size, dim)[:, :seq_len]
        return output_tokens * padded_mask[:, :seq_len].unsqueeze(-1).to(output_tokens.dtype)
