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
    ) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.router = nn.Linear(dim, 1)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.router_ratio = router_ratio
        self.chunk_size = chunk_size
        self.block_size = block_size
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        target = min(max(self.router_ratio, 1e-4), 1 - 1e-4)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, math.log(target / (1.0 - target)))

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(x)).squeeze(-1)

    def block_gate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        block_size = min(self.block_size, seq_len)
        num_blocks = (seq_len + block_size - 1) // block_size
        pad = num_blocks * block_size - seq_len
        if pad > 0:
            x = torch.cat([x, x.new_zeros(batch, pad, x.size(-1))], dim=1)
        blocks = x.view(batch, num_blocks, block_size, x.size(-1))
        block_summary = blocks.mean(dim=2)
        block_scores = torch.sigmoid(self.router(block_summary)).squeeze(-1)

        target_blocks = max(1, int(math.ceil(num_blocks * self.router_ratio)))
        topk = min(target_blocks, num_blocks)
        topk_indices = torch.topk(block_scores, k=topk, dim=-1).indices
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

        if pad > 0:
            padded_x = torch.cat([x, x.new_zeros(batch, pad, dim)], dim=1)
            padded_mask = torch.cat([token_mask, token_mask.new_zeros(batch, pad)], dim=1)
        else:
            padded_x = x
            padded_mask = token_mask

        blocks = padded_x.view(batch, num_blocks, block_size, dim)
        outputs = torch.zeros_like(blocks)

        for batch_idx in range(batch):
            active_blocks = torch.nonzero(selected_blocks[batch_idx], as_tuple=False).flatten()
            if active_blocks.numel() == 0:
                continue

            for block_index in active_blocks.tolist():
                start_block = max(0, block_index - max(1, int(math.ceil(num_blocks * self.router_ratio))) + 1)
                end_block = block_index + 1
                q_tokens = blocks[batch_idx : batch_idx + 1, block_index].reshape(1, block_size, dim)
                kv_tokens = blocks[batch_idx : batch_idx + 1, start_block:end_block].reshape(
                    1, (end_block - start_block) * block_size, dim
                )

                q = self.q_proj(q_tokens).view(1, block_size, self.heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(kv_tokens).view(1, kv_tokens.size(1), self.heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(kv_tokens).view(1, kv_tokens.size(1), self.heads, self.head_dim).transpose(1, 2)

                scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn = torch.softmax(scores, dim=-1)
                mixed = torch.matmul(attn, v)
                mixed = mixed.transpose(1, 2).contiguous().view(1, block_size, dim)
                outputs[batch_idx, block_index] = self.out_proj(mixed)[0]

        output_tokens = outputs.view(batch, num_blocks * block_size, dim)[:, :seq_len]
        return output_tokens * padded_mask[:, :seq_len].unsqueeze(-1).to(output_tokens.dtype)
