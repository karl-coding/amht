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
    """Router gating plus chunked sparse local attention."""

    def __init__(self, dim: int, heads: int, router_ratio: float) -> None:
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
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        target = min(max(self.router_ratio, 1e-4), 1 - 1e-4)
        nn.init.zeros_(self.router.weight)
        nn.init.constant_(self.router.bias, math.log(target / (1.0 - target)))

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(x)).squeeze(-1)

    def sparse_attention(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        context = max(1, int(seq_len * self.router_ratio))
        chunk_size = min(256, seq_len)
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
