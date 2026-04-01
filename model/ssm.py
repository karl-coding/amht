from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class SSMBlock(nn.Module):
    """Lightweight SSM-style surrogate for compressed recurrent memory."""

    def __init__(self, dim: int, state_size: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, state_size)
        self.out_proj = nn.Linear(state_size, dim)
        self.gate = nn.Linear(dim, state_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = torch.tanh(self.in_proj(x))
        gated = memory * torch.sigmoid(self.gate(x))
        return self.out_proj(gated)
