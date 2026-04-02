from __future__ import annotations

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class LatentMemory(nn.Module):
    """Compressed latent memory shared across the sequence."""

    def __init__(self, latent_tokens: int, dim: int) -> None:
        super().__init__()
        self.memory = nn.Parameter(torch.randn(1, latent_tokens, dim) * 0.02)
        self.read_proj = nn.Linear(dim, dim)
        self.write_proj = nn.Linear(dim, dim)

    def forward(self, batch_size: int) -> torch.Tensor:
        latent = self.memory.expand(batch_size, -1, -1)
        return latent.mean(dim=1, keepdim=True)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return self.memory.expand(batch_size, -1, -1)

    def read(self, latent_state: torch.Tensor) -> torch.Tensor:
        return self.read_proj(latent_state.mean(dim=1, keepdim=True))

    def write(self, latent_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        update = self.write_proj(x.mean(dim=1, keepdim=True))
        return latent_state + update.expand_as(latent_state)
