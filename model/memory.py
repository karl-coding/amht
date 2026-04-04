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
        self.read_q = nn.Linear(dim, dim)
        self.read_k = nn.Linear(dim, dim)
        self.read_v = nn.Linear(dim, dim)
        self.read_out = nn.Linear(dim, dim)
        self.write_q = nn.Linear(dim, dim)
        self.write_k = nn.Linear(dim, dim)
        self.write_v = nn.Linear(dim, dim)
        self.write_out = nn.Linear(dim, dim)
        self.write_gate = nn.Linear(dim, dim)

    def forward(self, batch_size: int) -> torch.Tensor:
        latent = self.memory.expand(batch_size, -1, -1)
        return latent.mean(dim=1, keepdim=True)

    def init_state(self, batch_size: int) -> torch.Tensor:
        return self.memory.expand(batch_size, -1, -1).clone()

    def read(self, latent_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.read_q(x)
        k = self.read_k(latent_state)
        v = self.read_v(latent_state)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, v)
        return self.read_out(context)

    def write(self, latent_state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        q = self.write_q(latent_state)
        k = self.write_k(x)
        v = self.write_v(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        update = self.write_out(torch.matmul(attn, v))
        gate = torch.sigmoid(self.write_gate(latent_state))
        return latent_state + gate * update
