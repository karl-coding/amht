from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc

from model.memory import LatentMemory
from model.router import SparseRouter
from model.ssm import SSMBlock


@dataclass
class LossBreakdown:
    total: torch.Tensor
    main: torch.Tensor
    router: torch.Tensor
    router_mean: torch.Tensor


class AMHTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        heads: int,
        router_ratio: float,
        state_size: int,
        attention_chunk_size: int,
        block_size: int,
        memory: LatentMemory,
        router_neighbor_radius: int = 0,
        router_neighbor_bonus: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SSMBlock(dim, state_size)
        self.router = SparseRouter(
            dim,
            heads,
            router_ratio,
            attention_chunk_size,
            block_size,
            router_neighbor_radius,
            router_neighbor_bonus,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.memory = memory
        self.router_ratio = router_ratio

    def forward(
        self,
        x: torch.Tensor,
        latent_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.norm(x)
        latent_read = self.memory.read(latent_state, h)
        h_with_memory = h + latent_read
        block_scores, token_mask, selected_blocks = self.router.block_gate(h_with_memory)
        mixed = self.ssm(h_with_memory) + self.router.routed_sparse_attention(h_with_memory, token_mask, selected_blocks)
        x = x + mixed
        x = x + self.ff(x)
        latent_state = self.memory.write(latent_state, x)
        router_mean = block_scores.mean()
        router_penalty = (router_mean - self.router_ratio).pow(2)
        return x, latent_state, router_penalty, router_mean


class AMHTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.max_seq_len = int(model_cfg["max_seq_len"])
        self.vocab_size = int(model_cfg["vocab_size"])
        self.dim = int(model_cfg["dim"])

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, self.dim) * 0.02)
        self.memory = LatentMemory(int(model_cfg["latent_tokens"]), self.dim)
        self.blocks = nn.ModuleList(
            [
                AMHTBlock(
                    dim=self.dim,
                    hidden_dim=int(model_cfg["hidden_dim"]),
                    heads=int(model_cfg["heads"]),
                    router_ratio=float(model_cfg["router_ratio"]),
                    state_size=int(model_cfg["ssm_state_size"]),
                    attention_chunk_size=int(model_cfg.get("attention_chunk_size", 256)),
                    block_size=int(model_cfg.get("block_size", 128)),
                    memory=self.memory,
                    router_neighbor_radius=int(model_cfg.get("router_neighbor_radius", 0)),
                    router_neighbor_bonus=float(model_cfg.get("router_neighbor_bonus", 0.0)),
                )
                for _ in range(int(model_cfg["layers"]))
            ]
        )
        self.norm = nn.LayerNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_emb(tokens) + self.pos_emb[:, :seq_len]
        latent_state = self.memory.init_state(batch_size=batch)
        x = x + self.memory.read(latent_state, x)

        router_penalty = x.new_zeros(())
        router_mean = x.new_zeros(())
        for block in self.blocks:
            x, latent_state, penalty, mean_score = block(x, latent_state)
            router_penalty = router_penalty + penalty
            router_mean = router_mean + mean_score

        stats = {
            "router_penalty": router_penalty / max(len(self.blocks), 1),
            "router_mean": router_mean / max(len(self.blocks), 1),
        }
        logits = self.lm_head(self.norm(x))
        return logits, stats


def synthetic_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len), device=device)


def compute_loss(
    model: nn.Module,
    tokens: torch.Tensor,
    main_weight: float,
    router_weight: float,
    loss_mode: str = "next_token",
) -> LossBreakdown:
    if loss_mode == "final_token":
        inputs = tokens[:, :-1]
        targets = tokens[:, -1]
        logits, stats = model(inputs)
        main_loss = F.cross_entropy(logits[:, -1, :], targets)
    else:
        inputs = tokens[:, :-1]
        targets = tokens[:, 1:]
        logits, stats = model(inputs)
        main_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    router_penalty = stats["router_penalty"]
    total = main_weight * main_loss + router_weight * router_penalty
    return LossBreakdown(
        total=total,
        main=main_loss,
        router=router_penalty,
        router_mean=stats["router_mean"],
    )
