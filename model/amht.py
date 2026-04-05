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

from model.memory import LatentMemory, LatentMemoryIO, LatentMemoryState
from model.router import SparseRouter
from model.ssm import SSMBlock


@dataclass
class LossBreakdown:
    total: torch.Tensor
    main: torch.Tensor
    router: torch.Tensor
    router_mean: torch.Tensor
    router_selected_ratio: torch.Tensor
    router_selected_score_mean: torch.Tensor
    router_unselected_score_mean: torch.Tensor
    router_score_gap: torch.Tensor


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
        memory_io: LatentMemoryIO | None,
        ssm_impl: str = "surrogate",
        ssm_groups: int = 4,
        ssm_conv_kernel: int = 3,
        ssm_complex: bool = False,
        router_neighbor_radius: int = 0,
        router_neighbor_bonus: float = 0.0,
        router_feature_sources: int = 2,
        router_expand_mode: str = "bonus",
        router_straight_through_scores: bool = False,
        router_straight_through_temperature: float = 0.1,
        router_straight_through_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ssm = SSMBlock(
            dim=dim,
            state_size=state_size,
            impl=ssm_impl,
            groups=ssm_groups,
            conv_kernel=ssm_conv_kernel,
            complex_state=ssm_complex,
        )
        self.router = SparseRouter(
            dim,
            heads,
            router_ratio,
            attention_chunk_size,
            block_size,
            router_neighbor_radius,
            router_neighbor_bonus,
            feature_sources=router_feature_sources,
            expand_mode=router_expand_mode,
            straight_through_scores=router_straight_through_scores,
            straight_through_temperature=router_straight_through_temperature,
            straight_through_scale=router_straight_through_scale,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.memory = memory_io
        self.router_ratio = router_ratio

    def forward(
        self,
        x: torch.Tensor,
        latent_state: torch.Tensor,
        memory_io: LatentMemory | LatentMemoryIO | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        memory = memory_io if memory_io is not None else self.memory
        if memory is None:
            raise ValueError("AMHTBlock requires a memory I/O module")

        h = self.norm(x)
        latent_read = memory.read(latent_state, h)
        h_with_memory = h + latent_read
        ssm_out, recurrent_features = self.ssm(h_with_memory, return_features=True)
        block_scores, token_mask, selected_blocks, selection_gate, selection_stats = self.router.block_gate(
            h_with_memory,
            recurrent_context=recurrent_features,
            latent_context=latent_state.mean(dim=1, keepdim=True),
        )
        mixed = ssm_out + self.router.routed_sparse_attention(
            h_with_memory,
            token_mask,
            selected_blocks,
            selection_gate=selection_gate,
        )
        x = x + mixed
        x = x + self.ff(x)
        latent_state = memory.write(latent_state, x)
        router_mean = block_scores.mean()
        return x, latent_state, {
            "router_mean": router_mean,
            "router_selected_ratio": selection_stats["selected_ratio"],
            "router_selected_score_mean": selection_stats["selected_score_mean"],
            "router_unselected_score_mean": selection_stats["unselected_score_mean"],
            "router_score_gap": selection_stats["score_gap"],
        }


class AMHTModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.max_seq_len = int(model_cfg["max_seq_len"])
        self.vocab_size = int(model_cfg["vocab_size"])
        self.dim = int(model_cfg["dim"])
        self.memory_per_layer_io = bool(model_cfg.get("memory_per_layer_io", False))

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, self.dim) * 0.02)

        layers = int(model_cfg["layers"])
        latent_tokens = int(model_cfg["latent_tokens"])
        if self.memory_per_layer_io:
            self.memory_state = LatentMemoryState(latent_tokens, self.dim)
            self.input_memory = LatentMemoryIO(self.dim)
            memory_ios: list[LatentMemoryIO | None] = [LatentMemoryIO(self.dim) for _ in range(layers)]
        else:
            self.memory = LatentMemory(latent_tokens, self.dim)
            self.memory_state = None
            self.input_memory = None
            memory_ios = [None for _ in range(layers)]

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
                    memory_io=memory_ios[layer_index],
                    ssm_impl=str(model_cfg.get("ssm_impl", "surrogate")),
                    ssm_groups=int(model_cfg.get("ssm_groups", max(1, int(model_cfg.get("heads", 1))))),
                    ssm_conv_kernel=int(model_cfg.get("ssm_conv_kernel", 3)),
                    ssm_complex=bool(model_cfg.get("ssm_complex", False)),
                    router_neighbor_radius=int(model_cfg.get("router_neighbor_radius", 0)),
                    router_neighbor_bonus=float(model_cfg.get("router_neighbor_bonus", 0.0)),
                    router_feature_sources=int(model_cfg.get("router_feature_sources", 2)),
                    router_expand_mode=str(model_cfg.get("router_expand_mode", "bonus")),
                    router_straight_through_scores=bool(model_cfg.get("router_straight_through_scores", False)),
                    router_straight_through_temperature=float(model_cfg.get("router_straight_through_temperature", 0.1)),
                    router_straight_through_scale=float(model_cfg.get("router_straight_through_scale", 0.1)),
                )
                for layer_index in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, self.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch, seq_len = tokens.shape
        if seq_len > self.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_emb(tokens) + self.pos_emb[:, :seq_len]
        if self.memory_per_layer_io:
            if self.memory_state is None or self.input_memory is None:
                raise ValueError("Per-layer memory I/O requires memory_state and input_memory")
            latent_state = self.memory_state.init_state(batch_size=batch)
            x = x + self.input_memory.read(latent_state, x)
            shared_memory: LatentMemory | LatentMemoryIO | None = None
        else:
            latent_state = self.memory.init_state(batch_size=batch)
            x = x + self.memory.read(latent_state, x)
            shared_memory = self.memory

        router_mean = x.new_zeros(())
        router_selected_ratio = x.new_zeros(())
        router_selected_score_mean = x.new_zeros(())
        router_unselected_score_mean = x.new_zeros(())
        router_score_gap = x.new_zeros(())
        for block in self.blocks:
            x, latent_state, block_stats = block(x, latent_state, memory_io=shared_memory)
            router_mean = router_mean + block_stats["router_mean"]
            router_selected_ratio = router_selected_ratio + block_stats["router_selected_ratio"]
            router_selected_score_mean = router_selected_score_mean + block_stats["router_selected_score_mean"]
            router_unselected_score_mean = router_unselected_score_mean + block_stats["router_unselected_score_mean"]
            router_score_gap = router_score_gap + block_stats["router_score_gap"]

        stats = {
            "router_mean": router_mean / max(len(self.blocks), 1),
            "router_selected_ratio": router_selected_ratio / max(len(self.blocks), 1),
            "router_selected_score_mean": router_selected_score_mean / max(len(self.blocks), 1),
            "router_unselected_score_mean": router_unselected_score_mean / max(len(self.blocks), 1),
            "router_score_gap": router_score_gap / max(len(self.blocks), 1),
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
    router_mean_target: float = 0.1,
    router_mean_weight: float = 1.0,
    router_score_margin: float = 0.02,
    router_score_weight: float = 0.0,
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
    router_mean_penalty = (stats["router_mean"] - float(router_mean_target)).pow(2)
    router_score_penalty = F.relu(float(router_score_margin) - stats["router_score_gap"]).pow(2)
    router_penalty = (
        float(router_mean_weight) * router_mean_penalty
        + float(router_score_weight) * router_score_penalty
    )
    total = main_weight * main_loss + router_weight * router_penalty
    return LossBreakdown(
        total=total,
        main=main_loss,
        router=router_penalty,
        router_mean=stats["router_mean"],
        router_selected_ratio=stats["router_selected_ratio"],
        router_selected_score_mean=stats["router_selected_score_mean"],
        router_unselected_score_mean=stats["router_unselected_score_mean"],
        router_score_gap=stats["router_score_gap"],
    )
