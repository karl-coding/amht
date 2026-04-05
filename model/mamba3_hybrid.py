from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run Mamba3-hybrid baselines. Install dependencies from requirements.txt."
    ) from exc

from model.ssm import SSMBlock
from model.transformer import LocalSelfAttention


class Mamba3HybridBlock(nn.Module):
    """Fixed-period SSM + sparse attention hybrid used as a non-AMHT baseline."""

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        heads: int,
        attention_window: int,
        use_attention: bool,
        ssm_state_size: int,
        ssm_groups: int,
        ssm_conv_kernel: int,
        ssm_complex: bool,
    ) -> None:
        super().__init__()
        self.ssm_norm = nn.LayerNorm(dim)
        self.ssm = SSMBlock(
            dim=dim,
            state_size=ssm_state_size,
            impl="selective",
            groups=ssm_groups,
            conv_kernel=ssm_conv_kernel,
            complex_state=ssm_complex,
        )
        self.use_attention = bool(use_attention)
        if self.use_attention:
            self.attn_norm = nn.LayerNorm(dim)
            self.attn = LocalSelfAttention(dim, heads, attention_window)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.ssm_norm(x))
        if self.use_attention:
            x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(x)
        return x


class Mamba3HybridModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.max_seq_len = int(model_cfg["max_seq_len"])
        self.vocab_size = int(model_cfg["vocab_size"])
        self.dim = int(model_cfg["dim"])
        layers = int(model_cfg["layers"])
        heads = int(model_cfg["heads"])
        attention_ratio = float(model_cfg.get("router_ratio", 0.1))
        attention_window = int(model_cfg.get("attention_window", max(128, int(self.max_seq_len * attention_ratio))))
        attention_every = max(1, int(model_cfg.get("attention_every", 2)))
        attention_offset = int(model_cfg.get("attention_offset", attention_every - 1))

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, self.dim) * 0.02)
        self.blocks = nn.ModuleList(
            [
                Mamba3HybridBlock(
                    dim=self.dim,
                    hidden_dim=int(model_cfg["hidden_dim"]),
                    heads=heads,
                    attention_window=attention_window,
                    use_attention=((layer_index - attention_offset) % attention_every == 0),
                    ssm_state_size=int(model_cfg["ssm_state_size"]),
                    ssm_groups=int(model_cfg.get("ssm_groups", max(1, heads))),
                    ssm_conv_kernel=int(model_cfg.get("ssm_conv_kernel", 3)),
                    ssm_complex=bool(model_cfg.get("ssm_complex", False)),
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
        for block in self.blocks:
            x = block(x)

        logits = self.lm_head(self.norm(x))
        zero = x.new_zeros(())
        return logits, {
            "router_penalty": zero,
            "router_mean": zero,
            "router_selected_ratio": zero,
            "router_selected_score_mean": zero,
            "router_unselected_score_mean": zero,
            "router_score_gap": zero,
        }
