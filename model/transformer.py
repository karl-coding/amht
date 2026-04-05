from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run Transformer baselines. Install dependencies from requirements.txt."
    ) from exc

from model.memory import LatentMemory


class LocalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attention_window: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by heads={heads}")
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.head_dim = dim // heads
        self.attention_window = attention_window

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, dim = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.heads, self.head_dim).transpose(1, 2)

        chunk_size = min(256, seq_len)
        outputs = []
        scale = self.head_dim ** -0.5
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            ctx_start = max(0, end - self.attention_window)
            q_chunk = q[:, :, start:end, :]
            k_ctx = k[:, :, ctx_start:end, :]
            v_ctx = v[:, :, ctx_start:end, :]
            scores = torch.matmul(q_chunk, k_ctx.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            outputs.append(torch.matmul(attn, v_ctx))

        mixed = torch.cat(outputs, dim=2)
        mixed = mixed.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(mixed)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, heads: int, attention_window: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = LocalSelfAttention(dim, heads, attention_window)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ff(x)
        return x


class LocalTransformerModel(nn.Module):
    def __init__(self, cfg: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        self.max_seq_len = int(model_cfg["max_seq_len"])
        self.vocab_size = int(model_cfg["vocab_size"])
        self.dim = int(model_cfg["dim"])
        attention_window = int(model_cfg.get("attention_window", max(128, self.max_seq_len // 10)))

        self.token_emb = nn.Embedding(self.vocab_size, self.dim)
        self.pos_emb = nn.Parameter(torch.randn(1, self.max_seq_len, self.dim) * 0.02)
        self.memory = LatentMemory(int(model_cfg["latent_tokens"]), self.dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=self.dim,
                    hidden_dim=int(model_cfg["hidden_dim"]),
                    heads=int(model_cfg["heads"]),
                    attention_window=attention_window,
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
        x = x + self.memory(batch)
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
