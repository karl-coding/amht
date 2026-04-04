from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required to run AMHT. Install dependencies from requirements.txt."
    ) from exc


class SurrogateSSMBlock(nn.Module):
    """V3 tokenwise SSM surrogate kept for backward-compatible configs."""

    def __init__(self, dim: int, state_size: int) -> None:
        super().__init__()
        self.in_proj = nn.Linear(dim, state_size)
        self.out_proj = nn.Linear(state_size, dim)
        self.gate = nn.Linear(dim, state_size)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        memory = torch.tanh(self.in_proj(x))
        gated = memory * torch.sigmoid(self.gate(x))
        output = self.out_proj(gated)
        if return_features:
            return output, output
        return output


class SelectiveRecurrentSSM(nn.Module):
    """Grouped recurrent state-space backbone with optional complex-state dynamics."""

    def __init__(
        self,
        dim: int,
        state_size: int,
        groups: int = 4,
        conv_kernel: int = 3,
        complex_state: bool = False,
    ) -> None:
        super().__init__()
        self.groups = max(1, int(groups))
        self.state_size = max(1, int(state_size))
        self.complex_state = bool(complex_state)

        proj_multiplier = 4 if self.complex_state else 2
        self.input_proj = nn.Linear(dim, self.groups)
        self.dt_proj = nn.Linear(dim, self.groups)
        self.bc_proj = nn.Linear(dim, self.groups * self.state_size * proj_multiplier)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(self.groups, dim)
        self.skip = nn.Parameter(torch.ones(self.groups))
        self.a_log = nn.Parameter(torch.zeros(self.groups, self.state_size))
        self.dt_bias = nn.Parameter(torch.zeros(self.groups))
        self.conv = None
        if conv_kernel > 1:
            self.conv = nn.Conv1d(
                self.groups,
                self.groups,
                kernel_size=int(conv_kernel),
                groups=self.groups,
                padding=int(conv_kernel) - 1,
            )
        if self.complex_state:
            self.freq = nn.Parameter(torch.randn(self.groups, self.state_size) * 0.02)

    def _causal_conv(self, u: torch.Tensor) -> torch.Tensor:
        if self.conv is None:
            return u
        mixed = self.conv(u.transpose(1, 2))
        return mixed[:, :, : u.size(1)].transpose(1, 2)

    def _scan_real(self, alpha: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        prefix = torch.cumprod(alpha, dim=1)
        safe_prefix = prefix.clamp_min(1e-6)
        return prefix * torch.cumsum(beta / safe_prefix, dim=1)

    def _scan_complex(self, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        prefix = torch.cumprod(gamma, dim=1)
        eps = torch.full_like(prefix, 1e-6 + 0.0j)
        safe_prefix = torch.where(prefix.abs() < 1e-6, eps, prefix)
        return prefix * torch.cumsum(beta / safe_prefix, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        u = self._causal_conv(self.input_proj(x))
        dt = F.softplus(self.dt_proj(x) + self.dt_bias) + 1e-4
        bc = self.bc_proj(x).view(batch, seq_len, self.groups, self.state_size, -1)
        a = F.softplus(self.a_log).unsqueeze(0).unsqueeze(0) + 1e-3
        dt_expanded = dt.unsqueeze(-1)
        drive = u.unsqueeze(-1)
        decay = torch.exp(-dt_expanded * a)

        if self.complex_state:
            b_real = bc[..., 0]
            b_imag = bc[..., 1]
            c_real = bc[..., 2]
            c_imag = bc[..., 3]
            freq = self.freq.unsqueeze(0).unsqueeze(0)
            angle = dt_expanded * freq
            gamma = torch.complex(decay * torch.cos(angle), decay * torch.sin(angle))
            beta = dt_expanded * torch.complex(b_real, b_imag) * drive
            state = self._scan_complex(gamma, beta)
            readout = torch.real(state * torch.conj(torch.complex(c_real, c_imag))).sum(dim=-1)
        else:
            b = bc[..., 0]
            c = bc[..., 1]
            beta = dt_expanded * b * drive
            state = self._scan_real(decay, beta)
            readout = (state * c).sum(dim=-1)

        recurrent = readout + self.skip.view(1, 1, -1) * u
        features = self.out_proj(recurrent)
        output = features * torch.sigmoid(self.gate_proj(x))
        if return_features:
            return output, features
        return output


class SSMBlock(nn.Module):
    """Configurable SSM wrapper supporting both V3 and V4 backbones."""

    def __init__(
        self,
        dim: int,
        state_size: int,
        impl: str = "surrogate",
        groups: int = 4,
        conv_kernel: int = 3,
        complex_state: bool = False,
    ) -> None:
        super().__init__()
        impl_name = str(impl).lower()
        if impl_name == "selective":
            self.impl = SelectiveRecurrentSSM(
                dim=dim,
                state_size=state_size,
                groups=groups,
                conv_kernel=conv_kernel,
                complex_state=complex_state,
            )
        elif impl_name == "surrogate":
            self.impl = SurrogateSSMBlock(dim=dim, state_size=state_size)
        else:
            raise ValueError(f"Unsupported ssm impl: {impl}")

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.impl(x, return_features=return_features)
