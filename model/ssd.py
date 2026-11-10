import torch
import torch.nn as nn


def _make_gn_groups(num_channels: int, max_groups: int = 4) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class _AxisProjectorH(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        gn_groups = _make_gn_groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AxisProjectorW(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        gn_groups = _make_gn_groups(channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0), groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.GroupNorm(gn_groups, channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _AxisProjectorC(nn.Module):
    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        gn_groups_hidden = _make_gn_groups(hidden_dim)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(gn_groups_hidden, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuralContextSelection(nn.Module):
    """SCS: average-statistic priors produce the structure gate G_s."""
    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.proj_h = _AxisProjectorH(channels)
        self.proj_w = _AxisProjectorW(channels)
        self.proj_c = _AxisProjectorC(channels, hidden_dim)

        gn_groups = _make_gn_groups(channels)
        self.out_norm = nn.GroupNorm(gn_groups, channels)
        self.alpha_s = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        a_h = x_nchw.mean(dim=2, keepdim=True)
        a_w = x_nchw.mean(dim=3, keepdim=True)
        a_c = x_nchw.mean(dim=1, keepdim=True)

        h_feat = self.proj_h(a_h)
        w_feat = self.proj_w(a_w)
        c_feat = self.proj_c(a_c)

        hw_feat = h_feat * w_feat
        z = hw_feat * (1.0 + self.alpha_s * c_feat)
        z = self.out_norm(z)

        return torch.sigmoid(z)


class DetailedSpatialVariation(nn.Module):
    """DSV: max-statistic priors produce the detail bias B_d."""
    def __init__(self, channels: int, hidden_dim: int):
        super().__init__()
        self.proj_h = _AxisProjectorH(channels)
        self.proj_w = _AxisProjectorW(channels)
        self.proj_c = _AxisProjectorC(channels, hidden_dim)

        gn_groups = _make_gn_groups(channels)
        self.out_norm = nn.GroupNorm(gn_groups, channels)
        self.alpha_d = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, x_nchw: torch.Tensor) -> torch.Tensor:
        m_h = torch.amax(x_nchw, dim=2, keepdim=True)
        m_w = torch.amax(x_nchw, dim=3, keepdim=True)
        m_c = torch.amax(x_nchw, dim=1, keepdim=True)

        h_feat = self.proj_h(m_h)
        w_feat = self.proj_w(m_w)
        c_feat = self.proj_c(m_c)

        hw_feat = h_feat * w_feat
        z = hw_feat + self.alpha_d * c_feat
        z = self.out_norm(z)

        return torch.tanh(z)


class SelectiveStructureDetailDecoupling(nn.Module):
    """SSD: Selective Structure-Detail Decoupling."""
    def __init__(
        self,
        channels: int = 64,
        reduction: int = 4,
        min_hidden_dim: int = 8,
        lambda_s_init: float = 0.1,
        lambda_d_init: float = 0.1,
    ):
        super().__init__()
        hidden_dim = max(channels // reduction, min_hidden_dim)

        self.scs = StructuralContextSelection(channels, hidden_dim)
        self.dsv = DetailedSpatialVariation(channels, hidden_dim)

        self.lambda_s = nn.Parameter(torch.tensor(lambda_s_init, dtype=torch.float32))
        self.lambda_d = nn.Parameter(torch.tensor(lambda_d_init, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,H,W,C), but got shape {tuple(x.shape)}")

        x_nchw = x.permute(0, 3, 1, 2).contiguous()

        g_s = self.scs(x_nchw)
        f_s = x_nchw * (1.0 + self.lambda_s * g_s)

        b_d = self.dsv(x_nchw)
        f_d = x_nchw + self.lambda_d * b_d

        f_s = f_s.permute(0, 2, 3, 1).contiguous()
        f_d = f_d.permute(0, 2, 3, 1).contiguous()

        return f_s, f_d
