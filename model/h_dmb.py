import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmamba2d import VMamba2DWrapper


_DCT_MATRIX_CACHE = {}
_DCT_MASK_CACHE = {}


def get_dct_matrix(n: int, device=None, dtype=None):
    """Return a cached orthonormal DCT-II basis.

    The basis depends only on spatial size, device, and dtype. Constructing it
    inside every forward pass adds Python and host-to-device overhead without
    changing the model, so it is cached as a non-trainable constant tensor.
    """
    device = torch.device('cpu') if device is None else torch.device(device)
    dtype = torch.float32 if dtype is None else dtype
    key = (int(n), device.type, device.index, dtype)
    cached = _DCT_MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    # Keep cached constants usable by autograd even when their first creation
    # happens inside torch.inference_mode().
    with torch.inference_mode(False):
        dct_m = torch.empty(n, n, dtype=torch.float32)
        for k in range(n):
            for i in range(n):
                dct_m[k, i] = math.cos(math.pi / n * (i + 0.5) * k)
        dct_m[0, :] *= 1.0 / math.sqrt(2.0)
        dct_m *= math.sqrt(2.0 / n)
        dct_m = dct_m.to(device=device, dtype=dtype)
    _DCT_MATRIX_CACHE[key] = dct_m
    return dct_m


def dct2_2d(x):
    h, w = x.shape[-2:]
    c_h = get_dct_matrix(h, x.device, x.dtype)
    c_w = get_dct_matrix(w, x.device, x.dtype)
    out = torch.matmul(c_h.unsqueeze(0).unsqueeze(0), x)
    out = torch.matmul(out, c_w.t().unsqueeze(0).unsqueeze(0))
    return out


def idct2_2d(x):
    h, w = x.shape[-2:]
    c_h = get_dct_matrix(h, x.device, x.dtype)
    c_w = get_dct_matrix(w, x.device, x.dtype)
    out = torch.matmul(c_h.t().unsqueeze(0).unsqueeze(0), x)
    out = torch.matmul(out, c_w.unsqueeze(0).unsqueeze(0))
    return out


class _Mamba2DBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.vmamba = VMamba2DWrapper(channels=channels)

    def forward(self, x):
        return self.vmamba(x)


class _Transformer2DBranch(nn.Module):
    def __init__(self, channels, num_heads=4, mlp_dim=304, dropout=0.0):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError('2D sinusoidal positions require channels divisible by four')
        self.channels = channels
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, channels),
            nn.Dropout(dropout),
        )

    def _position_embedding(self, height, width, device, dtype):
        cache = getattr(self, '_position_embedding_cache', None)
        if cache is None:
            cache = {}
            self._position_embedding_cache = cache
        key = (int(height), int(width), device.type, device.index, dtype)
        cached = cache.get(key)
        if cached is not None:
            return cached

        with torch.inference_mode(False):
            quarter = self.channels // 4
            frequency = torch.arange(quarter, device=device, dtype=torch.float32)
            frequency = 1.0 / (10000.0 ** (frequency / max(quarter - 1, 1)))
            rows = torch.arange(height, device=device, dtype=torch.float32)
            cols = torch.arange(width, device=device, dtype=torch.float32)
            row_phase = rows[:, None] * frequency[None, :]
            col_phase = cols[:, None] * frequency[None, :]
            row_pos = torch.cat((row_phase.sin(), row_phase.cos()), dim=-1)
            col_pos = torch.cat((col_phase.sin(), col_phase.cos()), dim=-1)
            row_pos = row_pos[:, None, :].expand(height, width, -1)
            col_pos = col_pos[None, :, :].expand(height, width, -1)
            pos = torch.cat((row_pos, col_pos), dim=-1).reshape(
                1, height * width, self.channels
            ).to(dtype=dtype)
        cache[key] = pos
        return pos

    def forward(self, x):
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self._position_embedding(height, width, x.device, x.dtype)
        normalized = self.norm1(tokens)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class HeterogeneousDCTMambaBlock(nn.Module):
    """H-DMB: Heterogeneous DCT-Mamba Block."""
    def __init__(self, channels=64, window_size=4, mask_ratio=0.5, mixer_type='mamba'):
        super().__init__()
        if mixer_type not in {'mamba', 'transformer'}:
            raise ValueError(f'Unsupported HDMB mixer: {mixer_type}')
        self.channels = channels
        self.window_size = window_size
        self.mask_ratio = mask_ratio
        self.mixer_type = mixer_type
        branch = _Mamba2DBranch if mixer_type == 'mamba' else _Transformer2DBranch

        self.magnitude_norm = nn.InstanceNorm2d(1)
        self.phase_norm = nn.InstanceNorm2d(1)

        setattr(self, f'global_{mixer_type}', branch(channels))
        setattr(self, f'local_{mixer_type}', branch(channels))

        self.cross_window_refiner = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    @property
    def global_mixer(self):
        return getattr(self, f'global_{self.mixer_type}')

    @property
    def local_mixer(self):
        return getattr(self, f'local_{self.mixer_type}')

    def generate_masks(self, h, w, device, dtype=torch.float32):
        # The masks are deterministic constants for a given feature shape. Cache
        # them to avoid two GPU allocations and slice assignments per forward.
        # inference_mode(False) keeps a mask first created during inference safe
        # to reuse later in an autograd-enabled training pass.
        device = torch.device(device)
        key = (
            int(h), int(w), device.type, device.index, dtype, float(self.mask_ratio)
        )
        cached = _DCT_MASK_CACHE.get(key)
        if cached is not None:
            return cached

        with torch.inference_mode(False):
            mask_low = torch.zeros(h, w, device=device, dtype=dtype)
            mask_high = torch.ones(h, w, device=device, dtype=dtype)

            h_thresh = int(h * self.mask_ratio)
            w_thresh = int(w * self.mask_ratio)

            mask_low[:h_thresh, :w_thresh] = 1.0
            mask_high[:h_thresh, :w_thresh] = 0.0

            masks = (
                mask_low.view(1, 1, h, w),
                mask_high.view(1, 1, h, w),
            )
        _DCT_MASK_CACHE[key] = masks
        return masks

    def forward(self, f_s, f_d):
        f_s = f_s.permute(0, 3, 1, 2).contiguous()
        f_d = f_d.permute(0, 3, 1, 2).contiguous()
        bsz, channels, h, w = f_s.shape

        fft_s = torch.fft.fft2(f_s)
        magnitude_s = torch.abs(fft_s)
        w_amp = torch.log1p(magnitude_s.mean(dim=1, keepdim=True))
        w_amp = torch.sigmoid(self.magnitude_norm(w_amp))
        f_s_mod = f_s * w_amp

        fft_d = torch.fft.fft2(f_d)
        phase_only = fft_d / (torch.abs(fft_d) + 1e-8)
        spatial_phase = torch.fft.ifft2(phase_only).real
        w_phase = spatial_phase.mean(dim=1, keepdim=True)
        w_phase = torch.sigmoid(self.phase_norm(w_phase))
        f_d_mod = f_d + f_d * w_phase

        d_s = dct2_2d(f_s_mod)
        d_d = dct2_2d(f_d_mod)

        m_low, m_high = self.generate_masks(h, w, f_s.device, f_s.dtype)
        d_s_filtered = d_s * m_low
        d_d_filtered = d_d * m_high

        x_s = idct2_2d(d_s_filtered)
        x_d = idct2_2d(d_d_filtered)

        z_s = self.global_mixer(x_s)

        pad_h = (self.window_size - h % self.window_size) % self.window_size
        pad_w = (self.window_size - w % self.window_size) % self.window_size
        x_d_padded = F.pad(x_d, (0, pad_w, 0, pad_h))
        hp, wp = x_d_padded.shape[2:]

        x_windows = x_d_padded.view(
            bsz,
            channels,
            hp // self.window_size,
            self.window_size,
            wp // self.window_size,
            self.window_size,
        )
        x_windows = x_windows.permute(0, 2, 4, 1, 3, 5).contiguous().view(
            -1, channels, self.window_size, self.window_size
        )

        z_windows = self.local_mixer(x_windows)

        z_windows = z_windows.view(
            bsz,
            hp // self.window_size,
            wp // self.window_size,
            channels,
            self.window_size,
            self.window_size,
        )
        z_d = z_windows.permute(0, 3, 1, 4, 2, 5).contiguous().view(bsz, channels, hp, wp)

        if pad_h > 0 or pad_w > 0:
            z_d = z_d[:, :, :h, :w]

        z_d = self.cross_window_refiner(z_d) + z_d

        concat_feat = torch.cat([z_s, z_d], dim=1)
        alpha_map = self.fusion_conv(concat_feat)
        z_out = z_s + alpha_map * z_d

        return z_out.permute(0, 2, 3, 1).contiguous()
