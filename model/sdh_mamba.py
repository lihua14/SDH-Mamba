import torch
from einops import rearrange
from torch import nn

from .h_dmb import HeterogeneousDCTMambaBlock
from .ssd import SelectiveStructureDetailDecoupling


_DATASET_FALLBACK_SPECS = {
    "sa": (30, 16),
    "pu": (103, 9),
    "whulk": (270, 9),
    "hrl": (176, 14),
    "flt": (80, 10),
    "ksc": (176, 13),
    "ip": (200, 16),
    "hus": (144, 15),
    "houston": (144, 15),
    "muufl": (64, 11),
    "trento": (63, 6),
    "botswana": (145, 14),
    "whuhh": (270, 22),
    "pc": (102, 9),
    "whuhc": (274, 16),
}


class SDHMamba(nn.Module):
    """Structure-Detail Decoupled Heterogeneous DCT-Mamba network."""
    def __init__(
        self,
        n_bands: int,
        num_classes: int,
        in_channels: int = 1,
        channels: int = 64,
        window_size: int = 4,
        use_ssd: bool = True,
        use_hdmb: bool = True,
        mixer_type: str = 'mamba',
    ):
        super().__init__()
        if n_bands < 3:
            raise ValueError(f"n_bands must be >= 3, got {n_bands}")
        self.use_ssd = bool(use_ssd)
        self.use_hdmb = bool(use_hdmb)

        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
        )

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=8 * (n_bands - 2), out_channels=channels, kernel_size=(3, 3), groups=8),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        if self.use_ssd:
            self.ssd = SelectiveStructureDetailDecoupling(channels=channels)
        else:
            self.ssd = None

        if self.use_hdmb:
            self.hdmb = HeterogeneousDCTMambaBlock(
                channels=channels, window_size=window_size, mixer_type=mixer_type
            )
        else:
            self.hdmb = None

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 2, num_classes),
        )

    def forward(self, x, mask=None):
        x = self.conv3d_features(x)
        x = rearrange(x, "b c h w y -> b (c h) w y")
        x = self.conv2d_features(x)

        x_bhwc = x.permute(0, 2, 3, 1).contiguous()
        if self.use_ssd:
            f_s, f_d = self.ssd(x_bhwc)
        else:
            f_s, f_d = x_bhwc, x_bhwc

        if self.use_hdmb:
            x_bhwc = self.hdmb(f_s, f_d)
        elif self.use_ssd:
            x_bhwc = 0.5 * (f_s + f_d)

        x = x_bhwc.permute(0, 3, 1, 2).contiguous()

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Accept parameter names from the original experiment checkpoints."""
        aliases = (
            ('dctmamba.', 'hdmb.'),
            ('hdmb.norm_gsp.', 'hdmb.magnitude_norm.'),
            ('hdmb.norm_dsp.', 'hdmb.phase_norm.'),
            ('ssd.avg_branch.', 'ssd.scs.'),
            ('ssd.max_branch.', 'ssd.dsv.'),
            ('ssd.scs.alpha_c', 'ssd.scs.alpha_s'),
            ('ssd.dsv.alpha_c', 'ssd.dsv.alpha_d'),
            ('ssd.lambda_g', 'ssd.lambda_s'),
        )
        for key in list(state_dict):
            renamed = key
            for old, new in aliases:
                if renamed.startswith(prefix + old):
                    renamed = prefix + new + renamed[len(prefix + old):]
            if renamed != key:
                value = state_dict.pop(key)
                if renamed in state_dict:
                    error_msgs.append(f'Conflicting checkpoint keys: {key} and {renamed}.')
                else:
                    state_dict[renamed] = value
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)


def _infer_specs_from_dataset(dataset: str):
    key = str(dataset).strip().lower()
    if key not in _DATASET_FALLBACK_SPECS:
        supported = sorted(_DATASET_FALLBACK_SPECS.keys())
        raise KeyError(
            f"Cannot infer (n_bands, num_classes) for dataset '{dataset}'. "
            f"Supported fallback datasets: {supported}."
        )
    return _DATASET_FALLBACK_SPECS[key]


def sdh_mamba(
    dataset, patch_size, n_bands=None, num_classes=None, use_ssd=True, use_hdmb=True,
    mixer_type='mamba',
):
    del patch_size
    if n_bands is None or num_classes is None:
        fb_bands, fb_classes = _infer_specs_from_dataset(dataset)
        n_bands = fb_bands if n_bands is None else n_bands
        num_classes = fb_classes if num_classes is None else num_classes
    return SDHMamba(
        n_bands=int(n_bands),
        num_classes=int(num_classes),
        use_ssd=use_ssd,
        use_hdmb=use_hdmb,
        mixer_type=mixer_type,
    )
