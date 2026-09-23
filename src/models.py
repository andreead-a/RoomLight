"""Convolutional VAE for equirectangular HDR panoramas.

The panoramas have four channels: HDR radiance and depth, stored as
log1p(radiance) / log(depth) and z-scored with the statistics in `dataset.py`.

Three things make the network panorama-aware:

* every convolution uses panorama padding (`circular_pad_2d`): wrap-around in
  longitude, flip-and-roll across the poles, so the output is seamless;
* the encoder input and the decoder input receive cos/sin of the polar angle as two
  extra channels (`LatitudeCoords`), so the convolutions know which latitude they are
  looking at;
* the global latent modulates every decoder normalisation layer (`AdaGN`) instead of
  entering only through the reshape onto the coarsest feature map.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf


def circular_pad_2d(x, pad):
    """Panorama padding for a (..., H, W) equirectangular map."""
    w = x.shape[-1]
    top = torch.flip(torch.roll(x[..., :pad, :], w // 2, -1), dims=[-2])
    bottom = torch.flip(torch.roll(x[..., -pad:, :], w // 2, -1), dims=[-2])
    x = torch.cat([top, x, bottom], dim=-2)
    return torch.cat([x[..., -pad:], x, x[..., :pad]], dim=-1)


class CircularConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.pad = padding
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=0)

    def forward(self, x):
        return self.conv(circular_pad_2d(x, self.pad))


def conv3x3(in_ch, out_ch, stride=1):
    return CircularConv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def group_norm(channels, num_groups):
    return nn.GroupNorm(math.gcd(channels, num_groups), channels, eps=1e-6)


class LatitudeCoords(nn.Module):
    """Appends cos(theta) and sin(theta) of each row's polar angle as two channels."""

    n_channels = 2

    def forward(self, x):
        h = x.shape[-2]
        theta = (torch.arange(h, device=x.device, dtype=x.dtype) + 0.5) * (math.pi / h)
        coords = torch.stack([theta.cos(), theta.sin()]).view(1, 2, h, 1)
        return torch.cat([x, coords.expand(x.shape[0], -1, -1, x.shape[-1])], dim=1)


class AdaGN(nn.Module):
    """GroupNorm whose scale and shift are predicted from the latent (zero-initialised)."""

    takes_latent = True

    def __init__(self, channels, latent_dim, num_groups):
        super().__init__()
        self.norm = nn.GroupNorm(math.gcd(channels, num_groups), channels, eps=1e-6,
                                 affine=False)
        self.to_scale_shift = nn.Linear(latent_dim, 2 * channels)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x, z):
        scale, shift = self.to_scale_shift(z).chunk(2, dim=1)
        return self.norm(x) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ResBlock(nn.Module):
    """Pre-activation residual block. With `latent_dim` the norms become `AdaGN`."""

    def __init__(self, channels, num_groups, latent_dim=None):
        super().__init__()
        self.takes_latent = latent_dim is not None
        if self.takes_latent:
            self.norm1 = AdaGN(channels, latent_dim, num_groups)
            self.norm2 = AdaGN(channels, latent_dim, num_groups)
        else:
            self.norm1 = group_norm(channels, num_groups)
            self.norm2 = group_norm(channels, num_groups)
        self.conv1 = conv3x3(channels, channels)
        self.conv2 = conv3x3(channels, channels)

    def _norm(self, norm, h, z):
        return norm(h, z) if self.takes_latent else norm(h)

    def forward(self, x, z=None):
        h = self.conv1(F.silu(self._norm(self.norm1, x, z)))
        h = self.conv2(F.silu(self._norm(self.norm2, h, z)))
        return x + h


class Upsample(nn.Module):
    """2x nearest-neighbour upsampling followed by a 3x3 convolution."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = conv3x3(in_ch, out_ch)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    """channels x H x W -> z_channels x H/2^L x W/2^L, with L = len(ch_mult)."""

    def __init__(self, channels, z_channels, ch, ch_mult, num_res_blocks, num_groups):
        super().__init__()
        layers = [LatitudeCoords(), conv3x3(channels + LatitudeCoords.n_channels, ch)]
        width = ch
        for mult in ch_mult:
            layers.append(conv3x3(width, ch * mult, stride=2))
            width = ch * mult
            layers += [ResBlock(width, num_groups) for _ in range(num_res_blocks)]
        layers += [group_norm(width, num_groups), nn.SiLU(), conv3x3(width, z_channels)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """Mirror of `Encoder`; the latent `z` conditions every normalisation layer."""

    def __init__(self, channels, z_channels, ch, ch_mult, num_res_blocks, num_groups,
                 latent_dim):
        super().__init__()
        widths = [ch * mult for mult in ch_mult]
        width = widths[-1]
        layers = [LatitudeCoords(), conv3x3(z_channels + LatitudeCoords.n_channels, width)]
        for out_width in reversed([ch] + widths[:-1]):
            layers.append(Upsample(width, out_width))
            width = out_width
            layers += [ResBlock(width, num_groups, latent_dim) for _ in range(num_res_blocks)]
        layers += [AdaGN(width, latent_dim, num_groups), nn.SiLU(), conv3x3(width, channels)]
        self.model = nn.ModuleList(layers)

    def forward(self, h, z):
        for layer in self.model:
            h = layer(h, z) if getattr(layer, "takes_latent", False) else layer(h)
        return h


class DiagonalGaussian:
    """Posterior q(z | x) with diagonal covariance, parameterised by (mean, logvar)."""

    def __init__(self, moments):
        self.mean, logvar = moments.chunk(2, dim=1)
        self.logvar = logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


class ConvVAE(nn.Module):
    """VAE with a single global latent vector per panorama.

    Args:
        resolution: (H, W) of the panoramas; W = 2H for equirectangular maps.
        latent_dim: length of the global latent vector.
        ch: base channel width; level i has `ch * ch_mult[i]` channels.
        ch_mult: one entry per 2x downsampling level.
        num_res_blocks: residual blocks per level, in both encoder and decoder.
        z_channels: width of the coarsest feature map that is projected to the latent.
        norm_groups: groups of every GroupNorm / AdaGN.
        channels: input and output channels (RGB radiance + depth).
    """

    def __init__(self, resolution=(128, 256), latent_dim=512, ch=64, ch_mult=(1, 2, 4, 4),
                 num_res_blocks=1, z_channels=32, norm_groups=8, channels=4):
        super().__init__()
        height, width = tuple(resolution)
        stride = 2 ** len(ch_mult)
        assert height % stride == 0 and width % stride == 0, (
            f"resolution {(height, width)} must be divisible by 2**len(ch_mult) = {stride}")
        self.latent_dim = latent_dim
        self.z_shape = (z_channels, height // stride, width // stride)

        self.encoder = Encoder(channels, z_channels, ch, ch_mult, num_res_blocks, norm_groups)
        self.decoder = Decoder(channels, z_channels, ch, ch_mult, num_res_blocks, norm_groups,
                               latent_dim)
        flat = math.prod(self.z_shape)
        self.quant = nn.Linear(flat, 2 * latent_dim)
        self.post_quant = nn.Linear(latent_dim, flat)

    def encode(self, x):
        return DiagonalGaussian(self.quant(self.encoder(x).flatten(1)))

    def decode(self, z):
        return self.decoder(self.post_quant(z).view(-1, *self.z_shape), z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        return self.decode(z), posterior


def load_model(run, device, ckpt_name="last_epoch.pth"):
    """Load a trained model from a run directory (or a checkpoint inside one).

    The run directory holds the `config.yaml` written by `train.py`, so the architecture
    is read from there rather than assumed. The model is returned frozen and in eval mode.
    """
    run = Path(run)
    ckpt = run / ckpt_name if run.is_dir() else run
    cfg = OmegaConf.load(ckpt.parent / "config.yaml")
    model = ConvVAE(**OmegaConf.to_container(cfg.model, resolve=True))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state.get("state_dict", state))
    return model.to(device).eval().requires_grad_(False)
