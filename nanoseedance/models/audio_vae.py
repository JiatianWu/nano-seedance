"""Causal 1D autoencoder over log-mel spectrograms.

The audio branch needs its own latent space with its own temporal rate: mel
frames arrive far faster than video frames, and forcing the two branches onto
a shared clock is exactly the mismatch the cross-modal bridge exists to
resolve. Mel bands act as input channels; only time is compressed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import AudioVAEConfig
from .vae import DiagonalGaussian


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.time_pad = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_pad:
            x = F.pad(x, (self.time_pad, 0), mode="replicate")
        return self.conv(x)


class CausalGroupNorm1d(nn.Module):
    """Per-frame GroupNorm over channels only.

    Group count matters far more here than in the video VAE. There, a group
    spans ``(C/G) * H * W`` values, so even one channel per group leaves
    hundreds of numbers to estimate statistics from. Here a frame has no
    spatial extent, so a group spans exactly ``C/G`` values -- at the default
    32 groups on 64 channels that is *two* numbers, and normalising a pair to
    zero mean and unit variance maps it to +/-0.707 and discards essentially
    all of it. Reconstruction L1 was 8x worse before this was capped.
    """

    MIN_CHANNELS_PER_GROUP = 8

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        groups = max(1, min(num_groups, channels // self.MIN_CHANNELS_PER_GROUP))
        while channels % groups:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        x = x.permute(0, 2, 1).reshape(b * t, c, 1)
        x = self.norm(x)
        return x.view(b, t, c).permute(0, 2, 1)


class ResBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = CausalGroupNorm1d(in_channels)
        self.conv1 = CausalConv1d(in_channels, out_channels)
        self.norm2 = CausalGroupNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        nn.init.zeros_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class AudioVAE(nn.Module):
    """(B, n_mels, T) <-> (B, latent_channels, T / temporal_compression)."""

    def __init__(self, cfg: AudioVAEConfig, n_mels: int):
        super().__init__()
        self.cfg = cfg
        self.n_mels = n_mels
        ch = cfg.base_channels

        enc: List[nn.Module] = [CausalConv1d(n_mels, ch)]
        for _ in range(cfg.temporal_stages):
            enc.append(ResBlock1d(ch, ch))
            enc.append(CausalConv1d(ch, ch, kernel_size=4, stride=2))
        enc.append(ResBlock1d(ch, ch))
        enc.append(CausalGroupNorm1d(ch))
        self.encoder = nn.Sequential(*enc)
        self.enc_out = CausalConv1d(ch, 2 * cfg.latent_channels)

        dec: List[nn.Module] = [CausalConv1d(cfg.latent_channels, ch), ResBlock1d(ch, ch)]
        for _ in range(cfg.temporal_stages):
            dec.append(Upsample1d(ch))
            dec.append(ResBlock1d(ch, ch))
        dec.append(CausalGroupNorm1d(ch))
        self.decoder = nn.Sequential(*dec)
        self.dec_out = CausalConv1d(ch, n_mels)

        self.register_buffer("latent_scale", torch.tensor(1.0))

    def encode(self, mel: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.enc_out(F.silu(self.encoder(mel))))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.dec_out(F.silu(self.decoder(latent))))

    def forward(self, mel: torch.Tensor,
                sample_posterior: bool = True) -> Tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode(mel)
        z = posterior.sample() if sample_posterior else posterior.mean
        return self.decode(z), posterior

    @torch.no_grad()
    def encode_to_latent(self, mel: torch.Tensor, sample: bool = True) -> torch.Tensor:
        posterior = self.encode(mel)
        z = posterior.sample() if sample else posterior.mean
        return z * self.latent_scale

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent / self.latent_scale).clamp(-1, 1)

    @torch.no_grad()
    def calibrate_latent_scale(self, mel: torch.Tensor) -> float:
        z = self.encode(mel).sample()
        scale = 1.0 / float(z.std().clamp(min=1e-4))
        self.latent_scale.fill_(scale)
        return scale


class Upsample1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv1d(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)
