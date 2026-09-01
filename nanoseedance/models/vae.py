"""Temporally-causal 3D video VAE.

Compresses (3, T, H, W) video into (C, 1 + (T-1)/rt, H/rs, W/rs) latents.

Two properties matter and are both enforced here:

1. **Causality.** Latent frame i depends only on input frames <= its own
   receptive field in the past, never on future frames. That is what lets the
   same model encode a single image (T=1) and a video with one set of weights,
   and it is what makes chunked/streaming encoding of long clips possible.
   Note that ordinary ``GroupNorm`` would break this silently by pooling
   statistics over the whole time axis, so normalisation is applied per frame.

2. **Frame-0 anchoring.** With T = 1 + k*rt frames the encoder emits 1 + k
   latent frames, where latent frame 0 corresponds to input frame 0 alone.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import VideoVAEConfig


class CausalConv3d(nn.Module):
    """3D conv with symmetric spatial padding and left-only temporal padding."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Tuple[int, int, int] = (3, 3, 3),
                 stride: Tuple[int, int, int] = (1, 1, 1)):
        super().__init__()
        kt, kh, kw = kernel_size
        self.time_pad = kt - 1
        self.spatial_pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Replicate the first frame backwards in time so that frame 0 is
        # well-defined without seeing anything after it.
        if self.time_pad:
            first = x[:, :, :1].expand(-1, -1, self.time_pad, -1, -1)
            x = torch.cat([first, x], dim=2)
        x = F.pad(x, self.spatial_pad + (0, 0), mode="constant", value=0.0)
        return self.conv(x)


class CausalGroupNorm(nn.Module):
    """GroupNorm applied independently per frame.

    Folding time into the batch dimension keeps each frame's statistics
    independent; a plain GroupNorm over (C, T, H, W) would leak information
    backwards in time and destroy causality.

    Group count is safe to leave high here because a group still spans
    ``(C/G) * H * W`` values. The 1D audio equivalent has no spatial extent and
    needs an explicit floor -- see ``audio_vae.CausalGroupNorm1d``.
    """

    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(min(num_groups, channels), channels, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)


class ResBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = CausalGroupNorm(in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels)
        self.norm2 = CausalGroupNorm(out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
        nn.init.zeros_(self.conv2.conv.weight)
        nn.init.zeros_(self.conv2.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class CausalDownsample(nn.Module):
    """Halve H and W, and optionally T, preserving the 1 + k frame structure."""

    def __init__(self, in_channels: int, out_channels: int, temporal: bool):
        super().__init__()
        self.temporal = temporal
        stride = (2 if temporal else 1, 2, 2)
        self.conv = CausalConv3d(in_channels, out_channels, (3, 3, 3), stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # With left pad 2, kernel 3, stride 2 and T = 1 + 2m: out = 1 + m.
        return self.conv(x)


class CausalUpsample(nn.Module):
    """Inverse of :class:`CausalDownsample`."""

    def __init__(self, in_channels: int, out_channels: int, temporal: bool):
        super().__init__()
        self.temporal = temporal
        self.conv = CausalConv3d(in_channels, out_channels, (3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = x.view(b, t, c, h * 2, w * 2).permute(0, 2, 1, 3, 4)
        if self.temporal and t > 1:
            # Frame 0 stays a single frame; every later latent frame expands to
            # two, mapping 1 + m -> 1 + 2m.
            head, tail = x[:, :, :1], x[:, :, 1:]
            tail = tail.repeat_interleave(2, dim=2)
            x = torch.cat([head, tail], dim=2)
        return self.conv(x)


class DiagonalGaussian:
    """Latent posterior with a diagonal covariance."""

    def __init__(self, parameters: torch.Tensor):
        self.mean, self.logvar = parameters.chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        noise = torch.randn(self.mean.shape, device=self.mean.device,
                            dtype=self.mean.dtype, generator=generator)
        return self.mean + self.std * noise

    def kl(self) -> torch.Tensor:
        return 0.5 * torch.sum(
            self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar,
            dim=list(range(1, self.mean.dim())),
        )


class Encoder(nn.Module):
    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        channels = cfg.channels_per_stage()
        self.conv_in = CausalConv3d(cfg.in_channels, channels[0])

        blocks: List[nn.Module] = []
        for i in range(cfg.spatial_stages):
            blocks.append(ResBlock3d(channels[i], channels[i]))
            # Temporal compression happens on the earliest stages: shrinking T
            # first makes every later stage cheaper.
            blocks.append(CausalDownsample(channels[i], channels[i + 1],
                                           temporal=i < cfg.temporal_stages))
        self.blocks = nn.ModuleList(blocks)

        self.mid = ResBlock3d(channels[-1], channels[-1])
        self.norm_out = CausalGroupNorm(channels[-1])
        self.conv_out = CausalConv3d(channels[-1], 2 * cfg.latent_channels, (3, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for block in self.blocks:
            h = block(h)
        h = self.mid(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        channels = cfg.channels_per_stage()
        self.conv_in = CausalConv3d(cfg.latent_channels, channels[-1])
        self.mid = ResBlock3d(channels[-1], channels[-1])

        blocks: List[nn.Module] = []
        for i in reversed(range(cfg.spatial_stages)):
            blocks.append(ResBlock3d(channels[i + 1], channels[i + 1]))
            blocks.append(CausalUpsample(channels[i + 1], channels[i],
                                         temporal=i < cfg.temporal_stages))
        self.blocks = nn.ModuleList(blocks)

        self.norm_out = CausalGroupNorm(channels[0])
        self.conv_out = CausalConv3d(channels[0], cfg.in_channels, (3, 3, 3))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.mid(self.conv_in(z))
        for block in self.blocks:
            h = block(h)
        return self.conv_out(F.silu(self.norm_out(h)))


class VideoVAE(nn.Module):
    def __init__(self, cfg: VideoVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        # Set by calibrate_latent_scale so DiT inputs have ~unit variance.
        self.register_buffer("latent_scale", torch.tensor(1.0))

    def encode(self, video: torch.Tensor) -> DiagonalGaussian:
        return DiagonalGaussian(self.encoder(video))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, video: torch.Tensor,
                sample_posterior: bool = True) -> Tuple[torch.Tensor, DiagonalGaussian]:
        posterior = self.encode(video)
        z = posterior.sample() if sample_posterior else posterior.mean
        return self.decode(z), posterior

    # -- helpers used by the DiT stages ------------------------------------
    @torch.no_grad()
    def encode_to_latent(self, video: torch.Tensor, sample: bool = True) -> torch.Tensor:
        posterior = self.encode(video)
        z = posterior.sample() if sample else posterior.mean
        return z * self.latent_scale

    @torch.no_grad()
    def decode_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent / self.latent_scale).clamp(-1, 1)

    @torch.no_grad()
    def calibrate_latent_scale(self, videos: torch.Tensor) -> float:
        """Set latent_scale to 1/std so DiT sees roughly unit-variance inputs."""
        z = self.encode(videos).sample()
        scale = 1.0 / float(z.std().clamp(min=1e-4))
        self.latent_scale.fill_(scale)
        return scale


def vae_loss(recon: torch.Tensor, target: torch.Tensor, posterior: DiagonalGaussian,
             kl_weight: float) -> Tuple[torch.Tensor, dict]:
    """L1 + MSE reconstruction with a KL term.

    A GAN/LPIPS term is what buys perceptual sharpness in production VAEs; it
    is deliberately omitted here because it roughly triples the training code
    for no benefit at 32x32.
    """
    l1 = (recon - target).abs().mean()
    mse = F.mse_loss(recon, target)
    kl = posterior.kl().mean() / target[0].numel()
    loss = l1 + mse + kl_weight * kl
    return loss, {"l1": l1.detach(), "mse": mse.detach(), "kl": kl.detach()}
