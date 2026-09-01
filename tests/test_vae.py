"""Autoencoder shape and causality guarantees."""

from __future__ import annotations

import pytest
import torch

from nanoseedance.config import Config
from nanoseedance.models.audio_vae import AudioVAE
from nanoseedance.models.vae import VideoVAE


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/smoke.yaml")


def test_video_latent_shape(cfg):
    vae = VideoVAE(cfg.video_vae).eval()
    x = torch.randn(2, 3, cfg.data.num_frames, cfg.data.resolution, cfg.data.resolution)
    with torch.no_grad():
        z = vae.encode(x).mean
        recon = vae.decode(z)
    assert tuple(z.shape[1:]) == cfg.video_latent_shape
    assert recon.shape == x.shape


def test_audio_latent_shape(cfg):
    vae = AudioVAE(cfg.audio_vae, cfg.data.n_mels).eval()
    mel = torch.randn(2, cfg.data.n_mels, cfg.data.mel_frames)
    with torch.no_grad():
        z = vae.encode(mel).mean
        recon = vae.decode(z)
    assert tuple(z.shape[1:]) == cfg.audio_latent_shape
    assert recon.shape == mel.shape


def test_temporal_causality(cfg):
    """Perturbing input frame f must not move any latent frame that precedes it.

    This is the property that makes chunked encoding of long clips valid; it
    breaks silently if any layer pools statistics over the whole time axis.
    """
    vae = VideoVAE(cfg.video_vae).eval()
    frames, size = cfg.data.num_frames, cfg.data.resolution
    x = torch.randn(1, 3, frames, size, size)
    with torch.no_grad():
        base = vae.encode(x).mean

    ratio = cfg.video_vae.temporal_compression
    for cut in range(1, frames):
        perturbed = x.clone()
        perturbed[:, :, cut:] = torch.randn_like(perturbed[:, :, cut:])
        with torch.no_grad():
            delta = (vae.encode(perturbed).mean - base).abs().amax(dim=(0, 1, 3, 4))

        # Latent frame i covers input frames up to 1 + i*ratio - 1.
        first_affected = max(0, (cut - 1) // ratio + (1 if cut > 0 else 0))
        for i in range(first_affected):
            assert float(delta[i]) == 0.0, (
                "latent frame %d changed when input frames >=%d were perturbed" % (i, cut)
            )


def test_image_matches_first_latent_frame(cfg):
    """A single frame encodes identically whether or not a video follows it."""
    vae = VideoVAE(cfg.video_vae).eval()
    x = torch.randn(1, 3, cfg.data.num_frames, cfg.data.resolution, cfg.data.resolution)
    with torch.no_grad():
        clip_latent = vae.encode(x).mean
        image_latent = vae.encode(x[:, :, :1]).mean
    assert image_latent.shape[2] == 1
    assert torch.allclose(image_latent, clip_latent[:, :, :1], atol=1e-5)


def test_latent_scale_calibration(cfg):
    vae = VideoVAE(cfg.video_vae).eval()
    x = torch.randn(4, 3, cfg.data.num_frames, cfg.data.resolution, cfg.data.resolution)
    scale = vae.calibrate_latent_scale(x)
    assert scale > 0
    with torch.no_grad():
        z = vae.encode_to_latent(x)
    assert 0.3 < float(z.std()) < 3.0
