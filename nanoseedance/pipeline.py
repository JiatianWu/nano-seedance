"""Ties the components into one object: text -> latents -> pixels + audio.

Everything downstream (sampling, SFT, GRPO, DPO) goes through this so that
classifier-free guidance, latent scaling and decoding are defined exactly once.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import Config
from .data.audio import MelSpectrogram
from .data.text import CaptionTokenizer
from .flow import FlowMatching, sample_ode, sample_sde
from .models import AudioVAE, DualBranchDiT, TinyTextEncoder, VideoVAE
from .utils import load_checkpoint, resolve_device


class SeedancePipeline(nn.Module):
    def __init__(self, cfg: Config, video_vae: VideoVAE, audio_vae: Optional[AudioVAE],
                 dit: DualBranchDiT, text_encoder: nn.Module,
                 tokenizer: Optional[CaptionTokenizer]):
        super().__init__()
        self.cfg = cfg
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.dit = dit
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.flow = FlowMatching(cfg.flow, num_tokens=cfg.num_video_tokens)
        self.mel = MelSpectrogram(
            sample_rate=cfg.data.sample_rate, n_fft=cfg.data.n_fft,
            hop_length=cfg.data.hop_length, n_mels=cfg.data.n_mels,
            n_frames=cfg.data.mel_frames,
        )

    # -- construction -------------------------------------------------------
    @classmethod
    def build(cls, cfg: Config) -> "SeedancePipeline":
        video_vae = VideoVAE(cfg.video_vae)
        audio_vae = AudioVAE(cfg.audio_vae, cfg.data.n_mels) if cfg.dit.enable_audio else None

        if cfg.dit.text_encoder == "hf":
            from .models import HFTextEncoder

            text_encoder: nn.Module = HFTextEncoder(cfg.dit.hf_text_model, cfg.dit.max_text_len)
            tokenizer = None
            text_dim = text_encoder.dim
        else:
            tokenizer = CaptionTokenizer(cfg.dit.max_text_len)
            text_encoder = TinyTextEncoder(
                len(tokenizer), cfg.dit.text_dim, cfg.dit.text_depth,
                cfg.dit.text_heads, cfg.dit.max_text_len, cfg.dit.rope_theta,
            )
            text_dim = cfg.dit.text_dim

        dit = DualBranchDiT(cfg.dit, cfg.video_vae.latent_channels,
                            cfg.audio_vae.latent_channels, text_dim=text_dim)
        return cls(cfg, video_vae, audio_vae, dit, text_encoder, tokenizer)

    @property
    def uses_audio(self) -> bool:
        return self.cfg.dit.enable_audio and self.audio_vae is not None

    # -- conditioning -------------------------------------------------------
    def encode_prompts(self, captions: Sequence[str], device: torch.device):
        if self.tokenizer is None:  # HF encoder tokenizes internally
            return self.text_encoder(captions, device=device)
        ids, mask = self.tokenizer.batch_encode(captions, device=device)
        return self.text_encoder(ids, mask)

    def velocity(
        self,
        z_v: torch.Tensor,
        z_a: Optional[torch.Tensor],
        t: torch.Tensor,
        text: torch.Tensor,
        pooled: torch.Tensor,
        mask: Optional[torch.Tensor],
        guidance_scale: float = 1.0,
        t_audio: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Velocity prediction with classifier-free guidance.

        Conditional and unconditional passes are batched together so guided
        sampling costs one forward call, not two.
        """
        if guidance_scale == 1.0:
            return self.dit(z_v, t, text, pooled, mask, audio_latent=z_a, t_audio=t_audio)

        batch = z_v.shape[0]
        z_v2 = torch.cat([z_v, z_v], dim=0)
        z_a2 = torch.cat([z_a, z_a], dim=0) if z_a is not None else None
        t2 = torch.cat([t, t], dim=0)
        ta2 = torch.cat([t_audio, t_audio], dim=0) if t_audio is not None else None
        text2 = torch.cat([text, text], dim=0)
        pooled2 = torch.cat([pooled, pooled], dim=0)
        mask2 = torch.cat([mask, mask], dim=0) if mask is not None else None
        drop = torch.cat([
            torch.zeros(batch, dtype=torch.bool, device=z_v.device),
            torch.ones(batch, dtype=torch.bool, device=z_v.device),
        ])

        v_v, v_a = self.dit(z_v2, t2, text2, pooled2, mask2,
                            audio_latent=z_a2, t_audio=ta2, drop_cond=drop)
        v_v_c, v_v_u = v_v.chunk(2, dim=0)
        guided_v = v_v_u + guidance_scale * (v_v_c - v_v_u)
        guided_a = None
        if v_a is not None:
            v_a_c, v_a_u = v_a.chunk(2, dim=0)
            guided_a = v_a_u + guidance_scale * (v_a_c - v_a_u)
        return guided_v, guided_a

    def make_velocity_fn(self, text, pooled, mask, guidance_scale: float):
        def velocity_fn(z_v, z_a, t):
            return self.velocity(z_v, z_a, t, text, pooled, mask, guidance_scale)
        return velocity_fn

    # -- generation ---------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        captions: Sequence[str],
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        decode: bool = True,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        device = device or next(self.dit.parameters()).device
        steps = num_steps or cfg.flow.num_sampling_steps
        guidance = cfg.flow.guidance_scale if guidance_scale is None else guidance_scale

        text, pooled, mask = self.encode_prompts(captions, device)
        batch = len(captions)
        video_shape = (batch,) + cfg.video_latent_shape
        audio_shape = (batch,) + cfg.audio_latent_shape if self.uses_audio else None

        z_v, z_a = sample_ode(
            self.make_velocity_fn(text, pooled, mask, guidance),
            video_shape, audio_shape, self.flow, steps, device, generator,
        )
        out: Dict[str, Any] = {"video_latent": z_v, "audio_latent": z_a,
                               "captions": list(captions)}
        if decode:
            out.update(self.decode_latents(z_v, z_a))
        return out

    @torch.no_grad()
    def decode_latents(self, z_v: torch.Tensor,
                       z_a: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {"video": self.video_vae.decode_from_latent(z_v)}
        if z_a is not None and self.audio_vae is not None:
            mel = self.audio_vae.decode_from_latent(z_a)
            out["mel"] = mel
            out["waveform"] = self.mel.inverse(
                mel.float().cpu(), num_samples=self.cfg.data.audio_samples, n_iter=24
            )
        return out

    @torch.no_grad()
    def rollout(self, captions: Sequence[str], num_steps: int, guidance_scale: float,
                eta: float, device: torch.device,
                generator: Optional[torch.Generator] = None):
        """Stochastic sampling that returns the full trajectory (for RL)."""
        cfg = self.cfg
        text, pooled, mask = self.encode_prompts(captions, device)
        batch = len(captions)
        video_shape = (batch,) + cfg.video_latent_shape
        audio_shape = (batch,) + cfg.audio_latent_shape if self.uses_audio else None
        z_v, z_a, traj = sample_sde(
            self.make_velocity_fn(text, pooled, mask, guidance_scale),
            video_shape, audio_shape, self.flow, num_steps, device, eta, generator,
        )
        return z_v, z_a, traj

    # -- encoding real data to latents (DiT training) -----------------------
    @torch.no_grad()
    def encode_batch(self, batch: Dict[str, Any], device: torch.device
                     ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        z_v = self.video_vae.encode_to_latent(batch["video"].to(device))
        z_a = None
        if self.uses_audio:
            z_a = self.audio_vae.encode_to_latent(batch["mel"].to(device))
        return z_v, z_a

    # -- persistence --------------------------------------------------------
    def generator_module(self) -> nn.Module:
        """A view over the trainable generator (DiT + text encoder).

        Used for EMA and optimisation so both are always saved, loaded and
        averaged together; keys line up with :meth:`dit_state`.
        """
        module = nn.ModuleDict()
        module["dit"] = self.dit
        module["text_encoder"] = self.text_encoder
        return module

    def dit_state(self) -> Dict[str, torch.Tensor]:
        return dict(self.generator_module().state_dict())

    def load_dit_state(self, state: Dict[str, torch.Tensor], strict: bool = True) -> None:
        dit_state = {k[len("dit."):]: v for k, v in state.items() if k.startswith("dit.")}
        text_state = {k[len("text_encoder."):]: v for k, v in state.items()
                      if k.startswith("text_encoder.")}
        self.dit.load_state_dict(dit_state, strict=strict)
        if text_state:
            self.text_encoder.load_state_dict(text_state, strict=strict)

    def load_vaes(self, path: str, device: str = "cpu") -> None:
        ckpt = load_checkpoint(path, map_location=device)
        self.video_vae.load_state_dict(ckpt["video_vae"])
        if self.audio_vae is not None and ckpt.get("audio_vae") is not None:
            self.audio_vae.load_state_dict(ckpt["audio_vae"])

    def freeze_vaes(self) -> None:
        self.video_vae.requires_grad_(False).eval()
        if self.audio_vae is not None:
            self.audio_vae.requires_grad_(False).eval()


def load_pipeline(cfg: Config, vae_ckpt: str, dit_ckpt: Optional[str] = None,
                  device: Optional[torch.device] = None,
                  use_ema: bool = True) -> SeedancePipeline:
    """Build a pipeline and populate it from trained checkpoints."""
    device = device or resolve_device(cfg.train.device)
    pipe = SeedancePipeline.build(cfg)
    pipe.load_vaes(vae_ckpt)

    if dit_ckpt:
        ckpt = load_checkpoint(dit_ckpt)
        state = ckpt.get("ema") if (use_ema and ckpt.get("ema")) else ckpt["model"]
        pipe.load_dit_state(state, strict=True)
    pipe.to(device)
    pipe.freeze_vaes()
    pipe.eval()
    return pipe
