"""Dual-branch diffusion transformer.

Reconstruction of the architecture attributed to Seedance 2.0 (see README for
what is documented vs. inferred). Three ideas carry the design:

*Decoupled spatial/temporal attention.* Full 3D attention over T*H*W tokens is
quadratic in the whole volume. Alternating attention within a frame and across
time at a fixed spatial position costs O(T*S^2 + S*T^2) instead of O((T*S)^2),
which is the difference between tractable and not once clips get long.

*A cross-modal bridge.* Video and audio each get their own stack running at
their own token rate, and exchange information through bidirectional cross
attention every ``bridge_every`` blocks. Because it is cross attention rather
than a shared sequence, the two branches never have to agree on a clock -- the
mismatch between mel frames and video frames is absorbed by the attention map.

*Velocity prediction.* Both branches output a flow-matching velocity; see
``flow.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..config import DiTConfig
from .layers import (
    AdaLNModulation,
    CrossAttention,
    FeedForward,
    FinalLayer,
    SelfAttention,
    TimestepEmbedder,
    modulate,
    rope_params_nd,
)

RopePair = Tuple[torch.Tensor, torch.Tensor]


# --------------------------------------------------------------------------
# patchify helpers
# --------------------------------------------------------------------------
def patchify_video(x: torch.Tensor, pt: int, ph: int) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """(B, C, T, H, W) -> (B, T'*H'*W', C*pt*ph*ph) in row-major (t, h, w) order."""
    b, c, t, h, w = x.shape
    tp, hp, wp = t // pt, h // ph, w // ph
    x = x.view(b, c, tp, pt, hp, ph, wp, ph)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(b, tp * hp * wp, c * pt * ph * ph), (tp, hp, wp)


def unpatchify_video(tokens: torch.Tensor, grid: Tuple[int, int, int],
                     channels: int, pt: int, ph: int) -> torch.Tensor:
    b = tokens.shape[0]
    tp, hp, wp = grid
    x = tokens.view(b, tp, hp, wp, channels, pt, ph, ph)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(b, channels, tp * pt, hp * ph, wp * ph)


def patchify_audio(x: torch.Tensor, pt: int) -> Tuple[torch.Tensor, int]:
    """(B, C, T) -> (B, T/pt, C*pt)."""
    b, c, t = x.shape
    tp = t // pt
    x = x.view(b, c, tp, pt).permute(0, 2, 1, 3).contiguous()
    return x.view(b, tp, c * pt), tp


def unpatchify_audio(tokens: torch.Tensor, channels: int, pt: int) -> torch.Tensor:
    b, tp, _ = tokens.shape
    x = tokens.view(b, tp, channels, pt).permute(0, 2, 1, 3).contiguous()
    return x.view(b, channels, tp * pt)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
class VideoBlock(nn.Module):
    """Spatial attention -> temporal attention -> text cross-attention -> MLP."""

    def __init__(self, cfg: DiTConfig, context_dim: int):
        super().__init__()
        dim, heads = cfg.dim, cfg.num_heads
        self.norm_spatial = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_spatial = SelfAttention(dim, heads)
        self.norm_temporal = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn_temporal = SelfAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(dim, context_dim, heads)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, cfg.mlp_ratio)
        self.modulation = AdaLNModulation(dim, 4)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text: torch.Tensor,
                text_mask: Optional[torch.Tensor], grid: Tuple[int, int, int],
                rope_spatial: RopePair, rope_temporal: RopePair) -> torch.Tensor:
        b, n, d = x.shape
        tp, hp, wp = grid
        s = hp * wp
        (sh_s, sc_s, g_s), (sh_t, sc_t, g_t), (sh_c, sc_c, g_c), (sh_m, sc_m, g_m) = \
            self.modulation(cond)

        # --- attention within each frame ---
        h = modulate(self.norm_spatial(x), sh_s, sc_s)
        h = h.view(b * tp, s, d)
        h = self.attn_spatial(h, rope=rope_spatial).view(b, n, d)
        x = x + g_s.unsqueeze(1) * h

        # --- attention across time at each spatial position ---
        h = modulate(self.norm_temporal(x), sh_t, sc_t)
        h = h.view(b, tp, s, d).transpose(1, 2).reshape(b * s, tp, d)
        h = self.attn_temporal(h, rope=rope_temporal)
        h = h.view(b, s, tp, d).transpose(1, 2).reshape(b, n, d)
        x = x + g_t.unsqueeze(1) * h

        # --- prompt conditioning ---
        h = modulate(self.norm_cross(x), sh_c, sc_c)
        x = x + g_c.unsqueeze(1) * self.cross_attn(h, text, text_mask)

        h = modulate(self.norm_mlp(x), sh_m, sc_m)
        return x + g_m.unsqueeze(1) * self.mlp(h)


class AudioBlock(nn.Module):
    """Temporal self-attention -> text cross-attention -> MLP."""

    def __init__(self, cfg: DiTConfig, context_dim: int):
        super().__init__()
        dim, heads = cfg.dim, cfg.num_heads
        self.norm_attn = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.cross_attn = CrossAttention(dim, context_dim, heads)
        self.norm_mlp = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(dim, cfg.mlp_ratio)
        self.modulation = AdaLNModulation(dim, 3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, text: torch.Tensor,
                text_mask: Optional[torch.Tensor], rope: RopePair) -> torch.Tensor:
        (sh_a, sc_a, g_a), (sh_c, sc_c, g_c), (sh_m, sc_m, g_m) = self.modulation(cond)
        h = modulate(self.norm_attn(x), sh_a, sc_a)
        x = x + g_a.unsqueeze(1) * self.attn(h, rope=rope)
        h = modulate(self.norm_cross(x), sh_c, sc_c)
        x = x + g_c.unsqueeze(1) * self.cross_attn(h, text, text_mask)
        h = modulate(self.norm_mlp(x), sh_m, sc_m)
        return x + g_m.unsqueeze(1) * self.mlp(h)


class AVBridge(nn.Module):
    """Bidirectional cross attention between the two branches.

    Updates are simultaneous: audio reads the pre-update video tokens and vice
    versa, so neither branch gets a half-step advantage over the other.
    """

    def __init__(self, cfg: DiTConfig):
        super().__init__()
        dim, heads = cfg.dim, cfg.num_heads
        self.norm_v = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_a = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.v_from_a = CrossAttention(dim, dim, heads)
        self.a_from_v = CrossAttention(dim, dim, heads)
        self.mod_v = AdaLNModulation(dim, 1)
        self.mod_a = AdaLNModulation(dim, 1)

    def forward(self, video: torch.Tensor, audio: torch.Tensor,
                cond_v: torch.Tensor, cond_a: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        (sh_v, sc_v, g_v), = self.mod_v(cond_v)
        (sh_a, sc_a, g_a), = self.mod_a(cond_a)
        v_in = modulate(self.norm_v(video), sh_v, sc_v)
        a_in = modulate(self.norm_a(audio), sh_a, sc_a)
        video = video + g_v.unsqueeze(1) * self.v_from_a(v_in, a_in)
        audio = audio + g_a.unsqueeze(1) * self.a_from_v(a_in, v_in)
        return video, audio


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class DualBranchDiT(nn.Module):
    def __init__(self, cfg: DiTConfig, video_channels: int, audio_channels: int,
                 text_dim: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        self.video_channels = video_channels
        self.audio_channels = audio_channels
        context_dim = text_dim if text_dim is not None else cfg.text_dim

        video_patch_dim = video_channels * cfg.patch_t * cfg.patch_hw * cfg.patch_hw
        audio_patch_dim = audio_channels * cfg.patch_audio_t

        self.video_in = nn.Linear(video_patch_dim, cfg.dim)
        self.video_out = FinalLayer(cfg.dim, video_patch_dim)
        self.t_embed = TimestepEmbedder(cfg.dim)
        self.text_proj = nn.Linear(context_dim, cfg.dim)
        self.pooled_proj = nn.Sequential(
            nn.Linear(context_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )

        self.enable_audio = cfg.enable_audio
        if self.enable_audio:
            self.audio_in = nn.Linear(audio_patch_dim, cfg.dim)
            self.audio_out = FinalLayer(cfg.dim, audio_patch_dim)

        self.video_blocks = nn.ModuleList(
            [VideoBlock(cfg, cfg.dim) for _ in range(cfg.depth)]
        )
        if self.enable_audio:
            self.audio_blocks = nn.ModuleList(
                [AudioBlock(cfg, cfg.dim) for _ in range(cfg.depth)]
            )
            self.bridge_at = [
                i for i in range(cfg.depth)
                if cfg.bridge_every > 0 and (i + 1) % cfg.bridge_every == 0
            ]
            self.bridges = nn.ModuleList([AVBridge(cfg) for _ in self.bridge_at])
        else:
            self.bridge_at = []

        # Null conditioning for classifier-free guidance.
        self.null_text = nn.Parameter(torch.randn(1, 1, context_dim) * 0.02)
        self.null_pooled = nn.Parameter(torch.randn(1, context_dim) * 0.02)

        self.head_dim = cfg.dim // cfg.num_heads
        self._rope_cache: Dict[str, RopePair] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # -- rope caching -------------------------------------------------------
    def _rope(self, key: str, sizes: Tuple[int, ...], device: torch.device) -> RopePair:
        cache_key = "%s-%s-%s" % (key, sizes, device)
        if cache_key not in self._rope_cache:
            self._rope_cache[cache_key] = rope_params_nd(
                sizes, self.head_dim, self.cfg.rope_theta, device
            )
        return self._rope_cache[cache_key]

    # -- conditioning -------------------------------------------------------
    def _prepare_text(self, text: torch.Tensor, pooled: torch.Tensor,
                      mask: Optional[torch.Tensor], drop: Optional[torch.Tensor]
                      ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if drop is not None and drop.any():
            keep = (~drop).view(-1, 1, 1).to(text.dtype)
            text = text * keep + self.null_text.expand_as(text) * (1 - keep)
            pooled = pooled * keep[:, :, 0] + self.null_pooled.expand_as(pooled) * (1 - keep[:, :, 0])
            if mask is not None:
                # A dropped prompt still needs at least one visible token.
                mask = mask | drop.view(-1, 1)
        return text, pooled, mask

    def forward(
        self,
        video_latent: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        pooled: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
        audio_latent: Optional[torch.Tensor] = None,
        t_audio: Optional[torch.Tensor] = None,
        drop_cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Predict flow velocities for both branches.

        Separate ``t`` and ``t_audio`` are allowed so one modality can be held
        clean while the other is generated (e.g. video-to-audio dubbing).
        """
        cfg = self.cfg
        device = video_latent.device
        text, pooled, text_mask = self._prepare_text(text, pooled, text_mask, drop_cond)
        text_tokens = self.text_proj(text)
        pooled_vec = self.pooled_proj(pooled)

        v_tokens, grid = patchify_video(video_latent, cfg.patch_t, cfg.patch_hw)
        v = self.video_in(v_tokens)
        tp, hp, wp = grid

        cond_v = self.t_embed(t) + pooled_vec
        rope_spatial = self._rope("sp", (hp, wp), device)
        rope_temporal = self._rope("tm", (tp,), device)

        use_audio = self.enable_audio and audio_latent is not None
        if use_audio:
            a_tokens, ap = patchify_audio(audio_latent, cfg.patch_audio_t)
            a = self.audio_in(a_tokens)
            t_a = t if t_audio is None else t_audio
            cond_a = self.t_embed(t_a) + pooled_vec
            rope_audio = self._rope("au", (ap,), device)
        else:
            a = cond_a = rope_audio = None

        bridge_lookup = {block: i for i, block in enumerate(self.bridge_at)}
        for i, block in enumerate(self.video_blocks):
            v = block(v, cond_v, text_tokens, text_mask, grid, rope_spatial, rope_temporal)
            if use_audio:
                a = self.audio_blocks[i](a, cond_a, text_tokens, text_mask, rope_audio)
                if i in bridge_lookup:
                    v, a = self.bridges[bridge_lookup[i]](v, a, cond_v, cond_a)

        v_out = unpatchify_video(self.video_out(v, cond_v), grid,
                                 self.video_channels, cfg.patch_t, cfg.patch_hw)
        a_out = None
        if use_audio:
            a_out = unpatchify_audio(self.audio_out(a, cond_a),
                                     self.audio_channels, cfg.patch_audio_t)
        return v_out, a_out
