"""Reward models shared by both RL algorithms.

Real systems train a neural reward model on human preference data. That is the
right design, but it makes it impossible to tell whether an RL implementation
is correct: if reward goes up you cannot say whether the policy improved or the
reward model was gamed.

So the rewards here are *verifiable*. The synthetic world has ground truth --
the caption states the colour, shape and direction, and the generative process
ties audio pitch to colour and audio transients to visual bounces. Every reward
below measures agreement with a fact that is knowable independently of the
model, so "reward went up" means the samples really did get better.

``NeuralReward`` shows where a learned reward model would slot in; the
interface is the same.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..config import RewardConfig
from ..data.audio import mel_center_frequencies, onset_envelope
from ..data.synthetic import (
    COLOR_NAMES,
    COLOR_PITCH_HZ,
    COLORS,
    DIRECTION_VECTORS,
    DIRECTIONS,
    SHAPES,
)


@dataclass
class RewardInput:
    """Decoded generations to be scored."""

    video: torch.Tensor                      # (B, 3, T, H, W) in [-1, 1]
    captions: Sequence[str]
    mel: Optional[torch.Tensor] = None       # (B, n_mels, T_mel) in [-1, 1]
    sample_rate: int = 8000
    fps: int = 8
    hop_length: int = 128


# --------------------------------------------------------------------------
# shared visual measurements
# --------------------------------------------------------------------------
def parse_caption(caption: str) -> Dict[str, Optional[str]]:
    words = caption.lower().split()
    out: Dict[str, Optional[str]] = {"color": None, "shape": None, "direction": None}
    for word in words:
        if word in COLOR_NAMES and out["color"] is None:
            out["color"] = word
        if word in SHAPES and out["shape"] is None:
            out["shape"] = word
        if word in DIRECTIONS and out["direction"] is None:
            out["direction"] = word
    return out


def color_mass(video: torch.Tensor, rgb: Tuple[float, float, float],
               tolerance: float = 0.55) -> torch.Tensor:
    """Soft per-frame count of pixels matching ``rgb``. Returns (B, T, H, W)."""
    target = torch.tensor(rgb, device=video.device, dtype=video.dtype).view(1, 3, 1, 1, 1)
    distance = ((video + 1) / 2 - target).pow(2).sum(1).sqrt()
    return torch.clamp(1.0 - distance / tolerance, min=0.0) ** 2


def centroid_trajectory(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Weighted centroid per frame. ``weight`` is (B, T, H, W).

    Returns ``(cx, cy, mass)`` with cx/cy shaped (B, T) in pixel units.
    """
    b, t, h, w = weight.shape
    idx_h = torch.arange(h, device=weight.device, dtype=weight.dtype)
    idx_w = torch.arange(w, device=weight.device, dtype=weight.dtype)
    mass = weight.sum(dim=(-2, -1))
    safe = mass.clamp(min=1e-6)
    cy = (weight.sum(-1) * idx_h).sum(-1) / safe
    cx = (weight.sum(-2) * idx_w).sum(-1) / safe
    return cx, cy, mass


def foreground_weight(video: torch.Tensor, blur: int = 3) -> torch.Tensor:
    """Colour-agnostic saliency: deviation from the per-clip temporal median.

    A 3x3 box blur first, because the degraded half of the corpus carries
    enough pixel noise to drag the centroid around by more than a bounce does.
    Squaring at the end concentrates the weight on the shape.
    """
    b, c, t, h, w = video.shape
    if blur > 1:
        flat = video.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        flat = torch.nn.functional.avg_pool2d(flat, blur, stride=1, padding=blur // 2)
        video = flat.view(b, t, c, h, w).permute(0, 2, 1, 3, 4)
    background = video.median(dim=2, keepdim=True).values
    return (video - background).abs().mean(dim=1) ** 2


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
class Reward:
    name = "reward"

    def __call__(self, sample: RewardInput) -> torch.Tensor:
        raise NotImplementedError


class PromptAlignmentReward(Reward):
    """Does the video show the colour and motion the caption asked for?

    Colour is scored competitively against the rest of the palette, so
    producing a blob of every colour does not win. Direction is scored by the
    centroid displacement over the opening frames, which the data generator
    guarantees matches the caption (see ``render_video``).
    """

    name = "prompt_align"

    def __init__(self, motion_window: int = 5):
        self.motion_window = motion_window

    def __call__(self, sample: RewardInput) -> torch.Tensor:
        video = sample.video.float()
        batch, _, frames, _, _ = video.shape
        device = video.device
        scores = torch.zeros(batch, device=device)

        masses = {name: color_mass(video, rgb) for name, rgb in COLORS.items()}
        total = torch.stack([m.sum(dim=(-3, -2, -1)) for m in masses.values()], dim=0).sum(0)

        window = min(self.motion_window, frames - 1)
        for i, caption in enumerate(sample.captions):
            parsed = parse_caption(caption)
            terms: List[torch.Tensor] = []

            if parsed["color"] is not None:
                target = masses[parsed["color"]][i].sum()
                terms.append(target / total[i].clamp(min=1e-6))

            if parsed["direction"] is not None and window >= 1:
                rgb = COLORS.get(parsed["color"]) if parsed["color"] else None
                weight = (masses[parsed["color"]][i:i + 1] if rgb is not None
                          else foreground_weight(video[i:i + 1]))
                cx, cy, mass = centroid_trajectory(weight)
                dx = cx[0, window] - cx[0, 0]
                dy = cy[0, window] - cy[0, 0]
                vx, vy = DIRECTION_VECTORS[parsed["direction"]]
                displacement = torch.stack([dx, dy])
                norm = displacement.norm().clamp(min=1e-6)
                cosine = (displacement[0] * vx + displacement[1] * vy) / norm
                # Require actual movement, not just the right sign of a jitter.
                magnitude = torch.clamp(norm / (0.15 * video.shape[-1]), max=1.0)
                # Only trust the direction when the shape was actually found.
                confidence = torch.clamp(mass.mean() / 5.0, max=1.0)
                terms.append(0.5 * (1 + cosine) * magnitude * confidence)

            scores[i] = torch.stack(terms).mean() if terms else torch.tensor(0.5, device=device)
        return scores


class AVSyncReward(Reward):
    """Do audible transients line up with visual direction changes?

    A bounce is a discontinuity in velocity, so the visual event signal is the
    magnitude of centroid acceleration; the audio event signal is spectral
    flux. This is the reward that actually exercises the cross-modal bridge.

    Alignment is the whole game here. Audio transients are ~1 video frame wide,
    so the two series must be matched on a real time axis -- the mel hop and
    the video fps are unrelated rates, and acceleration index ``j`` is centred
    on video frame ``j + 1``. Getting that offset wrong does not merely weaken
    the reward, it anti-correlates it.
    """

    name = "av_sync"

    def __call__(self, sample: RewardInput) -> torch.Tensor:
        video = sample.video.float()
        batch = video.shape[0]
        device = video.device
        if sample.mel is None:
            return torch.full((batch,), 0.5, device=device)

        weight = foreground_weight(video)
        cx, cy, _ = centroid_trajectory(weight)
        position = torch.stack([cx, cy], dim=-1)               # (B, T, 2)
        acceleration = (position[:, 2:] - 2 * position[:, 1:-1] + position[:, :-2])
        visual = acceleration.norm(dim=-1)                      # (B, T-2)

        audio = onset_envelope(sample.mel.float().to(device))   # (B, T_mel)
        audio = _align_to_video_frames(audio, visual.shape[1], sample.fps,
                                       sample.hop_length, sample.sample_rate,
                                       frame_offset=1.0)

        return _normalised_correlation(visual, audio).clamp(min=0.0)


class AudioPromptReward(Reward):
    """Is the generated tone at the pitch the caption's colour implies?

    Pitch is read straight off the mel spectrogram via band centre
    frequencies, and scored against the other palette pitches so that broadband
    noise cannot win.
    """

    name = "audio_prompt"

    def __init__(self, sample_rate: int = 8000, n_mels: int = 32,
                 sharpness: float = 6.0, contrast: float = 8.0):
        self.centers = mel_center_frequencies(sample_rate, n_mels)
        self.sharpness = sharpness
        self.contrast = contrast

    def __call__(self, sample: RewardInput) -> torch.Tensor:
        batch = sample.video.shape[0]
        device = sample.video.device
        if sample.mel is None:
            return torch.full((batch,), 0.5, device=device)

        mel = sample.mel.float().to(device)
        energy = ((mel + 1) / 2).mean(dim=-1)                   # (B, n_mels)
        # Every band carries a noise floor in the log domain; without removing
        # it the floor summed over 32 bands swamps the actual spectral peak.
        energy = torch.clamp(energy - energy.median(dim=-1, keepdim=True).values, min=0.0)
        centers = self.centers.to(device)

        # Log-frequency distance from each band to each palette pitch.
        pitches = torch.tensor([COLOR_PITCH_HZ[c] for c in COLOR_NAMES], device=device)
        log_dist = (centers.view(1, -1).log() - pitches.view(-1, 1).log()).abs()
        affinity = torch.exp(-self.sharpness * log_dist)        # (n_colors, n_mels)

        # A broad kernel ranks the palette reliably but leaves neighbouring
        # pitches nearly tied; the power transform separates the winner
        # without disturbing the ranking.
        response = (energy @ affinity.t()) ** self.contrast     # (B, n_colors)
        probs = response / response.sum(-1, keepdim=True).clamp(min=1e-9)

        scores = torch.full((batch,), 0.5, device=device)
        for i, caption in enumerate(sample.captions):
            color = parse_caption(caption)["color"]
            if color is not None:
                scores[i] = probs[i, COLOR_NAMES.index(color)]
        return scores


class AestheticReward(Reward):
    """Cheap stand-in for a learned aesthetic scorer.

    Rewards high foreground/background contrast and penalises the pixel noise
    that marks the degraded half of the corpus. Calibrated against real clips
    so the scale is interpretable.
    """

    name = "aesthetic"

    def __init__(self, tv_reference: float = 0.12, contrast_reference: float = 0.5):
        self.tv_reference = tv_reference
        self.contrast_reference = contrast_reference

    @torch.no_grad()
    def calibrate(self, videos: torch.Tensor) -> None:
        self.tv_reference = float(_total_variation(videos.float()).mean())
        self.contrast_reference = float(videos.float().flatten(1).std(dim=1).mean())

    def __call__(self, sample: RewardInput) -> torch.Tensor:
        video = sample.video.float()
        tv = _total_variation(video)
        contrast = video.flatten(1).std(dim=1)
        noise_penalty = torch.clamp(tv / max(self.tv_reference, 1e-6), min=0.0)
        contrast_score = torch.clamp(contrast / max(self.contrast_reference, 1e-6), max=1.5)
        return torch.clamp(contrast_score - 0.5 * torch.clamp(noise_penalty - 1.0, min=0.0),
                           min=0.0)


class NeuralReward(Reward):
    """Adapter for a learned reward model.

    Swap this in once you have preference data: train any module mapping
    (video, mel, caption) -> scalar and pass it here. Nothing else in the RL
    code changes.
    """

    name = "neural"

    def __init__(self, model: torch.nn.Module):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, sample: RewardInput) -> torch.Tensor:
        return self.model(sample.video, sample.mel, sample.captions).flatten()


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------
class RewardEnsemble:
    """Weighted sum of rewards, returning the total and the breakdown."""

    def __init__(self, cfg: RewardConfig, sample_rate: int = 8000, n_mels: int = 32):
        self.cfg = cfg
        self.rewards: List[Tuple[Reward, float]] = []
        if cfg.prompt_align:
            self.rewards.append((PromptAlignmentReward(), cfg.prompt_align))
        if cfg.av_sync:
            self.rewards.append((AVSyncReward(), cfg.av_sync))
        if cfg.audio_prompt:
            self.rewards.append((AudioPromptReward(sample_rate, n_mels), cfg.audio_prompt))
        if cfg.aesthetic:
            self.rewards.append((AestheticReward(), cfg.aesthetic))

    def calibrate(self, videos: torch.Tensor) -> None:
        for reward, _ in self.rewards:
            if isinstance(reward, AestheticReward):
                reward.calibrate(videos)

    @torch.no_grad()
    def __call__(self, sample: RewardInput) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total = torch.zeros(sample.video.shape[0], device=sample.video.device)
        breakdown: Dict[str, torch.Tensor] = {}
        for reward, weight in self.rewards:
            value = reward(sample).to(total.device)
            breakdown[reward.name] = value
            total = total + weight * value
        breakdown["total"] = total
        return total, breakdown


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _align_to_video_frames(envelope: torch.Tensor, num_frames: int, fps: int,
                           hop_length: int, sample_rate: int,
                           frame_offset: float = 0.0) -> torch.Tensor:
    """Pool an audio-rate envelope onto the video frame grid.

    Max-pooling over each frame's time window rather than point-sampling:
    a transient one mel-hop wide would simply be stepped over by linear
    interpolation when the audio rate is ~8x the video rate.
    """
    batch, num_mel = envelope.shape
    out = envelope.new_zeros(batch, num_frames)
    frames_per_second = float(sample_rate) / hop_length
    for j in range(num_frames):
        center = (j + frame_offset) / fps
        start = int(max((center - 0.5 / fps) * frames_per_second, 0))
        stop = max(int((center + 0.5 / fps) * frames_per_second), start + 1)
        if start < num_mel:
            out[:, j] = envelope[:, start:min(stop, num_mel)].amax(dim=-1)
    return out


def _normalised_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a - a.mean(dim=-1, keepdim=True)
    b = b - b.mean(dim=-1, keepdim=True)
    denominator = a.norm(dim=-1) * b.norm(dim=-1)
    return (a * b).sum(-1) / denominator.clamp(min=1e-6)


def _total_variation(video: torch.Tensor) -> torch.Tensor:
    dh = (video[..., 1:, :] - video[..., :-1, :]).abs().flatten(1).mean(1)
    dw = (video[..., :, 1:] - video[..., :, :-1]).abs().flatten(1).mean(1)
    return dh + dw
