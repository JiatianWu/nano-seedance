"""Procedural audio-video corpus with captions and ground-truth attributes.

Why synthetic: a real text-to-video corpus is terabytes and needs GPUs, which
would make none of this runnable. This generator produces a small world with
genuine cross-modal structure -- a shape bounces around the frame and every
bounce produces an audible transient at a pitch determined by the shape's
colour -- so the audio branch and the AV bridge have something real to learn,
and so rewards can be *verified* against known attributes instead of guessed.

Each sample carries a quality score, and roughly half the corpus is degraded
with noise and low contrast. That mirrors the real pretrain/SFT split: pretrain
on everything, fine-tune on the curated top slice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .audio import MelSpectrogram

SHAPES: Tuple[str, ...] = ("circle", "square", "triangle")
DIRECTIONS: Tuple[str, ...] = ("left", "right", "up", "down")
SPEEDS: Tuple[str, ...] = ("slow", "fast")

COLORS: Dict[str, Tuple[float, float, float]] = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.80, 0.25),
    "blue": (0.20, 0.35, 0.95),
    "yellow": (0.95, 0.85, 0.15),
    "magenta": (0.90, 0.20, 0.85),
    "cyan": (0.20, 0.85, 0.90),
}
COLOR_NAMES: Tuple[str, ...] = tuple(COLORS.keys())

BACKGROUNDS: Dict[str, Tuple[float, float, float]] = {
    "black": (0.04, 0.04, 0.05),
    "grey": (0.45, 0.45, 0.47),
    "white": (0.92, 0.92, 0.90),
}
BACKGROUND_NAMES: Tuple[str, ...] = tuple(BACKGROUNDS.keys())

# One pitch per colour: this is the cross-modal rule the model has to learn.
COLOR_PITCH_HZ: Dict[str, float] = {
    "red": 220.0,
    "green": 277.2,
    "blue": 329.6,
    "yellow": 392.0,
    "magenta": 466.2,
    "cyan": 523.3,
}
# One timbre per shape.
SHAPE_TIMBRE: Dict[str, str] = {"circle": "soft", "square": "harsh", "triangle": "bright"}

DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
}


@dataclass
class SampleSpec:
    """Ground-truth attributes for one clip."""

    shape: str
    color: str
    background: str
    direction: str
    speed: str
    noise: float
    quality: float
    bounce_times: List[float] = field(default_factory=list)

    @property
    def pitch_hz(self) -> float:
        return COLOR_PITCH_HZ[self.color]

    @property
    def timbre(self) -> str:
        return SHAPE_TIMBRE[self.shape]

    def caption(self, style: str = "simple") -> str:
        if style == "detailed":
            return (
                "a %s %s %s moving %s on a %s background with a %s tone"
                % (self.speed, self.color, self.shape, self.direction,
                   self.background, self.timbre)
            )
        return "a %s %s moving %s" % (self.color, self.shape, self.direction)

    def to_meta(self) -> Dict[str, Any]:
        return {
            "shape_idx": SHAPES.index(self.shape),
            "color_idx": COLOR_NAMES.index(self.color),
            "background_idx": BACKGROUND_NAMES.index(self.background),
            "direction_idx": DIRECTIONS.index(self.direction),
            "speed_idx": SPEEDS.index(self.speed),
            "pitch_hz": self.pitch_hz,
            "quality": self.quality,
            "noise": self.noise,
            "num_bounces": len(self.bounce_times),
        }


def sample_spec(rng: np.random.Generator) -> SampleSpec:
    shape = str(rng.choice(SHAPES))
    color = str(rng.choice(COLOR_NAMES))
    background = str(rng.choice(BACKGROUND_NAMES))
    direction = str(rng.choice(DIRECTIONS))
    speed = str(rng.choice(SPEEDS))

    # 40% of the corpus is clean; the rest is degraded to varying degrees.
    noise = float(rng.uniform(0.0, 0.04) if rng.random() < 0.4 else rng.uniform(0.08, 0.35))

    contrast = _rgb_distance(COLORS[color], BACKGROUNDS[background])
    quality = float(np.clip(contrast * (1.0 - 2.0 * noise), 0.0, 1.0))
    return SampleSpec(shape, color, background, direction, speed, noise, quality)


def _rgb_distance(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)) / math.sqrt(3.0))


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _shape_alpha(shape: str, yy: np.ndarray, xx: np.ndarray,
                 cy: float, cx: float, radius: float) -> np.ndarray:
    """Antialiased coverage mask via a signed distance field."""
    dy, dx = yy - cy, xx - cx
    if shape == "circle":
        dist = np.sqrt(dy * dy + dx * dx) - radius
    elif shape == "square":
        dist = np.maximum(np.abs(dy), np.abs(dx)) - radius
    elif shape == "triangle":
        # Upward-pointing equilateral triangle as an intersection of 3 half-planes.
        h = radius * 1.5
        d0 = dy - h * 0.5
        d1 = (-0.5 * dy - math.sqrt(3) / 2 * dx) - h * 0.5
        d2 = (-0.5 * dy + math.sqrt(3) / 2 * dx) - h * 0.5
        dist = np.maximum(np.maximum(d0, d1), d2)
    else:
        raise ValueError("unknown shape %r" % shape)
    return np.clip(0.5 - dist, 0.0, 1.0)


def render_video(spec: SampleSpec, resolution: int, num_frames: int,
                 rng: np.random.Generator) -> Tuple[np.ndarray, List[float]]:
    """Return (T, H, W, 3) float video in [-1, 1] and the list of bounce frames."""
    size = resolution
    radius = size * 0.16
    speed = size * (0.06 if spec.speed == "slow" else 0.12)

    vx, vy = DIRECTION_VECTORS[spec.direction]
    # Slight orthogonal drift keeps the motion from being perfectly axis-aligned.
    # Capped well below 1 so the captioned axis always dominates.
    drift = float(rng.uniform(-0.3, 0.3))
    vx, vy = (vx + drift * vy) * speed, (vy + drift * vx) * speed

    lo, hi = radius, size - radius
    span = hi - lo

    # Start with runway: the shape is placed at the far end of its travel axis
    # so the first (longest) leg of the trajectory matches the caption. Without
    # this a clip captioned "moving up" can start against the top wall, bounce
    # on frame 1, and move down for the rest of the clip.
    def _start(velocity: float) -> float:
        if velocity > 0:
            return float(rng.uniform(lo, lo + 0.2 * span))
        if velocity < 0:
            return float(rng.uniform(hi - 0.2 * span, hi))
        return float(rng.uniform(lo, hi))

    cx = _start(vx)
    cy = _start(vy)

    yy, xx = np.meshgrid(np.arange(size, dtype=np.float32),
                         np.arange(size, dtype=np.float32), indexing="ij")

    bg = np.array(BACKGROUNDS[spec.background], dtype=np.float32)
    fg = np.array(COLORS[spec.color], dtype=np.float32)

    frames = np.empty((num_frames, size, size, 3), dtype=np.float32)
    bounce_frames: List[float] = []

    for t in range(num_frames):
        alpha = _shape_alpha(spec.shape, yy, xx, cy, cx, radius)[..., None]
        frames[t] = bg[None, None, :] * (1 - alpha) + fg[None, None, :] * alpha

        cx += vx
        cy += vy
        bounced = False
        if cx < lo or cx > hi:
            cx = float(np.clip(cx, lo, hi))
            vx = -vx
            bounced = True
        if cy < lo or cy > hi:
            cy = float(np.clip(cy, lo, hi))
            vy = -vy
            bounced = True
        if bounced:
            bounce_frames.append(float(t + 1))

    if spec.noise > 0:
        frames += rng.normal(0.0, spec.noise, size=frames.shape).astype(np.float32)
        # Low quality also means washed-out contrast, not just noise.
        frames = 0.5 + (frames - 0.5) * (1.0 - 0.5 * spec.noise)

    frames = np.clip(frames, 0.0, 1.0) * 2.0 - 1.0
    return frames, bounce_frames


def render_audio(spec: SampleSpec, num_samples: int, sample_rate: int,
                 fps: int, rng: np.random.Generator) -> np.ndarray:
    """Tone at the colour's pitch, with a transient at every visual bounce."""
    t = np.arange(num_samples, dtype=np.float32) / sample_rate
    freq = spec.pitch_hz
    phase = 2 * np.pi * freq * t

    if spec.shape == "circle":
        tone = np.sin(phase)
    elif spec.shape == "square":
        tone = 0.7 * np.sign(np.sin(phase)) + 0.3 * np.sin(2 * phase)
    else:  # triangle -> bright, odd harmonics
        tone = (0.6 * np.sin(phase) + 0.25 * np.sin(3 * phase) + 0.15 * np.sin(5 * phase))

    envelope = np.full(num_samples, 0.18, dtype=np.float32)
    for frame in spec.bounce_times:
        onset = frame / fps
        decay = np.exp(-np.maximum(t - onset, 0.0) / 0.06)
        envelope += 0.9 * decay * (t >= onset)

    audio = tone.astype(np.float32) * envelope
    if spec.noise > 0:
        audio += rng.normal(0.0, 0.05 * spec.noise, size=audio.shape).astype(np.float32)

    peak = float(np.abs(audio).max())
    if peak > 1e-6:
        audio = audio / peak * 0.95
    return audio.astype(np.float32)


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
class SyntheticAVDataset(Dataset):
    """Deterministic per-index generation; nothing is cached to disk."""

    def __init__(self, cfg: DataConfig, split: str = "train",
                 caption_style: str = "simple", indices: Optional[List[int]] = None):
        self.cfg = cfg
        self.split = split
        self.caption_style = caption_style
        self.base_seed = cfg.seed + (0 if split == "train" else 1_000_000)
        total = cfg.num_train if split == "train" else cfg.num_val
        self.indices = list(range(total)) if indices is None else list(indices)
        self.mel = MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft,
            hop_length=cfg.hop_length, n_mels=cfg.n_mels, n_frames=cfg.mel_frames,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def spec_for(self, index: int) -> SampleSpec:
        """Attributes only -- cheap enough to call for corpus-wide curation."""
        rng = np.random.default_rng(self.base_seed * 1_000_003 + index)
        return sample_spec(rng)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        index = self.indices[i]
        rng = np.random.default_rng(self.base_seed * 1_000_003 + index)
        spec = sample_spec(rng)

        video, bounce_frames = render_video(spec, self.cfg.resolution,
                                            self.cfg.num_frames, rng)
        spec.bounce_times = bounce_frames
        waveform = render_audio(spec, self.cfg.audio_samples,
                                self.cfg.sample_rate, self.cfg.fps, rng)

        video_t = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)
        waveform_t = torch.from_numpy(waveform)
        mel_t = self.mel.forward(waveform_t)  # (n_mels, mel_frames)

        return {
            "video": video_t,
            "mel": mel_t,
            "waveform": waveform_t,
            "caption": spec.caption(self.caption_style),
            "meta": spec.to_meta(),
            "index": index,
        }


def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack tensors, keep captions as a list, stack meta into tensors."""
    out: Dict[str, Any] = {
        "video": torch.stack([b["video"] for b in batch]),
        "mel": torch.stack([b["mel"] for b in batch]),
        "waveform": torch.stack([b["waveform"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
    }
    meta_keys = batch[0]["meta"].keys()
    out["meta"] = {
        k: torch.tensor([b["meta"][k] for b in batch],
                        dtype=torch.long if k.endswith("_idx") or k == "num_bounces"
                        else torch.float32)
        for k in meta_keys
    }
    return out


def random_prompts(n: int, seed: int = 0, style: str = "detailed") -> List[str]:
    """Sample captions from the same grammar the corpus uses.

    RL stages need prompts without needing the matching ground-truth clip.
    """
    rng = np.random.default_rng(seed)
    return [sample_spec(rng).caption(style) for _ in range(n)]


def curate_indices(cfg: DataConfig, quantile: float, split: str = "train") -> List[int]:
    """Return the indices whose quality score is in the top ``1 - quantile``.

    This is the toy stand-in for a real SFT curation pass (aesthetic scoring,
    motion filtering, caption re-writing).
    """
    probe = SyntheticAVDataset(cfg, split=split)
    scores = np.array([probe.spec_for(i).quality for i in probe.indices])
    threshold = float(np.quantile(scores, quantile))
    keep = [i for i, s in zip(probe.indices, scores) if s >= threshold]
    return keep or probe.indices
