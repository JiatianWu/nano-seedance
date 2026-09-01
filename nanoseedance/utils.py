"""Small shared helpers: devices, seeding, EMA, checkpoints, media writing."""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
import wave
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# device / seeding
# --------------------------------------------------------------------------
def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def human_count(n: int) -> str:
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= scale:
            return "%.2f%s" % (n / scale, unit)
    return str(n)


# --------------------------------------------------------------------------
# training loop helpers
# --------------------------------------------------------------------------
def lr_lambda(step: int, warmup_steps: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """Linear warmup then cosine decay, expressed as a multiplier on base lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, progress))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


class EMA:
    """Exponential moving average of model parameters.

    Shadow copies live on the model's own device and are kept in float32 even
    if the model is not.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self._backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key not in self.shadow:
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach().float(), alpha=1 - self.decay)

    @contextlib.contextmanager
    def averaged(self, model: nn.Module) -> Iterator[None]:
        """Temporarily swap EMA weights into the model."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if k in self.shadow}
        model.load_state_dict(
            {k: v.to(dtype=self._backup[k].dtype) for k, v in self.shadow.items()}, strict=False
        )
        try:
            yield
        finally:
            model.load_state_dict(self._backup, strict=False)
            self._backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.clone().float() for k, v in state.items()}


class AverageMeter:
    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def add(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0


class Logger:
    """Prints to stdout and appends JSONL to ``out_dir/log.jsonl``."""

    def __init__(self, out_dir: str, name: str = "train"):
        os.makedirs(out_dir, exist_ok=True)
        self.path = os.path.join(out_dir, "log.jsonl")
        self.name = name
        self.start = time.time()

    def log(self, step: int, **metrics: float) -> None:
        record = {"step": step, "elapsed": round(time.time() - self.start, 2)}
        record.update({k: (round(float(v), 6) if isinstance(v, (int, float)) else v)
                       for k, v in metrics.items()})
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
        body = "  ".join(
            "%s %.4f" % (k, v) if isinstance(v, float) else "%s %s" % (k, v)
            for k, v in record.items() if k not in ("step", "elapsed")
        )
        print("[%s] step %5d | %6.1fs | %s" % (self.name, step, record["elapsed"], body), flush=True)


def infinite(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------
def save_checkpoint(path: str, **payload: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "checkpoint not found: %s\nRun the preceding pipeline stage first." % path
        )
    # These checkpoints contain config dicts alongside tensors, so full unpickling
    # is required; only load checkpoints you produced yourself.
    return torch.load(path, map_location=map_location, weights_only=False)


# --------------------------------------------------------------------------
# media writing
# --------------------------------------------------------------------------
def to_uint8_video(video: torch.Tensor) -> np.ndarray:
    """(C, T, H, W) in [-1, 1] -> (T, H, W, C) uint8."""
    video = video.detach().float().clamp(-1, 1)
    video = ((video + 1) * 127.5).round().to(torch.uint8)
    return video.permute(1, 2, 3, 0).cpu().numpy()


def save_video(path: str, video: torch.Tensor, fps: int = 8) -> str:
    """Write (C, T, H, W) in [-1, 1] to mp4, falling back to GIF."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = to_uint8_video(video)
    try:
        import imageio.v2 as imageio

        if path.endswith(".mp4"):
            try:
                imageio.mimwrite(path, list(frames), fps=fps, macro_block_size=None,
                                 codec="libx264", quality=8)
                return path
            except Exception:
                path = path[:-4] + ".gif"
        imageio.mimwrite(path, list(frames), duration=1.0 / max(fps, 1), loop=0)
        return path
    except Exception:
        npy_path = os.path.splitext(path)[0] + ".npy"
        np.save(npy_path, frames)
        return npy_path


def save_wav(path: str, waveform: torch.Tensor, sample_rate: int) -> str:
    """Write a mono waveform in [-1, 1] using only the stdlib."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    audio = waveform.detach().float().flatten().clamp(-1, 1).cpu().numpy()
    pcm = (audio * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())
    return path


def save_contact_sheet(path: str, video: torch.Tensor, max_frames: int = 8) -> str:
    """Write a horizontal strip of frames as a PNG, handy for quick eyeballing."""
    from PIL import Image

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = to_uint8_video(video)
    stride = max(1, len(frames) // max_frames)
    picked = frames[::stride][:max_frames]
    strip = np.concatenate(list(picked), axis=1)
    Image.fromarray(strip).save(path)
    return path
