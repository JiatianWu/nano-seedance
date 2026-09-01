"""Real-video dataset.

Expected layout -- either sidecar caption files::

    root/clip0001.mp4
    root/clip0001.txt          # the caption

or a manifest, which takes precedence if present::

    root/metadata.jsonl        # {"video": "clip0001.mp4", "caption": "..."}

Audio is read from the video's own track when torchaudio can demux it, else
from a sidecar ``.wav``, else treated as silence. Decoding uses decord when
installed and imageio-ffmpeg otherwise.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import DataConfig
from .audio import MelSpectrogram


def _discover(root: str) -> List[Dict[str, str]]:
    manifest = os.path.join(root, "metadata.jsonl")
    if os.path.exists(manifest):
        entries = []
        with open(manifest, "r") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    entries = []
    for ext in ("mp4", "webm", "mkv", "avi", "gif"):
        for path in sorted(glob.glob(os.path.join(root, "*." + ext))):
            caption_path = os.path.splitext(path)[0] + ".txt"
            caption = ""
            if os.path.exists(caption_path):
                with open(caption_path, "r") as fh:
                    caption = fh.read().strip()
            entries.append({"video": os.path.basename(path), "caption": caption})
    return entries


def _decode_video(path: str, num_frames: int, stride: int) -> np.ndarray:
    """Return (T, H, W, 3) uint8, sampling ``num_frames`` at ``stride``."""
    try:
        import decord  # type: ignore

        reader = decord.VideoReader(path)
        total = len(reader)
        span = min(total, num_frames * stride)
        start = max(0, (total - span) // 2)
        idx = [min(total - 1, start + i * stride) for i in range(num_frames)]
        return reader.get_batch(idx).asnumpy()
    except ImportError:
        pass

    import imageio.v2 as imageio

    reader = imageio.get_reader(path)
    frames = []
    try:
        for i, frame in enumerate(reader):
            if i % stride == 0:
                frames.append(np.asarray(frame)[..., :3])
            if len(frames) >= num_frames:
                break
    finally:
        reader.close()
    if not frames:
        raise RuntimeError("no frames decoded from %s" % path)
    while len(frames) < num_frames:  # loop short clips
        frames.append(frames[-1])
    return np.stack(frames)


def _load_audio(video_path: str, sample_rate: int, num_samples: int) -> torch.Tensor:
    wav_path = os.path.splitext(video_path)[0] + ".wav"
    waveform: Optional[torch.Tensor] = None

    try:
        import torchaudio  # type: ignore

        source = wav_path if os.path.exists(wav_path) else video_path
        data, sr = torchaudio.load(source)
        waveform = data.mean(0)
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    except Exception:
        waveform = None

    if waveform is None:
        return torch.zeros(num_samples)
    if waveform.numel() < num_samples:
        waveform = F.pad(waveform, (0, num_samples - waveform.numel()))
    return waveform[:num_samples]


def _resize_video(frames: np.ndarray, resolution: int) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (3, T, res, res) float in [-1, 1], centre-cropped."""
    video = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    _, _, h, w = video.shape
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    video = video[:, :, top:top + side, left:left + side]
    video = F.interpolate(video, size=(resolution, resolution),
                          mode="bilinear", align_corners=False)
    return video.permute(1, 0, 2, 3).contiguous()


class RealVideoDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str = "train",
                 caption_style: str = "simple", indices: Optional[List[int]] = None):
        if not cfg.root:
            raise ValueError("data.root must be set to use the real-video loader")
        self.cfg = cfg
        self.root = cfg.root
        self.entries = _discover(cfg.root)
        if not self.entries:
            raise FileNotFoundError("no videos found under %s" % cfg.root)

        # Deterministic split: every 10th clip is held out.
        all_idx = list(range(len(self.entries)))
        holdout = set(all_idx[::10])
        default = [i for i in all_idx if (i in holdout) == (split == "val")]
        self.indices = default if indices is None else list(indices)

        self.mel = MelSpectrogram(
            sample_rate=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
            n_mels=cfg.n_mels, n_frames=cfg.mel_frames,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        index = self.indices[i]
        entry = self.entries[index]
        path = os.path.join(self.root, entry["video"])

        frames = _decode_video(path, self.cfg.num_frames, stride=1)
        video = _resize_video(frames, self.cfg.resolution)
        waveform = _load_audio(path, self.cfg.sample_rate, self.cfg.audio_samples)
        mel = self.mel.forward(waveform)

        return {
            "video": video,
            "mel": mel,
            "waveform": waveform,
            "caption": entry.get("caption", ""),
            # Real corpora have no ground-truth attributes; the attribute-based
            # rewards fall back to their prompt-free terms.
            "meta": {"quality": 1.0},
            "index": index,
        }
