"""Config validation and dataset ground-truth consistency."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nanoseedance.config import Config
from nanoseedance.data import SyntheticAVDataset, collate, curate_indices
from nanoseedance.data.audio import MelSpectrogram
from nanoseedance.data.synthetic import COLORS, DIRECTION_VECTORS
from nanoseedance.data.text import CaptionTokenizer
from nanoseedance.rl.rewards import centroid_trajectory, color_mass


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/toy.yaml")


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
def test_overrides_and_derived_geometry():
    cfg = Config.load("configs/toy.yaml", ["train.batch_size=3", "dit.depth=2"])
    assert cfg.train.batch_size == 3 and cfg.dit.depth == 2
    assert cfg.latent_frames == 1 + (cfg.data.num_frames - 1) // cfg.video_vae.temporal_compression
    assert cfg.latent_size == cfg.data.resolution // cfg.video_vae.spatial_compression


def test_rejects_incompatible_frame_count():
    with pytest.raises(ValueError, match="num_frames"):
        Config.load("configs/toy.yaml", ["data.num_frames=16"])


def test_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown"):
        Config.load("configs/toy.yaml", ["dit.nonexistent=1"])


def test_rejects_bad_head_split():
    with pytest.raises(ValueError, match="divisible"):
        Config.load("configs/toy.yaml", ["dit.num_heads=5"])


# --------------------------------------------------------------------------
# tokenizer
# --------------------------------------------------------------------------
def test_tokenizer_roundtrip(cfg):
    tok = CaptionTokenizer(cfg.dit.max_text_len)
    ids, mask = tok.batch_encode(["a red circle moving left", "a cyan square moving up"])
    assert ids.shape == (2, cfg.dit.max_text_len)
    assert tok.decode(ids[0][mask[0]]) == "a red circle moving left"


def test_tokenizer_handles_unknown_words(cfg):
    tok = CaptionTokenizer(cfg.dit.max_text_len)
    ids, mask = tok.batch_encode(["a chartreuse dodecahedron moving widdershins"])
    assert ids.shape[0] == 1 and bool(mask[0].any())


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------
def test_batch_shapes(cfg):
    dataset = SyntheticAVDataset(cfg.data, "train")
    batch = collate([dataset[i] for i in range(4)])
    assert batch["video"].shape == (4, 3, cfg.data.num_frames,
                                    cfg.data.resolution, cfg.data.resolution)
    assert batch["mel"].shape == (4, cfg.data.n_mels, cfg.data.mel_frames)
    assert batch["waveform"].shape == (4, cfg.data.audio_samples)
    assert batch["video"].abs().max() <= 1.0 and batch["mel"].abs().max() <= 1.0


def test_generation_is_deterministic(cfg):
    a = SyntheticAVDataset(cfg.data, "train")[5]
    b = SyntheticAVDataset(cfg.data, "train")[5]
    assert torch.equal(a["video"], b["video"]) and a["caption"] == b["caption"]


def test_train_and_val_differ(cfg):
    train = SyntheticAVDataset(cfg.data, "train")[0]
    val = SyntheticAVDataset(cfg.data, "val")[0]
    assert not torch.equal(train["video"], val["video"])


def test_caption_direction_matches_pixels(cfg):
    """The captioned direction must be the dominant axis of early motion.

    The generator gives each clip runway in its captioned direction; without
    that a clip can bounce on frame 1 and spend most of its frames moving the
    opposite way, which would make the prompt-alignment reward incoherent.
    """
    dataset = SyntheticAVDataset(cfg.data, "train")
    window = 5
    correct = 0
    total = 60
    for i in range(total):
        sample = dataset[i]
        words = sample["caption"].split()
        colour, direction = words[1], words[-1]
        weight = color_mass(sample["video"].unsqueeze(0), COLORS[colour])
        cx, cy, _ = centroid_trajectory(weight)
        dx = float(cx[0, window] - cx[0, 0])
        dy = float(cy[0, window] - cy[0, 0])
        predicted = (("right" if dx > 0 else "left") if abs(dx) > abs(dy)
                     else ("down" if dy > 0 else "up"))
        correct += predicted == direction
    assert correct >= int(0.95 * total), "%d/%d captions matched pixel motion" % (correct, total)


def test_every_clip_has_an_audio_event(cfg):
    dataset = SyntheticAVDataset(cfg.data, "train")
    bounces = [dataset[i]["meta"]["num_bounces"] for i in range(40)]
    assert min(bounces) >= 1


def test_curation_selects_higher_quality(cfg):
    kept = curate_indices(cfg.data, 0.6)
    dataset = SyntheticAVDataset(cfg.data, "train")
    kept_quality = np.mean([dataset.spec_for(i).quality for i in kept])
    all_quality = np.mean([dataset.spec_for(i).quality for i in dataset.indices])
    assert 0 < len(kept) < cfg.data.num_train
    assert kept_quality > all_quality


# --------------------------------------------------------------------------
# audio front-end
# --------------------------------------------------------------------------
def test_mel_roundtrip_preserves_pitch(cfg):
    """Griffin-Lim is lossy, but the spectral peak must survive it."""
    mel = MelSpectrogram(cfg.data.sample_rate, cfg.data.n_fft, cfg.data.hop_length,
                         cfg.data.n_mels, cfg.data.mel_frames)
    t = torch.arange(cfg.data.audio_samples) / cfg.data.sample_rate
    tone = torch.sin(2 * np.pi * 392.0 * t)

    spec = mel.forward(tone)
    assert spec.shape == (cfg.data.n_mels, cfg.data.mel_frames)

    recovered = mel.inverse(spec, num_samples=cfg.data.audio_samples, n_iter=32)
    assert recovered.shape == (cfg.data.audio_samples,)
    assert int(mel.forward(recovered).mean(-1).argmax()) == int(spec.mean(-1).argmax())
