"""Configuration objects for nano-seedance.

Configs are plain dataclasses so they can be constructed in Python, and are
also loadable from YAML with dotted-key command line overrides:

    cfg = Config.load("configs/toy.yaml", ["train.batch_size=8", "dit.depth=4"])
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _log2_int(value: int, name: str) -> int:
    exponent = int(round(math.log2(value)))
    if 2 ** exponent != value:
        raise ValueError("%s must be a power of two, got %d" % (name, value))
    return exponent


@dataclass
class DataConfig:
    """Video/audio/caption sample geometry.

    ``num_frames`` must be ``1 + k * temporal_compression`` so the causal VAE
    maps it to ``1 + k`` latent frames, with frame 0 encoded on its own. That
    is what lets a single model treat an image as a 1-frame video.
    """

    resolution: int = 32
    num_frames: int = 17
    fps: int = 8

    sample_rate: int = 8000
    n_fft: int = 256
    hop_length: int = 128
    n_mels: int = 32
    mel_frames: int = 128

    num_train: int = 512
    num_val: int = 16
    seed: int = 0

    # Real-video training. When set, data.video_dataset is used instead of the
    # procedural generator. See README "Training on real video".
    root: Optional[str] = None
    loader: str = "synthetic"  # synthetic | folder | webdataset

    @property
    def audio_samples(self) -> int:
        return int(self.sample_rate * self.num_frames / self.fps)


@dataclass
class VideoVAEConfig:
    in_channels: int = 3
    latent_channels: int = 4
    base_channels: int = 32
    max_channel_mult: int = 4
    spatial_compression: int = 4
    temporal_compression: int = 4
    kl_weight: float = 1e-6

    def __post_init__(self) -> None:
        spatial_stages = _log2_int(self.spatial_compression, "spatial_compression")
        temporal_stages = _log2_int(self.temporal_compression, "temporal_compression")
        if temporal_stages > spatial_stages:
            raise ValueError(
                "temporal_compression (%d) cannot exceed spatial_compression (%d); "
                "each temporal halving rides on a spatial downsampling stage"
                % (self.temporal_compression, self.spatial_compression)
            )
        self.spatial_stages = spatial_stages
        self.temporal_stages = temporal_stages

    def channels_per_stage(self) -> List[int]:
        return [
            self.base_channels * min(2 ** i, self.max_channel_mult)
            for i in range(self.spatial_stages + 1)
        ]


@dataclass
class AudioVAEConfig:
    latent_channels: int = 8
    base_channels: int = 64
    temporal_compression: int = 4
    kl_weight: float = 1e-6

    def __post_init__(self) -> None:
        self.temporal_stages = _log2_int(self.temporal_compression, "temporal_compression")


@dataclass
class DiTConfig:
    dim: int = 192
    depth: int = 6
    num_heads: int = 3
    mlp_ratio: float = 4.0

    # Patchification. Video latents are (C, T, H, W); audio latents are (C, T).
    patch_t: int = 1
    patch_hw: int = 2
    patch_audio_t: int = 2

    text_dim: int = 192
    text_depth: int = 2
    text_heads: int = 3
    max_text_len: int = 24
    text_encoder: str = "tiny"  # tiny | hf
    hf_text_model: str = "google/flan-t5-small"

    # Dual-branch wiring. The AV bridge is the "2.0" feature: audio and video
    # tokens exchange information every `bridge_every` blocks.
    enable_audio: bool = True
    bridge_every: int = 2

    cond_dropout: float = 0.1  # for classifier-free guidance
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.dim % self.num_heads != 0:
            raise ValueError(
                "dim (%d) must be divisible by num_heads (%d)" % (self.dim, self.num_heads)
            )


@dataclass
class FlowConfig:
    """Rectified-flow / flow-matching schedule.

    Convention (SD3-style): ``z_t = (1 - t) * x + t * noise`` with ``t=0`` at
    data and ``t=1`` at noise, and the network predicts velocity
    ``v = noise - x``. Sampling integrates from t=1 down to t=0.
    """

    # Logit-normal timestep sampling concentrates training on mid timesteps.
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # Resolution-aware timestep shift: more tokens => more noise needed to
    # destroy signal => schedule shifted toward high t.
    base_shift: float = 0.5
    max_shift: float = 1.15
    base_tokens: int = 64
    max_tokens: int = 4096

    num_sampling_steps: int = 24
    guidance_scale: float = 4.0
    audio_guidance_scale: float = 4.0

    # Fraction of training steps where the two branches get independent
    # timesteps. Training only on a shared timestep would leave the model
    # undefined on inputs like "clean video, noisy audio", which is exactly
    # what video-to-audio dubbing needs at inference.
    decoupled_timestep_prob: float = 0.25


@dataclass
class TrainConfig:
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    steps: int = 400
    # The autoencoder converges long before the DiT does; give it its own
    # budget rather than paying for DiT-length training twice.
    vae_steps: Optional[int] = None
    warmup_steps: int = 50
    log_every: int = 20
    eval_every: int = 200
    ckpt_every: int = 200
    ema_decay: float = 0.999
    use_ema: bool = True
    num_workers: int = 0
    device: str = "auto"
    seed: int = 0
    out_dir: str = "runs/toy"
    amp: bool = False


@dataclass
class SFTConfig:
    """Supervised fine-tuning on a curated, higher-quality subset."""

    init_from: str = "runs/toy/dit/ckpt_last.pt"
    steps: int = 150
    lr: float = 5e-5
    batch_size: int = 4
    # Curation: keep only the top fraction by data-quality score.
    quality_quantile: float = 0.5
    caption_style: str = "detailed"  # synthetic captions get richer at SFT time
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_targets: Tuple[str, ...] = ("qkv", "proj", "to_q", "to_kv", "fc1", "fc2")
    out_dir: str = "runs/toy/sft"


@dataclass
class RewardConfig:
    """Weights for the reward ensemble used by both RL algorithms."""

    av_sync: float = 1.0
    prompt_align: float = 1.0
    audio_prompt: float = 1.0
    aesthetic: float = 0.25
    normalize: bool = True


@dataclass
class GRPOConfig:
    """Flow-GRPO: group-relative policy optimisation over an SDE sampler."""

    init_from: str = "runs/toy/sft/ckpt_last.pt"
    steps: int = 40
    lr: float = 1e-5
    num_prompts: int = 4          # prompts per iteration
    group_size: int = 4           # rollouts per prompt (the GRPO "group")
    sampling_steps: int = 10      # denoising steps per rollout
    noise_level: float = 0.7      # SDE noise injected into the ODE path
    clip_range: float = 0.2
    kl_coef: float = 0.02
    inner_epochs: int = 1
    minibatch_size: int = 8       # (rollout, timestep) pairs per grad step
    adv_eps: float = 1e-4
    guidance_scale: float = 2.0
    out_dir: str = "runs/toy/grpo"


@dataclass
class DPOConfig:
    """Diffusion-DPO on preference pairs, scored offline by the reward model."""

    init_from: str = "runs/toy/sft/ckpt_last.pt"
    steps: int = 60
    lr: float = 1e-6
    batch_size: int = 2
    beta: float = 500.0
    num_pairs: int = 32
    candidates_per_prompt: int = 4
    sampling_steps: int = 10
    guidance_scale: float = 2.0
    out_dir: str = "runs/toy/dpo"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    video_vae: VideoVAEConfig = field(default_factory=VideoVAEConfig)
    audio_vae: AudioVAEConfig = field(default_factory=AudioVAEConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def __post_init__(self) -> None:
        tc = self.video_vae.temporal_compression
        if (self.data.num_frames - 1) % tc != 0:
            raise ValueError(
                "data.num_frames must be 1 + k*%d for the causal VAE, got %d"
                % (tc, self.data.num_frames)
            )
        if self.data.resolution % self.video_vae.spatial_compression != 0:
            raise ValueError(
                "data.resolution (%d) must be divisible by spatial_compression (%d)"
                % (self.data.resolution, self.video_vae.spatial_compression)
            )
        if self.data.mel_frames % self.audio_vae.temporal_compression != 0:
            raise ValueError(
                "data.mel_frames (%d) must be divisible by audio temporal_compression (%d)"
                % (self.data.mel_frames, self.audio_vae.temporal_compression)
            )

    # -- derived latent geometry ------------------------------------------
    @property
    def latent_frames(self) -> int:
        return 1 + (self.data.num_frames - 1) // self.video_vae.temporal_compression

    @property
    def latent_size(self) -> int:
        return self.data.resolution // self.video_vae.spatial_compression

    @property
    def audio_latent_frames(self) -> int:
        return self.data.mel_frames // self.audio_vae.temporal_compression

    @property
    def video_latent_shape(self) -> Tuple[int, int, int, int]:
        return (
            self.video_vae.latent_channels,
            self.latent_frames,
            self.latent_size,
            self.latent_size,
        )

    @property
    def audio_latent_shape(self) -> Tuple[int, int]:
        return (self.audio_vae.latent_channels, self.audio_latent_frames)

    @property
    def num_video_tokens(self) -> int:
        t = self.latent_frames // self.dit.patch_t
        s = self.latent_size // self.dit.patch_hw
        return t * s * s

    # -- (de)serialisation --------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Optional[Dict[str, Any]]) -> "Config":
        raw = dict(raw or {})
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            section = raw.pop(f.name, None)
            if section is None:
                continue
            if not isinstance(section, dict):
                raise ValueError("config section '%s' must be a mapping" % f.name)
            kwargs[f.name] = _build_section(f, section)
        if raw:
            raise ValueError("unknown config sections: %s" % sorted(raw))
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Optional[str] = None, overrides: Optional[List[str]] = None) -> "Config":
        raw: Dict[str, Any] = {}
        if path:
            with open(path, "r") as fh:
                raw = yaml.safe_load(fh) or {}
        for override in overrides or []:
            if "=" not in override:
                raise ValueError("override must look like section.key=value, got %r" % override)
            key, value = override.split("=", 1)
            _assign_dotted(raw, key.strip(), yaml.safe_load(value))
        return cls.from_dict(raw)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)


_SECTION_TYPES = {
    "data": DataConfig,
    "video_vae": VideoVAEConfig,
    "audio_vae": AudioVAEConfig,
    "dit": DiTConfig,
    "flow": FlowConfig,
    "train": TrainConfig,
    "sft": SFTConfig,
    "reward": RewardConfig,
    "grpo": GRPOConfig,
    "dpo": DPOConfig,
}


def _build_section(f: "dataclasses.Field", values: Dict[str, Any]) -> Any:
    section_cls = _SECTION_TYPES[f.name]
    valid = {sf.name for sf in dataclasses.fields(section_cls)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError("unknown keys in '%s': %s" % (f.name, sorted(unknown)))
    coerced = {k: (tuple(v) if isinstance(v, list) else v) for k, v in values.items()}
    return section_cls(**coerced)


def _assign_dotted(raw: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value
