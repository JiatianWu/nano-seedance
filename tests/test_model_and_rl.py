"""DiT wiring, reward discrimination, and RL consistency."""

from __future__ import annotations

import pytest
import torch

from nanoseedance.config import Config
from nanoseedance.data import SyntheticAVDataset, collate
from nanoseedance.data.synthetic import COLORS, DIRECTION_VECTORS
from nanoseedance.flow import gaussian_log_prob
from nanoseedance.models.dit import (
    patchify_audio,
    patchify_video,
    unpatchify_audio,
    unpatchify_video,
)
from nanoseedance.pipeline import SeedancePipeline
from nanoseedance.rl.grpo import policy_logprob
from nanoseedance.rl.rewards import RewardEnsemble, RewardInput


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/smoke.yaml")


@pytest.fixture(scope="module")
def toy_cfg() -> Config:
    return Config.load("configs/toy.yaml")


# --------------------------------------------------------------------------
# patchify
# --------------------------------------------------------------------------
def test_patchify_roundtrip(cfg):
    x = torch.randn(2, *cfg.video_latent_shape)
    tokens, grid = patchify_video(x, cfg.dit.patch_t, cfg.dit.patch_hw)
    back = unpatchify_video(tokens, grid, cfg.video_vae.latent_channels,
                            cfg.dit.patch_t, cfg.dit.patch_hw)
    assert torch.equal(x, back)


def test_patchify_audio_roundtrip(cfg):
    x = torch.randn(2, *cfg.audio_latent_shape)
    tokens, _ = patchify_audio(x, cfg.dit.patch_audio_t)
    back = unpatchify_audio(tokens, cfg.audio_vae.latent_channels, cfg.dit.patch_audio_t)
    assert torch.equal(x, back)


# --------------------------------------------------------------------------
# DiT conditioning paths
# --------------------------------------------------------------------------
def test_dit_forward_paths(cfg):
    pipe = SeedancePipeline.build(cfg).eval()
    device = torch.device("cpu")
    captions = ["a red circle moving left", "a cyan square moving up"]
    text, pooled, mask = pipe.encode_prompts(captions, device)
    zv = torch.randn(2, *cfg.video_latent_shape)
    za = torch.randn(2, *cfg.audio_latent_shape)
    t = torch.rand(2)

    v, a = pipe.dit(zv, t, text, pooled, mask, audio_latent=za)
    assert v.shape == zv.shape and a.shape == za.shape

    # audio branch is optional at call time
    v_only, none_audio = pipe.dit(zv, t, text, pooled, mask, audio_latent=None)
    assert v_only.shape == zv.shape and none_audio is None

    # decoupled timesteps (clean audio, noisy video)
    v2, _ = pipe.dit(zv, t, text, pooled, mask, audio_latent=za, t_audio=torch.zeros(2))
    assert v2.shape == zv.shape

    # conditioning dropout changes the prediction (CFG needs this to differ)
    dropped, _ = pipe.dit(zv, t, text, pooled, mask, audio_latent=za,
                          drop_cond=torch.ones(2, dtype=torch.bool))
    assert not torch.allclose(v, dropped)


def test_bridge_couples_modalities(cfg):
    """Changing only the audio latent must change the video prediction.

    If it does not, the cross-modal bridge is not wired up and the model is
    two independent single-modality DiTs.
    """
    pipe = SeedancePipeline.build(cfg).eval()
    device = torch.device("cpu")
    text, pooled, mask = pipe.encode_prompts(["a red circle moving left"], device)
    zv = torch.randn(1, *cfg.video_latent_shape)
    t = torch.rand(1)

    with torch.no_grad():
        v1, _ = pipe.dit(zv, t, text, pooled, mask,
                         audio_latent=torch.zeros(1, *cfg.audio_latent_shape))
        v2, _ = pipe.dit(zv, t, text, pooled, mask,
                         audio_latent=torch.randn(1, *cfg.audio_latent_shape))
    assert not torch.allclose(v1, v2, atol=1e-6)


# --------------------------------------------------------------------------
# rewards
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def scored(toy_cfg):
    dataset = SyntheticAVDataset(toy_cfg.data, "train")
    batch = collate([dataset[i] for i in range(24)])
    ensemble = RewardEnsemble(toy_cfg.reward, toy_cfg.data.sample_rate, toy_cfg.data.n_mels)
    ensemble.calibrate(batch["video"])

    def score(video, captions, mel):
        _, breakdown = ensemble(RewardInput(
            video=video, captions=captions, mel=mel,
            sample_rate=toy_cfg.data.sample_rate, fps=toy_cfg.data.fps,
            hop_length=toy_cfg.data.hop_length,
        ))
        return {k: float(v.mean()) for k, v in breakdown.items()}

    return batch, score


def test_prompt_alignment_detects_wrong_caption(scored):
    batch, score = scored
    matched = score(batch["video"], batch["caption"], batch["mel"])
    shuffled = score(batch["video"], batch["caption"][12:] + batch["caption"][:12],
                     batch["mel"])
    assert matched["prompt_align"] > 3 * shuffled["prompt_align"]


def test_av_sync_detects_mismatched_audio(scored):
    batch, score = scored
    matched = score(batch["video"], batch["caption"], batch["mel"])
    torch.manual_seed(0)
    shuffled = score(batch["video"], batch["caption"],
                     batch["mel"][torch.randperm(batch["mel"].shape[0])])
    assert matched["av_sync"] > shuffled["av_sync"] + 0.1


def test_audio_prompt_detects_wrong_pitch(scored):
    batch, score = scored
    matched = score(batch["video"], batch["caption"], batch["mel"])
    noise = score(batch["video"], batch["caption"], torch.rand_like(batch["mel"]) * 2 - 1)
    assert matched["audio_prompt"] > 0.5
    assert noise["audio_prompt"] < 0.2


def test_aesthetic_penalises_noise(scored):
    batch, score = scored
    matched = score(batch["video"], batch["caption"], batch["mel"])
    noisy = score(torch.rand_like(batch["video"]) * 2 - 1, batch["caption"], batch["mel"])
    assert matched["aesthetic"] > noisy["aesthetic"] + 0.3


def test_total_reward_ranks_real_data_highest(scored):
    batch, score = scored
    matched = score(batch["video"], batch["caption"], batch["mel"])
    blank = score(torch.zeros_like(batch["video"]), batch["caption"], batch["mel"])
    assert matched["total"] > blank["total"]


# --------------------------------------------------------------------------
# GRPO consistency
# --------------------------------------------------------------------------
def test_grpo_importance_ratio_starts_at_one(cfg):
    """Recomputing the behaviour policy's log-prob must reproduce it exactly.

    A mismatch here means the PPO ratio is wrong on the very first update,
    which silently destroys the policy gradient.
    """
    torch.manual_seed(0)
    device = torch.device("cpu")
    pipe = SeedancePipeline.build(cfg).to(device).eval()
    pipe.freeze_vaes()
    prompts = ["a red circle moving left", "a cyan square moving up"]

    _, _, traj = pipe.rollout(prompts, num_steps=3, guidance_scale=2.0,
                              eta=0.7, device=device)

    for step in range(2):  # skip the final step where sigma == 0
        recorded = (gaussian_log_prob(traj["video_next"][:, step],
                                      traj["video_mean"][:, step],
                                      traj["video_std"][:, step])
                    + gaussian_log_prob(traj["audio_next"][:, step],
                                        traj["audio_mean"][:, step],
                                        traj["audio_std"][:, step]))
        recomputed = policy_logprob(
            pipe, prompts,
            traj["video_prev"][:, step], traj["audio_prev"][:, step],
            traj["video_next"][:, step], traj["audio_next"][:, step],
            traj["t"][:, step], traj["s"][:, step],
            traj["video_std"][:, step], traj["audio_std"][:, step],
            guidance_scale=2.0, device=device,
        )
        ratio = torch.exp(recomputed - recorded)
        assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-3), \
            "importance ratio at step %d is %s, expected 1" % (step, ratio.tolist())


def test_final_sde_step_is_deterministic(cfg):
    """The last step lands on t=0 with sigma=0 and must be excluded from the loss."""
    device = torch.device("cpu")
    pipe = SeedancePipeline.build(cfg).to(device).eval()
    _, _, traj = pipe.rollout(["a red circle moving left"], num_steps=3,
                              guidance_scale=1.0, eta=0.7, device=device)
    std_per_step = traj["video_std"].flatten(2).amax(-1)[0]
    assert float(std_per_step[-1]) == 0.0
    assert all(float(s) > 0 for s in std_per_step[:-1])
