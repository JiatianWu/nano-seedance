"""Stage 4a: Flow-GRPO.

Group Relative Policy Optimisation adapted to a flow-matching sampler. The
idea in one paragraph:

A deterministic ODE sampler has no policy to optimise -- given the prompt and
the initial noise, the output is fixed, so there is no action distribution and
no log-probability. So we convert the sampler to an SDE with the *same
marginals* (see ``FlowMatching.sde_transition``). Each denoising step then
becomes a Gaussian action with a tractable log-probability, and the whole
rollout is a short trajectory that policy gradient can act on.

GRPO's contribution is that it needs no value network: sample a *group* of
rollouts per prompt and use the group's own mean reward as the baseline. The
advantage is the within-group z-score, which cancels prompt difficulty --
"this sample was better than the other attempts at the same prompt" is exactly
the signal we want, and it costs nothing to compute.

The loss is then PPO's clipped surrogate over (rollout, timestep) pairs, plus a
KL penalty to the frozen reference policy to keep the model from drifting off
the manifold it learned in pretraining.

    python -m nanoseedance.rl.grpo --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config import Config
from ..data import build_dataloader
from ..data.synthetic import random_prompts
from ..flow import FlowMatching, gaussian_log_prob
from ..pipeline import SeedancePipeline, load_pipeline
from ..utils import (
    AverageMeter,
    Logger,
    infinite,
    load_checkpoint,
    resolve_device,
    save_checkpoint,
    save_contact_sheet,
    set_seed,
)
from .rewards import RewardEnsemble, RewardInput


# --------------------------------------------------------------------------
# rollout collection
# --------------------------------------------------------------------------
@torch.no_grad()
def collect_rollouts(pipe: SeedancePipeline, prompts: List[str], group_size: int,
                     cfg: Config, rewards: RewardEnsemble, device: torch.device
                     ) -> Dict[str, Any]:
    """Sample ``group_size`` completions per prompt and score them."""
    grpo = cfg.grpo
    repeated = [p for p in prompts for _ in range(group_size)]

    z_v, z_a, traj = pipe.rollout(repeated, grpo.sampling_steps, grpo.guidance_scale,
                                  grpo.noise_level, device)

    decoded = pipe.decode_latents(z_v, z_a)
    reward_input = RewardInput(
        video=decoded["video"], captions=repeated,
        mel=decoded.get("mel"), sample_rate=cfg.data.sample_rate,
        fps=cfg.data.fps, hop_length=cfg.data.hop_length,
    )
    total, breakdown = rewards(reward_input)

    # -- group-relative advantage: the group mean is the baseline -----------
    grouped = total.view(len(prompts), group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True)
    advantage = ((grouped - mean) / (std + grpo.adv_eps)).view(-1)

    return {
        "prompts": repeated,
        "advantage": advantage,
        "reward": total,
        "breakdown": {k: v.detach().cpu() for k, v in breakdown.items()},
        "traj": traj,
        "video": decoded["video"],
        "reward_std_within_group": float(std.mean()),
    }


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def policy_logprob(pipe: SeedancePipeline, prompts: List[str], z_v: torch.Tensor,
                   z_a: Optional[torch.Tensor], z_v_next: torch.Tensor,
                   z_a_next: Optional[torch.Tensor], t: torch.Tensor, s: torch.Tensor,
                   std_v: torch.Tensor, std_a: Optional[torch.Tensor],
                   guidance_scale: float, device: torch.device) -> torch.Tensor:
    """Log-probability of the recorded transition under the current policy.

    Only the transition *mean* depends on the model -- the std is a fixed
    function of (t, s, eta) -- so this is where all the gradient comes from.
    """
    text, pooled, mask = pipe.encode_prompts(prompts, device)
    v_v, v_a = pipe.velocity(z_v, z_a, t, text, pooled, mask, guidance_scale)

    mean_v = _mean_with_std(z_v, v_v, t, s, std_v)
    logp = gaussian_log_prob(z_v_next, mean_v, std_v)

    if z_a is not None and v_a is not None and z_a_next is not None and std_a is not None:
        mean_a = _mean_with_std(z_a, v_a, t, s, std_a)
        logp = logp + gaussian_log_prob(z_a_next, mean_a, std_a)
    return logp


def _mean_with_std(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor,
                   s: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Transition mean consistent with a pre-recorded noise level."""
    t_b = t.view(-1, *([1] * (z.dim() - 1))).to(z.dtype)
    s_b = s.view(-1, *([1] * (z.dim() - 1))).to(z.dtype)
    x_hat = z - t_b * v
    noise_hat = z + (1.0 - t_b) * v
    keep = torch.sqrt(torch.clamp(s_b ** 2 - std ** 2, min=0.0))
    return (1.0 - s_b) * x_hat + keep * noise_hat


def grpo_step(pipe: SeedancePipeline, reference: Optional[SeedancePipeline],
              rollouts: Dict[str, Any], indices: torch.Tensor, cfg: Config,
              device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    """PPO-clipped surrogate over a minibatch of (rollout, timestep) pairs."""
    grpo = cfg.grpo
    traj = rollouts["traj"]
    steps = traj["video_prev"].shape[1]

    sample_idx = indices // steps
    step_idx = indices % steps

    prompts = [rollouts["prompts"][int(i)] for i in sample_idx]
    z_v = traj["video_prev"][sample_idx, step_idx]
    z_v_next = traj["video_next"][sample_idx, step_idx]
    std_v = traj["video_std"][sample_idx, step_idx]
    t = traj["t"][sample_idx, step_idx]
    s = traj["s"][sample_idx, step_idx]

    z_a = z_a_next = std_a = None
    if traj["audio_prev"] is not None:
        z_a = traj["audio_prev"][sample_idx, step_idx]
        z_a_next = traj["audio_next"][sample_idx, step_idx]
        std_a = traj["audio_std"][sample_idx, step_idx]

    advantage = rollouts["advantage"][sample_idx]
    old_logp = rollouts["old_logprob"][sample_idx, step_idx]

    new_logp = policy_logprob(pipe, prompts, z_v, z_a, z_v_next, z_a_next, t, s,
                              std_v, std_a, grpo.guidance_scale, device)

    ratio = torch.exp(torch.clamp(new_logp - old_logp, -10.0, 10.0))
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - grpo.clip_range, 1 + grpo.clip_range) * advantage
    policy_loss = -torch.min(unclipped, clipped).mean()

    loss = policy_loss
    stats = {
        "policy_loss": float(policy_loss.detach()),
        "ratio": float(ratio.mean().detach()),
        "clip_frac": float(((ratio - 1).abs() > grpo.clip_range).float().mean()),
    }

    if reference is not None and grpo.kl_coef > 0:
        with torch.no_grad():
            ref_logp = policy_logprob(reference, prompts, z_v, z_a, z_v_next, z_a_next,
                                      t, s, std_v, std_a, grpo.guidance_scale, device)
        # Schulman's k3 estimator. The naive (new - ref) difference is a
        # signed, high-variance estimate that is routinely negative, so
        # penalising it needs an abs() that fights the gradient as often as it
        # helps; k3 is non-negative by construction and far lower variance.
        log_ratio = ref_logp - new_logp
        kl = (torch.exp(torch.clamp(log_ratio, max=10.0)) - 1 - log_ratio).mean()
        loss = loss + grpo.kl_coef * kl
        stats["kl"] = float(kl.detach())

    return loss, stats


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
@torch.no_grad()
def heldout_reward(pipe: SeedancePipeline, prompts: List[str], cfg: Config,
                   rewards: RewardEnsemble, device: torch.device,
                   seed: int, batch_size: int = 8) -> Dict[str, float]:
    """Score the live policy on fixed prompts with the deterministic sampler.

    Rollout reward measures the SDE sampling distribution the policy is being
    trained on, which is not the thing anyone ships. This measures the ODE
    sampler at evaluation settings instead. Prompts and noise are pinned, so
    the only thing that differs between two calls is the policy itself.
    """
    rng_state = torch.get_rng_state()
    was_training = pipe.dit.training
    pipe.eval()
    set_seed(seed)

    totals: Dict[str, List[torch.Tensor]] = {}
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        out = pipe.generate(chunk, device=device)
        sample = RewardInput(
            video=out["video"], captions=chunk, mel=out.get("mel"),
            sample_rate=cfg.data.sample_rate, fps=cfg.data.fps,
            hop_length=cfg.data.hop_length,
        )
        _, breakdown = rewards(sample)
        for key, value in breakdown.items():
            totals.setdefault(key, []).append(value.detach().cpu())

    if was_training:
        pipe.dit.train()
        pipe.text_encoder.train()
    torch.set_rng_state(rng_state)
    return {k: float(torch.cat(v).mean()) for k, v in totals.items()}


def main(argv: Optional[list] = None) -> str:
    parser = argparse.ArgumentParser(description="Flow-GRPO fine-tuning")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    grpo = cfg.grpo
    out_dir = grpo.out_dir
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.train.seed + 2)

    device = resolve_device(cfg.train.device)
    logger = Logger(out_dir, "grpo")
    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    init_from = args.init or grpo.init_from

    pipe = load_pipeline(cfg, vae_ckpt, init_from, device)
    pipe.dit.train()
    pipe.text_encoder.train()

    reference = None
    if grpo.kl_coef > 0:
        reference = load_pipeline(cfg, vae_ckpt, init_from, device)
        reference.generator_module().requires_grad_(False)
        reference.eval()

    rewards = RewardEnsemble(cfg.reward, cfg.data.sample_rate, cfg.data.n_mels)
    calib = next(iter(build_dataloader(cfg, "val", batch_size=16, shuffle=False)))
    rewards.calibrate(calib["video"].to(device))

    generator = pipe.generator_module()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=grpo.lr,
                                  betas=cfg.train.betas)

    print("device %s | %d prompts x %d rollouts x %d steps per iteration"
          % (device, grpo.num_prompts, grpo.group_size, grpo.sampling_steps))

    eval_prompts = random_prompts(grpo.eval_prompts, seed=grpo.eval_seed)
    eval_curve: List[Dict[str, float]] = []
    if grpo.eval_every:
        base = heldout_reward(pipe, eval_prompts, cfg, rewards, device, grpo.eval_seed)
        eval_curve.append({"iteration": 0, **base})
        print("[heldout] iter    0 | total %.4f" % base["total"])

    history: List[float] = []
    for iteration in range(grpo.steps):
        prompts = random_prompts(grpo.num_prompts, seed=1000 + iteration)

        pipe.eval()
        rollouts = collect_rollouts(pipe, prompts, grpo.group_size, cfg, rewards, device)
        pipe.dit.train()
        pipe.text_encoder.train()

        # Behaviour-policy log-probs, recorded once at collection time.
        traj = rollouts["traj"]
        with torch.no_grad():
            logp = gaussian_log_prob(
                traj["video_next"].flatten(0, 1),
                traj["video_mean"].flatten(0, 1),
                traj["video_std"].flatten(0, 1),
            )
            if traj["audio_prev"] is not None:
                logp = logp + gaussian_log_prob(
                    traj["audio_next"].flatten(0, 1),
                    traj["audio_mean"].flatten(0, 1),
                    traj["audio_std"].flatten(0, 1),
                )
        num_samples, steps = traj["video_prev"].shape[:2]
        rollouts["old_logprob"] = logp.view(num_samples, steps)

        # The final step lands exactly on t=0 where sigma=0: a deterministic
        # transition carries no policy gradient, so drop it.
        # std is stored broadcast-shaped (N, steps, 1, 1, 1, 1); reduce the
        # trailing singleton dims only, keeping one flag per (sample, step).
        valid = traj["video_std"].flatten(2).amax(-1) > 1e-6
        pair_index = torch.nonzero(valid.flatten(), as_tuple=False).flatten()

        meters = {k: AverageMeter() for k in ("policy_loss", "ratio", "clip_frac", "kl")}
        for _ in range(grpo.inner_epochs):
            order = pair_index[torch.randperm(pair_index.numel(), device=pair_index.device)]
            for start in range(0, order.numel(), grpo.minibatch_size):
                batch_idx = order[start:start + grpo.minibatch_size]
                if batch_idx.numel() == 0:
                    continue
                loss, stats = grpo_step(pipe, reference, rollouts, batch_idx, cfg, device)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.train.grad_clip)
                optimizer.step()
                for key, value in stats.items():
                    meters[key].add(value)

        mean_reward = float(rollouts["reward"].mean())
        history.append(mean_reward)
        logger.log(
            iteration + 1,
            reward=mean_reward,
            reward_spread=rollouts["reward_std_within_group"],
            **{k: m.mean for k, m in meters.items() if m.count},
            **{k: float(v.mean()) for k, v in rollouts["breakdown"].items() if k != "total"},
        )

        if grpo.eval_every and (iteration + 1) % grpo.eval_every == 0:
            scores = heldout_reward(pipe, eval_prompts, cfg, rewards, device,
                                    grpo.eval_seed)
            eval_curve.append({"iteration": iteration + 1, **scores})
            print("[heldout] iter %4d | total %.4f | %s"
                  % (iteration + 1, scores["total"],
                     "  ".join("%s %.4f" % (k, v) for k, v in scores.items()
                               if k != "total")))

        if (iteration + 1) % max(1, grpo.steps // 4) == 0:
            best = int(rollouts["reward"].argmax())
            save_contact_sheet(
                os.path.join(out_dir, "samples", "iter%03d_best.png" % (iteration + 1)),
                rollouts["video"][best].cpu(),
            )

    if history:
        first = sum(history[: max(1, len(history) // 4)]) / max(1, len(history) // 4)
        last = sum(history[-max(1, len(history) // 4):]) / max(1, len(history) // 4)
        print("mean reward: %.4f (first quarter) -> %.4f (last quarter)" % (first, last))

    if len(eval_curve) > 1:
        print("held-out total: %.4f (start) -> %.4f (end)"
              % (eval_curve[0]["total"], eval_curve[-1]["total"]))
        with open(os.path.join(out_dir, "heldout.json"), "w") as fh:
            json.dump(eval_curve, fh, indent=2)

    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    save_checkpoint(ckpt_path, model=pipe.dit_state(), ema=None,
                    config=cfg.to_dict(), stage="grpo", reward_history=history)
    print("saved %s" % ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    main()
