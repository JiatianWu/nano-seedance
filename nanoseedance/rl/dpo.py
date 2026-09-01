"""Stage 4b: Diffusion-DPO.

The complement to GRPO. Where GRPO does online rollouts and needs a scalar
reward per sample, DPO works offline from *pairs*: for each prompt, generate
several candidates, keep the best and worst, and train the model to prefer the
winner. No sampling inside the training loop, no log-probabilities through the
sampler, and it tolerates a reward model that is only trustworthy as a ranker.

The objective adapts Direct Preference Optimisation to diffusion. For a
preference pair (w, l) the implicit reward is the improvement in denoising
error relative to a frozen reference model, so the loss is::

    -log sigmoid( -beta * [ (err_w - err_w_ref) - (err_l - err_l_ref) ] )

Both members of the pair are noised at the *same* timestep with the *same*
noise, which removes the dominant source of variance -- without that the
timestep difference between the two branches drowns out the preference signal.

    python -m nanoseedance.rl.dpo --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from ..config import Config
from ..data import build_dataloader
from ..data.synthetic import random_prompts
from ..pipeline import SeedancePipeline, load_pipeline
from ..utils import (
    AverageMeter,
    Logger,
    resolve_device,
    save_checkpoint,
    save_contact_sheet,
    set_seed,
)
from .rewards import RewardEnsemble, RewardInput


# --------------------------------------------------------------------------
# preference dataset construction
# --------------------------------------------------------------------------
@torch.no_grad()
def build_preference_pairs(pipe: SeedancePipeline, cfg: Config,
                           rewards: RewardEnsemble, device: torch.device
                           ) -> Dict[str, Any]:
    """Generate candidates per prompt and keep the best/worst as (win, lose)."""
    dpo = cfg.dpo
    num_prompts = max(1, dpo.num_pairs)
    candidates = dpo.candidates_per_prompt

    win_v, lose_v, win_a, lose_a, kept_prompts = [], [], [], [], []
    margins: List[float] = []

    pipe.eval()
    batch_prompts = 4
    prompt_pool = random_prompts(num_prompts, seed=7, style="detailed")

    for start in range(0, num_prompts, batch_prompts):
        chunk = prompt_pool[start:start + batch_prompts]
        repeated = [p for p in chunk for _ in range(candidates)]

        out = pipe.generate(repeated, num_steps=dpo.sampling_steps,
                            guidance_scale=dpo.guidance_scale, device=device)
        reward_input = RewardInput(
            video=out["video"], captions=repeated, mel=out.get("mel"),
            sample_rate=cfg.data.sample_rate, fps=cfg.data.fps,
            hop_length=cfg.data.hop_length,
        )
        total, _ = rewards(reward_input)

        grouped = total.view(len(chunk), candidates)
        best = grouped.argmax(dim=1)
        worst = grouped.argmin(dim=1)

        for i, prompt in enumerate(chunk):
            b, w = int(best[i]), int(worst[i])
            if b == w or float(grouped[i, b] - grouped[i, w]) < 1e-4:
                continue  # no usable preference signal for this prompt
            offset = i * candidates
            win_v.append(out["video_latent"][offset + b])
            lose_v.append(out["video_latent"][offset + w])
            if out.get("audio_latent") is not None:
                win_a.append(out["audio_latent"][offset + b])
                lose_a.append(out["audio_latent"][offset + w])
            kept_prompts.append(prompt)
            margins.append(float(grouped[i, b] - grouped[i, w]))

    if not kept_prompts:
        raise RuntimeError(
            "no preference pairs had a reward margin; the policy may be "
            "collapsed, or candidates_per_prompt is too small"
        )

    pairs: Dict[str, Any] = {
        "prompts": kept_prompts,
        "win_video": torch.stack(win_v),
        "lose_video": torch.stack(lose_v),
        "win_audio": torch.stack(win_a) if win_a else None,
        "lose_audio": torch.stack(lose_a) if lose_a else None,
        "mean_margin": sum(margins) / len(margins),
    }
    return pairs


# --------------------------------------------------------------------------
# loss
# --------------------------------------------------------------------------
def _denoise_error(pipe: SeedancePipeline, latent_v: torch.Tensor,
                   latent_a: Optional[torch.Tensor], noise_v: torch.Tensor,
                   noise_a: Optional[torch.Tensor], t: torch.Tensor,
                   prompts: List[str], device: torch.device) -> torch.Tensor:
    """Per-sample flow-matching error at a fixed (t, noise)."""
    flow = pipe.flow
    zt_v = flow.add_noise(latent_v, noise_v, t)
    target_v = flow.velocity_target(latent_v, noise_v)

    zt_a = target_a = None
    if latent_a is not None and noise_a is not None:
        zt_a = flow.add_noise(latent_a, noise_a, t)
        target_a = flow.velocity_target(latent_a, noise_a)

    text, pooled, mask = pipe.encode_prompts(prompts, device)
    pred_v, pred_a = pipe.dit(zt_v, t, text, pooled, mask, audio_latent=zt_a)

    error = (pred_v - target_v).pow(2).flatten(1).mean(1)
    if pred_a is not None and target_a is not None:
        error = error + (pred_a - target_a).pow(2).flatten(1).mean(1)
    return error


def dpo_loss(pipe: SeedancePipeline, reference: SeedancePipeline,
             pairs: Dict[str, Any], indices: torch.Tensor, cfg: Config,
             device: torch.device) -> Tuple[torch.Tensor, Dict[str, float]]:
    prompts = [pairs["prompts"][int(i)] for i in indices]
    win_v = pairs["win_video"][indices]
    lose_v = pairs["lose_video"][indices]
    win_a = pairs["win_audio"][indices] if pairs["win_audio"] is not None else None
    lose_a = pairs["lose_audio"][indices] if pairs["lose_audio"] is not None else None

    # Shared timestep and shared noise across the pair: the comparison is only
    # meaningful if both sides face an identical denoising problem.
    batch = win_v.shape[0]
    t = pipe.flow.sample_timesteps(batch, device)
    noise_v = torch.randn_like(win_v)
    noise_a = torch.randn_like(win_a) if win_a is not None else None

    err_win = _denoise_error(pipe, win_v, win_a, noise_v, noise_a, t, prompts, device)
    err_lose = _denoise_error(pipe, lose_v, lose_a, noise_v, noise_a, t, prompts, device)
    with torch.no_grad():
        ref_win = _denoise_error(reference, win_v, win_a, noise_v, noise_a, t, prompts, device)
        ref_lose = _denoise_error(reference, lose_v, lose_a, noise_v, noise_a, t, prompts, device)

    # Lower error on the winner (relative to reference) => positive logit.
    logits = -cfg.dpo.beta * ((err_win - ref_win) - (err_lose - ref_lose))
    loss = -F.logsigmoid(logits).mean()

    return loss, {
        "loss": float(loss.detach()),
        "accuracy": float((logits > 0).float().mean()),
        "margin": float((err_lose - err_win).mean().detach()),
    }


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> str:
    parser = argparse.ArgumentParser(description="Diffusion-DPO fine-tuning")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--init", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    dpo = cfg.dpo
    out_dir = dpo.out_dir
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.train.seed + 3)

    device = resolve_device(cfg.train.device)
    logger = Logger(out_dir, "dpo")
    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    init_from = args.init or dpo.init_from

    pipe = load_pipeline(cfg, vae_ckpt, init_from, device)
    reference = load_pipeline(cfg, vae_ckpt, init_from, device)
    reference.generator_module().requires_grad_(False)
    reference.eval()

    rewards = RewardEnsemble(cfg.reward, cfg.data.sample_rate, cfg.data.n_mels)
    calib = next(iter(build_dataloader(cfg, "val", batch_size=16, shuffle=False)))
    rewards.calibrate(calib["video"].to(device))

    print("building preference pairs: %d prompts x %d candidates"
          % (dpo.num_pairs, dpo.candidates_per_prompt))
    pairs = build_preference_pairs(pipe, cfg, rewards, device)
    print("kept %d pairs (mean reward margin %.4f)"
          % (len(pairs["prompts"]), pairs["mean_margin"]))

    save_contact_sheet(os.path.join(out_dir, "samples", "pair_winner.png"),
                       pipe.video_vae.decode_from_latent(pairs["win_video"][:1])[0].cpu())
    save_contact_sheet(os.path.join(out_dir, "samples", "pair_loser.png"),
                       pipe.video_vae.decode_from_latent(pairs["lose_video"][:1])[0].cpu())

    pipe.dit.train()
    pipe.text_encoder.train()
    generator = pipe.generator_module()
    optimizer = torch.optim.AdamW(generator.parameters(), lr=dpo.lr, betas=cfg.train.betas)

    num_pairs = len(pairs["prompts"])
    meters = {k: AverageMeter() for k in ("loss", "accuracy", "margin")}
    for step in range(dpo.steps):
        indices = torch.randint(0, num_pairs, (min(dpo.batch_size, num_pairs),),
                                device=device)
        loss, stats = dpo_loss(pipe, reference, pairs, indices, cfg, device)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), cfg.train.grad_clip)
        optimizer.step()

        for key, value in stats.items():
            meters[key].add(value)
        if (step + 1) % max(1, cfg.train.log_every // 5) == 0:
            logger.log(step + 1, **{k: m.mean for k, m in meters.items()})
            for m in meters.values():
                m.reset()

    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    save_checkpoint(ckpt_path, model=pipe.dit_state(), ema=None,
                    config=cfg.to_dict(), stage="dpo")
    print("saved %s" % ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    main()
