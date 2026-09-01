"""Score one or more checkpoints with the reward suite.

This is how you tell whether SFT and RL actually did anything:

    python -m nanoseedance.evaluate --config configs/toy.yaml \
        --ckpt runs/toy/dit/ckpt_last.pt runs/toy/sft/ckpt_last.pt \
               runs/toy/grpo/ckpt_last.pt runs/toy/dpo/ckpt_last.pt

All checkpoints are scored on the same prompts with the same seed, so the
comparison is paired.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import torch

from .config import Config
from .data import build_dataloader
from .data.synthetic import random_prompts
from .pipeline import load_pipeline
from .rl.rewards import RewardEnsemble, RewardInput
from .utils import resolve_device, set_seed


@torch.no_grad()
def score_checkpoint(cfg: Config, vae_ckpt: str, dit_ckpt: Optional[str],
                     prompts: List[str], rewards: RewardEnsemble,
                     device: torch.device, batch_size: int = 8,
                     steps: Optional[int] = None,
                     guidance: Optional[float] = None) -> Dict[str, float]:
    pipe = load_pipeline(cfg, vae_ckpt, dit_ckpt, device)
    totals: Dict[str, List[torch.Tensor]] = {}

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        out = pipe.generate(chunk, num_steps=steps, guidance_scale=guidance,
                            device=device)
        sample = RewardInput(
            video=out["video"], captions=chunk, mel=out.get("mel"),
            sample_rate=cfg.data.sample_rate, fps=cfg.data.fps,
            hop_length=cfg.data.hop_length,
        )
        _, breakdown = rewards(sample)
        for key, value in breakdown.items():
            totals.setdefault(key, []).append(value.detach().cpu())

    return {k: float(torch.cat(v).mean()) for k, v in totals.items()}


def main(argv: Optional[list] = None) -> Dict[str, Dict[str, float]]:
    parser = argparse.ArgumentParser(description="Reward-score checkpoints")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--ckpt", nargs="*", default=None)
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    device = resolve_device(cfg.train.device)
    set_seed(args.seed)

    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    checkpoints = args.ckpt or [os.path.join(cfg.train.out_dir, "dit", "ckpt_last.pt")]
    prompts = random_prompts(args.num_prompts, seed=args.seed, style="detailed")

    rewards = RewardEnsemble(cfg.reward, cfg.data.sample_rate, cfg.data.n_mels)
    val = next(iter(build_dataloader(cfg, "val", batch_size=16, shuffle=False)))
    rewards.calibrate(val["video"].to(device))

    results: Dict[str, Dict[str, float]] = {}

    # Real data is the ceiling: it tells you how much of the gap is left.
    real = RewardInput(video=val["video"].to(device), captions=val["caption"],
                       mel=val["mel"].to(device), sample_rate=cfg.data.sample_rate,
                       fps=cfg.data.fps, hop_length=cfg.data.hop_length)
    _, real_breakdown = rewards(real)
    results["(real data)"] = {k: float(v.mean()) for k, v in real_breakdown.items()}

    for path in checkpoints:
        if not os.path.exists(path):
            print("skipping missing checkpoint %s" % path)
            continue
        set_seed(args.seed)
        label = os.path.basename(os.path.dirname(os.path.dirname(path))) + "/" + \
            os.path.basename(os.path.dirname(path))
        results[label] = score_checkpoint(cfg, vae_ckpt, path, prompts, rewards,
                                          device, steps=args.steps,
                                          guidance=args.guidance)

    keys = ["total", "prompt_align", "av_sync", "audio_prompt", "aesthetic"]
    keys = [k for k in keys if any(k in r for r in results.values())]
    width = max(len(label) for label in results)
    header = "%-*s  " % (width, "checkpoint") + "  ".join("%12s" % k for k in keys)
    print("\n" + header)
    print("-" * len(header))
    for label, scores in results.items():
        row = "%-*s  " % (width, label) + "  ".join("%12.4f" % scores.get(k, float("nan"))
                                                    for k in keys)
        print(row)
    return results


if __name__ == "__main__":
    main()
