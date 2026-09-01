"""Render the same prompts from several checkpoints with the eval sampler.

Rollout contact sheets are drawn from the noisy SDE path, so they cannot be
compared across checkpoints. This uses the deterministic sampler at evaluation
settings and reports the statistics the aesthetic reward is built from.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoseedance.config import Config
from nanoseedance.data import build_dataloader
from nanoseedance.pipeline import load_pipeline
from nanoseedance.rl.rewards import RewardEnsemble, RewardInput, _total_variation
from nanoseedance.utils import resolve_device, save_contact_sheet, set_seed

PROMPTS = [
    "a red circle moving left",
    "a cyan square moving up",
]

cfg = Config.load("configs/toy.yaml")
device = resolve_device(cfg.train.device)
vae = os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")

rewards = RewardEnsemble(cfg.reward, cfg.data.sample_rate, cfg.data.n_mels)
val = next(iter(build_dataloader(cfg, "val", batch_size=16, shuffle=False)))
real = val["video"].to(device)
rewards.calibrate(real)

print("%-12s %8s %8s %10s %10s" % ("ckpt", "std", "tv", "aesthetic", "total"))
print("%-12s %8.4f %8.4f %10s %10s"
      % ("(real)", float(real.float().flatten(1).std(dim=1).mean()),
         float(_total_variation(real.float()).mean()), "-", "-"))

for name in ["sft", "grpo_scale", "grpo_lr"]:
    ckpt = os.path.join(cfg.train.out_dir, name, "ckpt_last.pt")
    if not os.path.exists(ckpt):
        continue
    pipe = load_pipeline(cfg, vae, ckpt, device)
    pipe.eval()
    set_seed(0)
    with torch.no_grad():
        out = pipe.generate(PROMPTS, device=device)
    video = out["video"].float()
    sample = RewardInput(video=out["video"], captions=PROMPTS, mel=out.get("mel"),
                         sample_rate=cfg.data.sample_rate, fps=cfg.data.fps,
                         hop_length=cfg.data.hop_length)
    total, breakdown = rewards(sample)
    print("%-12s %8.4f %8.4f %10.4f %10.4f"
          % (name, float(video.flatten(1).std(dim=1).mean()),
             float(_total_variation(video).mean()),
             float(breakdown["aesthetic"].mean()), float(total.mean())))
    save_contact_sheet("runs/toy/compare_%s.png" % name, out["video"][0].cpu())
