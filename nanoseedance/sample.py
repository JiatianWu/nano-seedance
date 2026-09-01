"""Inference: prompts in, video + audio out.

    python -m nanoseedance.sample --config configs/toy.yaml \
        --ckpt runs/toy/sft/ckpt_last.pt \
        --prompt "a red circle moving left" "a fast cyan square moving up"
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import torch

from .config import Config
from .pipeline import load_pipeline
from .utils import (
    resolve_device,
    save_contact_sheet,
    save_video,
    save_wav,
    set_seed,
)

DEFAULT_PROMPTS = [
    "a red circle moving left",
    "a cyan square moving up",
    "a green triangle moving right",
    "a yellow circle moving down",
]


def main(argv: Optional[list] = None) -> List[str]:
    parser = argparse.ArgumentParser(description="Generate audio-video from prompts")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--ckpt", default=None, help="DiT checkpoint from any stage")
    parser.add_argument("--prompt", nargs="*", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    device = resolve_device(cfg.train.device)
    set_seed(args.seed)

    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    dit_ckpt = args.ckpt or os.path.join(cfg.train.out_dir, "dit", "ckpt_last.pt")
    out_dir = args.out or os.path.join(os.path.dirname(dit_ckpt), "generated")
    prompts = args.prompt or DEFAULT_PROMPTS

    pipe = load_pipeline(cfg, vae_ckpt, dit_ckpt, device, use_ema=not args.no_ema)

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    noise_device = torch.device("cpu") if device.type == "mps" else device
    if noise_device != device:
        generator = torch.Generator(device=noise_device).manual_seed(args.seed)

    out = pipe.generate(prompts, num_steps=args.steps, guidance_scale=args.guidance,
                        device=device)

    written: List[str] = []
    for i, prompt in enumerate(prompts):
        slug = "%02d_%s" % (i, prompt.replace(" ", "_")[:60])
        written.append(save_video(os.path.join(out_dir, slug + ".mp4"),
                                  out["video"][i].cpu(), fps=cfg.data.fps))
        written.append(save_contact_sheet(os.path.join(out_dir, slug + ".png"),
                                          out["video"][i].cpu()))
        if "waveform" in out:
            written.append(save_wav(os.path.join(out_dir, slug + ".wav"),
                                    out["waveform"][i], cfg.data.sample_rate))
        print("%-50s -> %s" % (prompt, os.path.join(out_dir, slug + ".*")))

    print("wrote %d files to %s" % (len(written), out_dir))
    return written


if __name__ == "__main__":
    main()
