"""Stage 3: supervised fine-tuning.

SFT is not "more pretraining at a lower learning rate". Three things change:

1. **Data is curated.** Only the top-quality slice of the corpus is kept
   (``sft.quality_quantile``); the noisy, low-contrast half that taught the
   model coverage during pretraining now only teaches it to produce artifacts.
2. **Captions get richer.** Pretraining sees terse captions, SFT sees detailed
   ones, which is what buys instruction-following.
3. **The update is constrained**, via LoRA by default, so a few hundred steps
   on a small set cannot wash out pretraining.

    python -m nanoseedance.sft --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from .config import Config
from .data import build_dataloader
from .data.synthetic import curate_indices
from .lora import inject_lora, merge_lora
from .pipeline import SeedancePipeline
from .train_dit import flow_loss, preview
from .utils import (
    EMA,
    AverageMeter,
    Logger,
    count_parameters,
    human_count,
    infinite,
    load_checkpoint,
    lr_lambda,
    resolve_device,
    save_checkpoint,
    set_seed,
)


def main(argv: Optional[list] = None) -> str:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of the DiT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None)
    parser.add_argument("--init", default=None, help="pretrained DiT checkpoint")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    out_dir = cfg.sft.out_dir
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.train.seed + 1)

    device = resolve_device(cfg.train.device)
    logger = Logger(out_dir, "sft")
    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    init_from = args.init or cfg.sft.init_from

    pipe = SeedancePipeline.build(cfg)
    pipe.load_vaes(vae_ckpt)

    ckpt = load_checkpoint(init_from)
    state = ckpt.get("ema") or ckpt["model"]
    pipe.load_dit_state(state, strict=True)
    print("initialised from %s (%s weights)"
          % (init_from, "ema" if ckpt.get("ema") else "raw"))

    pipe.to(device)
    pipe.freeze_vaes()

    # -- 1. curate ---------------------------------------------------------
    if cfg.data.loader == "synthetic":
        keep = curate_indices(cfg.data, cfg.sft.quality_quantile)
        print("curation: kept %d / %d clips (quality quantile %.2f)"
              % (len(keep), cfg.data.num_train, cfg.sft.quality_quantile))
    else:
        keep = None
        print("curation: skipped (non-synthetic loader has no quality score)")

    loader = build_dataloader(cfg, "train", batch_size=cfg.sft.batch_size,
                              caption_style=cfg.sft.caption_style, indices=keep)
    stream = infinite(loader)
    print("captions: %r" % next(iter(loader))["caption"][0])

    # -- 2. constrain the update -------------------------------------------
    generator = pipe.generator_module()
    if cfg.sft.use_lora:
        generator.requires_grad_(False)
        trainable = inject_lora(pipe.dit, cfg.sft.lora_targets,
                                cfg.sft.lora_rank, cfg.sft.lora_alpha)
        # The zero-initialised output head and AdaLN gates stay trainable:
        # they are what let SFT change global style cheaply.
        for name, param in pipe.dit.named_parameters():
            if "modulation" in name or "video_out" in name or "audio_out" in name:
                param.requires_grad_(True)
                trainable.append(param)
        pipe.to(device)
        print("LoRA: rank %d on %s -> %s trainable of %s total"
              % (cfg.sft.lora_rank, list(cfg.sft.lora_targets),
                 human_count(sum(p.numel() for p in trainable)),
                 human_count(count_parameters(pipe.dit))))
    else:
        generator.requires_grad_(True)
        trainable = [p for p in generator.parameters() if p.requires_grad]
        print("full fine-tune: %s trainable params"
              % human_count(sum(p.numel() for p in trainable)))

    optimizer = torch.optim.AdamW(trainable, lr=cfg.sft.lr, betas=cfg.train.betas,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, min(cfg.train.warmup_steps, cfg.sft.steps // 10),
                                       cfg.sft.steps)
    )
    ema = EMA(pipe.generator_module(), cfg.train.ema_decay) if cfg.train.use_ema else None

    # -- 3. train ----------------------------------------------------------
    pipe.dit.train()
    pipe.text_encoder.train()
    meters = {"loss": AverageMeter(), "video": AverageMeter(), "audio": AverageMeter()}
    for step in range(cfg.sft.steps):
        batch = next(stream)
        loss, stats = flow_loss(pipe, batch, device, pipe.flow, cfg.dit.cond_dropout,
                                cfg.flow.decoupled_timestep_prob)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(pipe.generator_module())

        meters["loss"].add(float(loss.detach()))
        for key, value in stats.items():
            meters[key].add(value)
        if (step + 1) % cfg.train.log_every == 0:
            logger.log(step + 1, lr=scheduler.get_last_lr()[0],
                       **{k: m.mean for k, m in meters.items() if m.count})
            for m in meters.values():
                m.reset()

    # -- 4. merge and save a plain checkpoint ------------------------------
    if ema is not None:
        # Bake the averaged weights in before merging adapters away.
        ema_state = {k: v for k, v in ema.state_dict().items()}
        pipe.generator_module().load_state_dict(
            {k: v.to(dtype=torch.float32) for k, v in ema_state.items()}, strict=False
        )
    if cfg.sft.use_lora:
        merged = merge_lora(pipe.dit)
        print("merged %d LoRA adapters into the base weights" % merged)

    pipe.to(device)
    preview(pipe, cfg, out_dir, "sft_final", device,
            captions=["a slow red circle moving left on a black background with a soft tone",
                      "a fast cyan square moving up on a grey background with a harsh tone"])

    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    save_checkpoint(ckpt_path, model=pipe.dit_state(), ema=None,
                    config=cfg.to_dict(), stage="sft")
    print("saved %s" % ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    main()
