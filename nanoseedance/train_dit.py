"""Stage 2: flow-matching pretraining of the dual-branch DiT.

The VAEs are frozen; every batch is encoded to latents, noised at a sampled
timestep, and the model is asked to predict the velocity that transports noise
back to data.

    python -m nanoseedance.train_dit --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import Config
from .data import build_dataloader
from .flow import FlowMatching
from .pipeline import SeedancePipeline
from .utils import (
    EMA,
    AverageMeter,
    Logger,
    count_parameters,
    human_count,
    infinite,
    lr_lambda,
    resolve_device,
    save_checkpoint,
    save_contact_sheet,
    save_video,
    save_wav,
    set_seed,
)


def flow_loss(pipe: SeedancePipeline, batch: Dict[str, Any], device: torch.device,
              flow: FlowMatching, cond_dropout: float,
              decoupled_prob: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    z_v, z_a = pipe.encode_batch(batch, device)
    batch_size = z_v.shape[0]

    t = flow.sample_timesteps(batch_size, device)
    noise_v = torch.randn_like(z_v)
    zt_v = flow.add_noise(z_v, noise_v, t)
    target_v = flow.velocity_target(z_v, noise_v)

    t_audio = None
    zt_a = target_a = None
    if z_a is not None:
        # Independent timesteps on a fraction of steps; see FlowConfig.
        decoupled = torch.rand(batch_size, device=device) < decoupled_prob
        t_audio = torch.where(decoupled, flow.sample_timesteps(batch_size, device), t)
        noise_a = torch.randn_like(z_a)
        zt_a = flow.add_noise(z_a, noise_a, t_audio)
        target_a = flow.velocity_target(z_a, noise_a)

    text, pooled, mask = pipe.encode_prompts(batch["caption"], device)
    drop = torch.rand(batch_size, device=device) < cond_dropout

    pred_v, pred_a = pipe.dit(zt_v, t, text, pooled, mask, audio_latent=zt_a,
                              t_audio=t_audio, drop_cond=drop)

    loss_v = F.mse_loss(pred_v, target_v)
    loss = loss_v
    stats = {"video": float(loss_v.detach())}
    if pred_a is not None and target_a is not None:
        loss_a = F.mse_loss(pred_a, target_a)
        loss = loss + loss_a
        stats["audio"] = float(loss_a.detach())
    return loss, stats


@torch.no_grad()
def preview(pipe: SeedancePipeline, cfg: Config, out_dir: str, tag: str,
            device: torch.device, captions=None, ema: Optional[EMA] = None) -> None:
    """Sample a couple of prompts and write mp4 / wav / contact sheet.

    Previews use the EMA weights when available, since those are what the
    checkpoint will ship.
    """
    captions = captions or ["a red circle moving left", "a cyan square moving up"]
    pipe.dit.eval()
    context = ema.averaged(pipe.generator_module()) if ema is not None else nullcontext()
    with context:
        out = pipe.generate(captions, num_steps=cfg.flow.num_sampling_steps,
                            guidance_scale=cfg.flow.guidance_scale, device=device)
    for i, caption in enumerate(captions):
        slug = caption.replace(" ", "_")
        save_contact_sheet(os.path.join(out_dir, "samples", "%s_%s.png" % (tag, slug)),
                           out["video"][i].cpu())
        save_video(os.path.join(out_dir, "samples", "%s_%s.mp4" % (tag, slug)),
                   out["video"][i].cpu(), fps=cfg.data.fps)
        if "waveform" in out:
            save_wav(os.path.join(out_dir, "samples", "%s_%s.wav" % (tag, slug)),
                     out["waveform"][i], cfg.data.sample_rate)
    pipe.dit.train()


def main(argv: Optional[list] = None) -> str:
    parser = argparse.ArgumentParser(description="Flow-matching pretraining of the DiT")
    parser.add_argument("--config", default=None)
    parser.add_argument("--vae", default=None, help="path to the stage-1 checkpoint")
    parser.add_argument("--set", nargs="*", default=[])
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    out_dir = os.path.join(cfg.train.out_dir, "dit")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)
    logger = Logger(out_dir, "dit")
    vae_ckpt = args.vae or os.path.join(cfg.train.out_dir, "vae", "ckpt_last.pt")
    print("device: %s | vae: %s" % (device, vae_ckpt))

    pipe = SeedancePipeline.build(cfg)
    pipe.load_vaes(vae_ckpt)
    pipe.to(device)
    pipe.freeze_vaes()
    pipe.dit.train()
    pipe.text_encoder.train()

    generator = pipe.generator_module()
    trainable = list(generator.parameters())
    print("dit: %s params | text encoder: %s params"
          % (human_count(count_parameters(pipe.dit)),
             human_count(count_parameters(pipe.text_encoder))))

    optimizer = torch.optim.AdamW(trainable, lr=cfg.train.lr, betas=cfg.train.betas,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg.train.warmup_steps, cfg.train.steps)
    )
    ema = EMA(generator, cfg.train.ema_decay) if cfg.train.use_ema else None

    loader = build_dataloader(cfg, "train")
    stream = infinite(loader)
    flow = pipe.flow
    print("timestep shift %.3f for %d video tokens" % (flow.shift, cfg.num_video_tokens))

    meters = {"loss": AverageMeter(), "video": AverageMeter(), "audio": AverageMeter()}
    for step in range(cfg.train.steps):
        batch = next(stream)
        loss, stats = flow_loss(pipe, batch, device, flow, cfg.dit.cond_dropout,
                                cfg.flow.decoupled_timestep_prob)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(generator)

        meters["loss"].add(float(loss.detach()))
        for key, value in stats.items():
            meters[key].add(value)

        if (step + 1) % cfg.train.log_every == 0:
            logger.log(step + 1, lr=scheduler.get_last_lr()[0],
                       **{k: m.mean for k, m in meters.items() if m.count})
            for m in meters.values():
                m.reset()

        if (step + 1) % cfg.train.eval_every == 0:
            preview(pipe, cfg, out_dir, "step%06d" % (step + 1), device, ema=ema)

    preview(pipe, cfg, out_dir, "final", device, ema=ema)
    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    save_checkpoint(
        ckpt_path,
        model=pipe.dit_state(),
        ema=dict(ema.state_dict()) if ema else None,
        config=cfg.to_dict(),
        step=cfg.train.steps,
    )
    print("saved %s" % ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    main()
