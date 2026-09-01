"""Stage 1: train the video and audio autoencoders.

Both VAEs are trained together (they share no weights, only the loop) and the
resulting latent scales are calibrated at the end so that stage 2 sees roughly
unit-variance inputs. Nothing downstream trains these weights again.

    python -m nanoseedance.train_vae --config configs/toy.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch

from .config import Config
from .data import build_dataloader
from .models import AudioVAE, VideoVAE, vae_loss
from .utils import (
    AverageMeter,
    Logger,
    count_parameters,
    human_count,
    infinite,
    lr_lambda,
    resolve_device,
    save_checkpoint,
    save_contact_sheet,
    set_seed,
)


def evaluate(video_vae: VideoVAE, audio_vae: Optional[AudioVAE], loader, device,
             max_batches: int = 4) -> dict:
    video_vae.eval()
    if audio_vae is not None:
        audio_vae.eval()
    video_l1, audio_l1 = AverageMeter(), AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            video = batch["video"].to(device)
            recon, _ = video_vae(video, sample_posterior=False)
            video_l1.add(float((recon - video).abs().mean()), video.shape[0])
            if audio_vae is not None:
                mel = batch["mel"].to(device)
                mel_recon, _ = audio_vae(mel, sample_posterior=False)
                audio_l1.add(float((mel_recon - mel).abs().mean()), mel.shape[0])

    video_vae.train()
    if audio_vae is not None:
        audio_vae.train()
    return {"val_video_l1": video_l1.mean, "val_audio_l1": audio_l1.mean}


def main(argv: Optional[list] = None) -> str:
    parser = argparse.ArgumentParser(description="Train the nano-seedance autoencoders")
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.steps=100")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    out_dir = os.path.join(cfg.train.out_dir, "vae")
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.train.seed)

    device = resolve_device(cfg.train.device)
    logger = Logger(out_dir, "vae")
    total_steps = cfg.train.vae_steps or cfg.train.steps
    print("device: %s | %d steps" % (device, total_steps))

    video_vae = VideoVAE(cfg.video_vae).to(device)
    audio_vae = AudioVAE(cfg.audio_vae, cfg.data.n_mels).to(device) if cfg.dit.enable_audio else None
    print("video vae: %s params" % human_count(count_parameters(video_vae)))
    if audio_vae is not None:
        print("audio vae: %s params" % human_count(count_parameters(audio_vae)))

    params = list(video_vae.parameters())
    if audio_vae is not None:
        params += list(audio_vae.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, betas=cfg.train.betas,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, cfg.train.warmup_steps, total_steps)
    )

    train_loader = build_dataloader(cfg, "train")
    val_loader = build_dataloader(cfg, "val", shuffle=False)
    stream = infinite(train_loader)

    meters = {k: AverageMeter() for k in ("loss", "v_l1", "v_kl", "a_l1")}
    for step in range(total_steps):
        batch = next(stream)
        video = batch["video"].to(device)

        recon, posterior = video_vae(video)
        loss, parts = vae_loss(recon, video, posterior, cfg.video_vae.kl_weight)
        meters["v_l1"].add(float(parts["l1"]))
        meters["v_kl"].add(float(parts["kl"]))

        if audio_vae is not None:
            mel = batch["mel"].to(device)
            mel_recon, mel_post = audio_vae(mel)
            audio_loss, audio_parts = vae_loss(mel_recon, mel, mel_post,
                                               cfg.audio_vae.kl_weight)
            loss = loss + audio_loss
            meters["a_l1"].add(float(audio_parts["l1"]))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
        optimizer.step()
        scheduler.step()
        meters["loss"].add(float(loss.detach()))

        if (step + 1) % cfg.train.log_every == 0:
            logger.log(step + 1, lr=scheduler.get_last_lr()[0],
                       **{k: m.mean for k, m in meters.items()})
            for m in meters.values():
                m.reset()

        if (step + 1) % cfg.train.eval_every == 0 or step + 1 == total_steps:
            logger.log(step + 1, **evaluate(video_vae, audio_vae, val_loader, device))

    # Calibrate latent scales on a real batch so the DiT sees ~unit variance.
    video_vae.eval()
    calib = next(iter(val_loader))
    v_scale = video_vae.calibrate_latent_scale(calib["video"].to(device))
    a_scale = None
    if audio_vae is not None:
        audio_vae.eval()
        a_scale = audio_vae.calibrate_latent_scale(calib["mel"].to(device))
    print("latent scales -> video %.4f, audio %s"
          % (v_scale, "%.4f" % a_scale if a_scale else "n/a"))

    # Reconstruction preview: top row is ground truth, bottom is the VAE.
    with torch.no_grad():
        recon, _ = video_vae(calib["video"][:1].to(device), sample_posterior=False)
    save_contact_sheet(os.path.join(out_dir, "recon_target.png"), calib["video"][0])
    save_contact_sheet(os.path.join(out_dir, "recon_output.png"), recon[0].cpu())

    ckpt_path = os.path.join(out_dir, "ckpt_last.pt")
    save_checkpoint(
        ckpt_path,
        video_vae=video_vae.state_dict(),
        audio_vae=audio_vae.state_dict() if audio_vae is not None else None,
        config=cfg.to_dict(),
        latent_scale={"video": v_scale, "audio": a_scale},
    )
    print("saved %s" % ckpt_path)
    return ckpt_path


if __name__ == "__main__":
    main()
