# nano-seedance

A minimal but complete reconstruction of a Seedance-style audio-video generation
stack, small enough to train end to end on a laptop:

```
synthetic AV corpus  ->  causal 3D VAE  ->  dual-branch flow-matching DiT  ->  SFT  ->  RL
                         (+ mel VAE)        (video + audio + AV bridge)      (LoRA)   (GRPO / DPO)
```

Every stage is runnable, every stage is tested, and the RL stages optimise
rewards that can be *verified* against ground truth rather than assumed.

`ENGINEERING_LOG.md` is the working record behind this: the bug ledger, the
iteration history, and the conclusions that turned out to be wrong.

## What is actually Seedance, and what is reconstruction

Worth stating plainly, because the repo is named after a model whose internals
are not public.

The official [Seedance 2.0 report](https://arxiv.org/pdf/2604.14148) is almost
entirely an **evaluation** paper: benchmarks, arena results, per-category
ratings. It does not disclose the architecture. What is publicly attributed to
the model — a dual-branch DiT that generates video and audio jointly, a
temporally-causal VAE, flow matching with velocity prediction, decoupled
spatial/temporal attention, resolution-aware timestep shift — comes from
secondary write-ups and the documented Seedance 1.0/1.5 lineage.

So: **this is a faithful implementation of that described design, not a
reimplementation of disclosed code.** Where the public description is silent I
picked the standard modern choice (SD3-style logit-normal timesteps and dynamic
shifting, PixArt-style cross-attention conditioning, DDIM-style ancestral
noise) and said so in the source. Nothing here is claimed to match ByteDance's
actual weights, hyperparameters, or results.

## Quickstart

```bash
pip install -r requirements.txt

./scripts/run_pipeline.sh configs/smoke.yaml   # ~1 min, checks the wiring
./scripts/run_pipeline.sh                      # ~30 min, the real toy run
pytest tests/ -q
```

Individual stages:

```bash
python -m nanoseedance.train_vae  --config configs/toy.yaml   # 1. autoencoders
python -m nanoseedance.train_dit  --config configs/toy.yaml   # 2. flow-matching pretrain
python -m nanoseedance.sft        --config configs/toy.yaml   # 3. supervised fine-tune
python -m nanoseedance.rl.grpo    --config configs/toy.yaml   # 4a. online RL
python -m nanoseedance.rl.dpo     --config configs/toy.yaml   # 4b. offline preference RL
python -m nanoseedance.evaluate   --config configs/toy.yaml --ckpt runs/toy/*/ckpt_last.pt
python -m nanoseedance.sample     --config configs/toy.yaml --ckpt runs/toy/sft/ckpt_last.pt \
                                  --prompt "a red circle moving left"
```

Config values can be overridden inline: `--set train.steps=500 dit.depth=4`.

Three configs ship: `smoke.yaml` (1 minute, correctness only), `toy.yaml`
(CPU/MPS, the default), `base.yaml` (GPU scale: 256px, 49 frames, ~500M DiT).

## The data

Real text-to-video corpora are terabytes and need a cluster, which would make
none of this runnable. Instead `data/synthetic.py` generates a small world with
genuine cross-modal structure: a coloured shape moves across the frame and
bounces off the walls, and **every bounce produces an audible transient at a
pitch determined by the shape's colour**.

That gives the audio branch and the AV bridge something real to learn, and it
makes rewards checkable. Two details matter more than they look:

- **Runway.** Each clip starts at the far end of its travel axis, so the first
  and longest leg of the trajectory matches the caption. Without this a clip
  captioned "moving up" can start against the top wall, bounce on frame 1, and
  move down for the rest of the clip — the caption would contradict the pixels
  and the prompt-alignment reward would be incoherent.
- **Mixed quality.** Roughly 60% of the corpus is degraded with noise and
  washed-out contrast. Pretraining sees all of it; SFT keeps only the clean
  slice. That is the point of having an SFT stage at all.

`data/video_dataset.py` is the real-video path (a folder of mp4s plus captions,
or a `metadata.jsonl` manifest); set `data.loader=folder` and `data.root`.

## Architecture

### Causal 3D video VAE (`models/vae.py`)

Compresses `(3, T, H, W)` to `(C, 1 + (T-1)/rt, H/rs, W/rs)` — 4x temporal and
4x spatial in `toy.yaml`, 4x/8x in `base.yaml`. Two properties are enforced and
tested:

**Temporal causality.** Latent frame *i* never depends on input frames after
it. This is what lets one model encode both a single image and a video, and
what makes chunked encoding of long clips valid. It is also easy to break by
accident: an ordinary `GroupNorm` over `(C, T, H, W)` pools statistics across
the entire time axis and leaks the future backwards. Normalisation here is
applied per frame (`CausalGroupNorm`), and convolutions pad only on the left.

**Frame-0 anchoring.** With `T = 1 + 4k` the encoder emits `1 + k` latent
frames where latent frame 0 corresponds to input frame 0 alone.
`tests/test_vae.py` asserts both, including that encoding a lone image gives
bit-identical results to latent frame 0 of a full clip.

### Dual-branch DiT (`models/dit.py`)

Video and audio each get their own transformer stack at their own token rate,
exchanging information through the bridge.

- **Decoupled spatial/temporal attention.** Attention within a frame, then
  across time at each spatial position: `O(T·S² + S·T²)` instead of `O((T·S)²)`
  for full 3D attention.
- **AV bridge** (`AVBridge`). Bidirectional cross attention every
  `bridge_every` blocks. Because it is cross attention and not a shared
  sequence, the branches never need a common clock — mel frames arrive ~8x
  faster than video frames and the attention map absorbs the mismatch.
  Updates are simultaneous, so neither branch gets a half-step advantage.
- **Conditioning.** AdaLN-Zero from (timestep + pooled text) plus cross
  attention to text tokens. A learned null embedding provides classifier-free
  guidance; cond/uncond are batched into one forward pass.
- **3D RoPE**, axial over (t, h, w) for video and 1D for audio.

`tests/test_model_and_rl.py` includes a test that perturbing *only* the audio
latent changes the video prediction — otherwise the "dual-branch" model would
just be two independent DiTs.

### Flow matching (`flow.py`)

SD3 convention: `z_t = (1-t)·x + t·noise`, network predicts velocity
`v = noise - x`, sampling integrates from t=1 to t=0. Two identities recover
both endpoints from any `(z_t, v)` and are used throughout:

```
x_hat = z_t - t·v          noise_hat = z_t + (1-t)·v
```

Timesteps are sampled logit-normally and pushed through a **resolution-aware
shift**: more tokens means more redundancy, so a given noise level destroys
less information and the schedule must spend more budget at high t.

Training also uses **decoupled timesteps** on 25% of steps — the two branches
get independent noise levels. Training only on a shared timestep would leave
the model undefined on inputs like "clean video, noisy audio", which is exactly
what video-to-audio dubbing needs at inference.

## The five stages

**1. Autoencoders.** Video and mel VAEs, L1 + MSE + KL. Latent scales are
calibrated at the end so the DiT sees unit-variance inputs. (GAN/LPIPS terms
are deliberately omitted — they roughly triple the code for no benefit at
32x32.)

Read the reconstruction number carefully. Validation L1 against the *stored*
clips plateaus around 0.19 and looks like a failure to converge, but 60% of the
corpus carries i.i.d. pixel noise, and noise is incompressible — no 40x
bottleneck can reproduce it. Scored against the clean signal instead:

| subset | L1 vs stored clip | L1 vs clean signal | noise floor |
|---|---|---|---|
| clean clips | 0.091 | 0.075 | 0.037 |
| noisy clips | 0.284 | **0.129** | 0.267 |
| all | 0.187 | **0.102** | 0.152 |

The VAE sits *below* the noise floor because it denoises rather than memorises.
The apparent plateau was the metric, not the model.

**2. DiT pretraining.** Frozen VAEs, flow-matching loss on both branches, EMA,
10% conditioning dropout for CFG.

**3. SFT.** Not "more pretraining at a lower LR" — three things change:
curated data (top-quality slice only), richer captions, and a constrained
update via LoRA. Adapters are merged into the base weights on save, so every
downstream stage just loads an ordinary state dict.

**4a. Flow-GRPO** (`rl/grpo.py`). A deterministic ODE sampler has no policy to
optimise: given prompt and initial noise the output is fixed, so there is no
action distribution and no log-probability. Converting the sampler to an SDE
with *identical marginals* makes each denoising step a Gaussian action with a
tractable log-prob. GRPO then needs no value network — sample a group of
rollouts per prompt and use the group's own mean reward as the baseline, so the
advantage is a within-group z-score that cancels prompt difficulty. Loss is
PPO's clipped surrogate plus a KL penalty to a frozen reference.

**4b. Diffusion-DPO** (`rl/dpo.py`). The offline complement: generate several
candidates per prompt, keep best and worst, train the model to prefer the
winner relative to a frozen reference. Both members of a pair are noised at the
*same* timestep with the *same* noise — otherwise the timestep difference
between the two branches drowns out the preference signal.

## Verifiable rewards

Production systems train a neural reward model on human preferences. That is
the right design, but it makes it impossible to debug an RL implementation: if
reward goes up, did the policy improve or did it learn to game the scorer?

So the rewards here measure agreement with facts knowable independently of the
model (`rl/rewards.py`). Measured on held-out real clips versus targeted
corruptions:

| reward | matched | wrong caption | wrong audio | noise video | blank video |
|---|---|---|---|---|---|
| `prompt_align` | **0.89** | 0.17 | 0.89 | 0.17 | 0.09 |
| `av_sync` | **0.62** | 0.62 | 0.32 | 0.13 | 0.00 |
| `audio_prompt` | **0.72** | 0.19 | 0.28 | 0.72 | 0.72 |
| `aesthetic` | **0.78** | 0.78 | 0.78 | 0.07 | 0.00 |

Each reward moves only for the corruption it is meant to detect and ignores the
others. `NeuralReward` is the drop-in for a learned model; nothing else in the
RL code changes.

Getting `av_sync` to work took the most care, and the failure was instructive:
audio transients are about one video frame wide, the mel hop and the video fps
are unrelated rates, and centroid acceleration index `j` is centred on video
frame `j+1`. Getting that offset wrong does not merely weaken the reward — it
*anti-correlates* it (matched 0.03 vs shuffled 0.20, i.e. confidently backwards).
Pooling also has to be max over each frame's window, since point-sampling an
8x-faster signal steps straight over the spikes.

## What the toy run actually shows

Full `configs/toy.yaml` pipeline, all four checkpoints scored on the same 48
held-out prompts with the same seeds. Real clips are the ceiling; the reward
scale is not normalised, so only relative gaps mean anything.

| checkpoint | total | prompt_align | av_sync | audio_prompt | aesthetic |
|---|---|---|---|---|---|
| (real data) | 2.335 | 0.877 | 0.529 | 0.730 | 0.798 |
| dit | 1.207 | 0.282 | 0.124 | 0.686 | 0.460 |
| sft | 1.259 | 0.310 | 0.127 | 0.702 | 0.483 |
| grpo | 1.219 | 0.283 | 0.129 | 0.693 | 0.457 |
| dpo | 1.266 | 0.317 | 0.139 | 0.703 | 0.456 |

SFT is a clear gain over pretraining, mostly on `prompt_align`. **GRPO is not.**
Its own rollout reward rises over training (1.231 to 1.276 from the first to the
last quarter of 40 iterations), but that does not survive held-out evaluation.
Paired over 5 seeds at GRPO's own 10-step sampling budget, GRPO minus SFT is
-0.009 +/- 0.008, with the sign flipping across seeds — indistinguishable from
noise, and if anything slightly negative. Between-seed spread (~0.09) is an
order of magnitude larger than the effect being claimed.

That gap between training reward and held-out reward is the thing to take
seriously, and the honest reading at this scale is that 40 iterations over 4
prompts moves the rollout distribution without moving the policy. Do not read
a single-seed eval win as an improvement; the first version of this table
showed GRPO ahead at 10 steps on seed 0 alone, and it did not replicate.

Two GRPO settings mattered a lot for stability, both visible in the logs:

- **KL estimator.** Penalising `|new_logp - ref_logp|` — a signed, high-variance
  estimate — let KL drift monotonically to 52 while `clip_frac` sat at 0.61.
  Schulman's k3 estimator (`exp(r) - 1 - r`, non-negative by construction) holds
  KL near 5 with `clip_frac` at 0.25.
- **Updates per rollout.** 18 optimiser steps on one batch of rollouts puts most
  samples far outside the trust region by the end of the batch. Larger
  minibatches and a lower LR keep ratios near 1.

## A silent MPS correctness bug

On torch 2.8.0, `F.scaled_dot_product_attention` on the MPS backend returned
results differing from CPU by ~1.4 (on values of scale ~1.6) **when autograd
was disabled**, while agreeing to ~1e-6 with it enabled.

That combination is nasty here, because everything that matters runs under
`torch.no_grad`: encoding video to latents for DiT training, every sampling
step, and every RL rollout. The observable symptom was GRPO importance ratios
of 0.0006 on the first update — which should be exactly 1.0 — while training
losses looked entirely normal.

`models/layers.attention` routes MPS around the bad kernel with an explicit
implementation. After the fix, MPS matches CPU to ~3e-5 and grad/no-grad are
bit-identical. `tests/test_backend_parity.py` fails if the workaround is
removed. Note the bug was in the *inference* path, not training — if you hit
something similar, check both modes before trusting either.

## Scaling up

`configs/base.yaml` sketches a real run: 256px, 49 frames, 8x spatial and 4x
temporal compression, a ~500M-parameter DiT, frozen T5 text encoder, real video
loaders. What it does **not** include, and what you would need to add:

- Multi-GPU (FSDP/DDP), gradient checkpointing, mixed precision beyond the
  `amp` flag
- A VAE discriminator and perceptual loss, which is where real sharpness comes
  from
- Sequence/context parallelism for long clips
- A learned reward model trained on human preferences
- Prompt rewriting, safety filtering, and the data curation pipeline that
  dominates real quality

## Layout

```
nanoseedance/
  config.py            dataclass configs, YAML + dotted CLI overrides
  flow.py              rectified flow, timestep shift, ODE + SDE samplers
  pipeline.py          ties components together; CFG and decoding live here
  lora.py              low-rank adapters, merge-on-save
  train_vae.py  train_dit.py  sft.py  sample.py  evaluate.py
  data/    synthetic.py  video_dataset.py  audio.py  text.py  loader.py
  models/  vae.py  audio_vae.py  dit.py  layers.py  text_encoder.py
  rl/      rewards.py  grpo.py  dpo.py
configs/   smoke.yaml  toy.yaml  base.yaml
tests/
```

## Known limitations

- The toy world is procedural. A model that fits it has learned colour, shape,
  direction and pitch — not video generation in any general sense.
- Griffin-Lim over 32 mel bands is a lossy vocoder; audio is intelligible as
  pitch and rhythm, not as fidelity.
- The VAE has no adversarial loss, so reconstructions are soft.
- GRPO here uses full-trajectory backprop through single steps with a small
  group size; at scale you want more rollouts and gradient accumulation. As
  measured above it does not beat SFT on held-out prompts at toy scale.
- Reward differences below ~0.05 total are inside seed noise for a 48-prompt
  eval. Compare across several seeds before believing any ranking.
