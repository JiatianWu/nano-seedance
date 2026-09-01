# Engineering log

Running record of what was tried, what broke, and what the evidence actually
said. The README documents the *design*; this file documents the *process*,
including dead ends and conclusions that were later overturned.

Conventions: append new entries at the top of "Iteration history". Never delete
a superseded finding — strike it through and record what replaced it, because
the reversals are the most useful thing in here.

---

## Bug ledger

| # | Area | Symptom | Root cause | Status |
|---|---|---|---|---|
| 1 | `models/layers.py` | GRPO importance ratio 0.0006 on first update (must be 1.0) | MPS `scaled_dot_product_attention` wrong under `no_grad` only | Fixed, regression test |
| 2 | `models/audio_vae.py` | Audio recon L1 ~8x worse than expected | `CausalGroupNorm1d` normalising over 2 values per group | Fixed |
| 3 | `models/vae.py` | Video recon washed out | First-stage norm was effectively per-channel InstanceNorm, stripping brightness/colour | Fixed |
| 4 | `rl/rewards.py` | `av_sync` 0.03 matched vs 0.20 shuffled | Off-by-one: accel index `j` centres on video frame `j+1`; point-sampling missed transients | Fixed, validated |
| 5 | `rl/rewards.py` | `audio_prompt` barely discriminated | Baseline mel energy swamping the signal | Fixed (median subtraction + sharpening) |
| 6 | `rl/grpo.py` | KL drift to 52, `clip_frac` 0.61, flat reward | Signed high-variance KL estimator; its `abs()` fights the gradient | Fixed (k3 estimator) |
| 7 | `rl/grpo.py` | `RuntimeError: shape` in rollout | `video_std` flattened on wrong dims | Fixed |
| 8 | `data/synthetic.py` | Caption said "moving left", clip moved right | Shape could bounce within the first frames | Fixed (runway logic) |
| 9 | `train_vae.py`, `train_dit.py` | `RuntimeWarning` on `float(loss)` | Converting a grad-tracking tensor | Fixed (`.detach()`) |
| 10 | `config.py` | VAE and DiT shared one step budget | `train.steps` used by both | Fixed (`vae_steps`) |

---

## Findings that were wrong the first time

Three conclusions I stated and later had to retract. Recorded because the
failure mode is the same each time: reading a number without checking it
against something independently knowable.

1. **"The VAE has failed to converge."** Validation L1 plateaued at 0.19 and I
   spent a tuning cycle on LR and width chasing it. Wrong target: 60% of the
   corpus carries i.i.d. pixel noise, which no bottleneck can reproduce. Against
   the clean signal the VAE scores 0.102 versus a 0.152 noise floor — it is
   *below* the floor because it denoises. The plateau was the metric.

2. **"`av_sync` is a weak reward."** It was not weak, it was backwards
   (0.03 matched, 0.20 shuffled). A weak reward wastes an RL run; an
   anti-correlated one actively trains the model toward desynchronised audio.
   "Weak" and "inverted" look similar if you only check the matched score.

3. **"GRPO improves the model at its own sampling budget."** Seed 0 showed
   1.2713 vs SFT 1.2612 and I had a tidy explanation ready about RL optimising
   the sampler it rolls out with. Four more seeds: −0.009 ± 0.008 with the sign
   flipping. Between-seed spread ~0.09, an order of magnitude above the claimed
   effect. See entry 2026-08-31c.

---

## Iteration history

### 2026-08-31c — GRPO: stability fixed, benefit not demonstrated

Tried, in order:

- **k3 KL estimator.** Replaced `kl_coef * |new_logp - ref_logp|` with
  `exp(r) - 1 - r` where `r = ref_logp - new_logp`, non-negative by
  construction and much lower variance. KL 52 → ~5, `clip_frac` 0.61 → 0.25.
- **Fewer updates per rollout.** 18 optimiser steps on a single rollout batch
  puts late minibatches far outside the trust region. `minibatch_size` 8 → 32,
  `lr` 1e-5 → 3e-6, `kl_coef` 0.02 → 0.05.
- **Bigger groups.** `group_size` 4 → 8 for less noisy within-group advantages.
- **Guidance alignment.** Rollout `guidance_scale` 2.0 → 4.0 to match eval.

Result: rollout reward rises 1.231 → 1.276 across 40 iterations. Held-out
reward does not move. Paired over 5 seeds at 10 steps, GRPO − SFT is
−0.009 ± 0.008 with the sign flipping.

**Conclusion:** the optimiser is now behaving; the policy is not improving on
held-out prompts. 40 iterations over 4 prompts is very little RL, so this is
evidence about this configuration, not about Flow-GRPO.

**Open probe:** widen to more prompts and iterations (a run is <2 min) and see
whether the training/held-out gap closes or is structural.

### 2026-08-31b — Full toy pipeline, all five stages

VAE → DiT → SFT → GRPO → DPO, then a common evaluation on 48 held-out prompts.

| checkpoint | total | prompt_align | av_sync | audio_prompt | aesthetic |
|---|---|---|---|---|---|
| (real data) | 2.335 | 0.877 | 0.529 | 0.730 | 0.798 |
| dit | 1.207 | 0.282 | 0.124 | 0.686 | 0.460 |
| sft | 1.259 | 0.310 | 0.127 | 0.702 | 0.483 |
| grpo | 1.219 | 0.283 | 0.129 | 0.693 | 0.457 |
| dpo | 1.266 | 0.317 | 0.139 | 0.703 | 0.456 |

SFT and DPO beat pretraining, mostly on `prompt_align`. GRPO does not.

**Noise floor for this eval:** differences below ~0.05 total are inside seed
noise at 48 prompts. Established the hard way (see reversal 3).

### 2026-08-31a — VAE reconstruction: the plateau was the metric

Chased a 0.19 validation L1 through an LR sweep and a width increase before
questioning the target. Built a reconstruction comparison image, then scored
against the clean signal rather than the stored (noisy) clip:

| subset | L1 vs stored | L1 vs clean | noise floor |
|---|---|---|---|
| clean clips | 0.091 | 0.075 | 0.037 |
| noisy clips | 0.284 | **0.129** | 0.267 |
| all | 0.187 | **0.102** | 0.152 |

Also gave the VAE its own step budget (`train.vae_steps`) — it had been sharing
`train.steps` with the DiT, which is wrong since the two converge at different
rates.

### 2026-08-30b — Normalisation degeneracy in both VAEs

Audio: `CausalGroupNorm1d` with 32 groups over 64 channels normalises over two
values per group, which destroys the statistics it is meant to standardise.
Capped group count so each group sees at least `MIN_CHANNELS_PER_GROUP = 8`
values. Video: the first-stage norm was effectively per-channel InstanceNorm,
stripping the brightness and colour statistics the decoder needs; documented
the distinction so it is not "simplified" back.

### 2026-08-30a — The MPS attention bug

Most serious defect found in the project, and the one most likely to recur
elsewhere.

- **Symptom:** GRPO importance ratio 0.0006 on the first update, where the
  policy has not yet moved and the ratio is 1.0 by construction.
- **Diagnosis:** bisected the rollout path comparing CPU and MPS tensors
  layer by layer. `F.scaled_dot_product_attention` (torch 2.8.0, MPS) differed
  from CPU by ~1.4 on values of scale ~1.6 — but *only* under `torch.no_grad`.
  With autograd enabled it agreed to 1e-6.
- **Why it hid:** training losses were normal, because training runs with grad
  enabled. Everything under `no_grad` was silently corrupt: latent encoding for
  DiT training, every sampling step, every RL rollout.
- **Fix:** explicit attention implementation for MPS in `models/layers.py`, all
  call sites routed through it. Parity now ~3e-5, grad and no-grad
  bit-identical. `tests/test_backend_parity.py` fails if the workaround is
  removed.
- **Lesson:** verify grad and no-grad paths separately. A green training curve
  says nothing about the inference path.

### 2026-08-29b — Reward validation against deliberate corruptions

Refused to run RL before each reward was shown to move only for the corruption
it targets. Final behaviour:

| reward | matched | wrong caption | wrong audio | noise video | blank video |
|---|---|---|---|---|---|
| `prompt_align` | **0.89** | 0.17 | 0.89 | 0.17 | 0.09 |
| `av_sync` | **0.62** | 0.62 | 0.32 | 0.13 | 0.00 |
| `audio_prompt` | **0.72** | 0.19 | 0.28 | 0.72 | 0.72 |
| `aesthetic` | **0.78** | 0.78 | 0.78 | 0.07 | 0.00 |

`av_sync` took by far the most work — it started anti-correlated. Audio
transients are ~1 video frame wide, the mel hop and video fps are unrelated
rates, and centroid acceleration index `j` centres on video frame `j+1`.
Fixed the offset and switched to max-pooling over each frame's window;
point-sampling an 8x-faster signal steps over the spikes entirely.

### 2026-08-29a — Synthetic corpus and caption/motion agreement

Built the procedural world (coloured shape, wall bounces, colour-determined
pitch on each bounce). Found captions could contradict the clip when a shape
bounced within the first few frames. Added runway logic guaranteeing the
opening motion matches the caption, verified numerically with a colour-matched
centroid estimator (the naive centroid was too noisy to adjudicate).

---

## Standing habits that earned their keep

- Check any metric against something independently knowable: CPU vs MPS,
  reconstruction vs the *clean* signal, rewards vs deliberate corruptions,
  one seed vs five.
- Before believing a ranking, establish the noise floor of the measurement.
- When a doc or an assumption states an invariant, count how often the code
  violates it before acting on it.
- A reward that scores low on matched data may be inverted rather than weak.
  Always score the corruption too.
