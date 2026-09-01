"""Rectified flow matching.

Convention throughout (SD3-style)::

    z_t = (1 - t) * x + t * noise        t = 0 is data, t = 1 is pure noise
    v   = dz_t/dt = noise - x            the network predicts this

Sampling integrates from t=1 down to t=0. Two useful identities fall out of
the parameterisation and are used everywhere below -- given ``z_t`` and a
predicted velocity ``v`` you can recover both endpoints exactly::

    x_hat     = z_t - t * v
    noise_hat = z_t + (1 - t) * v

That is what lets the stochastic sampler re-noise correctly, which in turn is
what gives Flow-GRPO a tractable Gaussian transition to compute log-probs on.
"""

from __future__ import annotations

import math
from typing import Callable, List, Optional, Sequence, Tuple

import torch

from .config import FlowConfig


def resolution_shift(num_tokens: int, cfg: FlowConfig) -> float:
    """Timestep shift as a function of sequence length.

    More tokens means more redundancy, so a given noise level destroys less
    information and the schedule has to spend more of its budget at high t.
    Linear in token count between two anchors, exponentiated (SD3 dynamic
    shifting).
    """
    span = max(cfg.max_tokens - cfg.base_tokens, 1)
    frac = (num_tokens - cfg.base_tokens) / span
    frac = min(max(frac, 0.0), 1.0)
    mu = cfg.base_shift + (cfg.max_shift - cfg.base_shift) * frac
    return math.exp(mu)


def shift_timesteps(t: torch.Tensor, shift: float) -> torch.Tensor:
    """Map t in [0,1] through the shift curve; shift=1 is the identity."""
    return shift * t / (1.0 + (shift - 1.0) * t)


class FlowMatching:
    def __init__(self, cfg: FlowConfig, num_tokens: Optional[int] = None):
        self.cfg = cfg
        self.shift = resolution_shift(num_tokens, cfg) if num_tokens else 1.0

    # -- training ----------------------------------------------------------
    def sample_timesteps(self, batch_size: int, device: torch.device,
                         generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Logit-normal sampling: concentrates training on middle timesteps,
        where the velocity field is hardest to predict."""
        normal = torch.randn(batch_size, device=device, generator=generator)
        t = torch.sigmoid(self.cfg.logit_normal_mean + self.cfg.logit_normal_std * normal)
        return shift_timesteps(t, self.shift)

    @staticmethod
    def add_noise(x: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = _broadcast(t, x)
        return (1.0 - t) * x + t * noise

    @staticmethod
    def velocity_target(x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return noise - x

    @staticmethod
    def to_x0(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z - _broadcast(t, z) * v

    @staticmethod
    def to_noise(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return z + (1.0 - _broadcast(t, z)) * v

    # -- sampling grid -----------------------------------------------------
    def timestep_grid(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Decreasing grid from 1 to 0 with ``num_steps + 1`` entries."""
        grid = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        return shift_timesteps(grid, self.shift)

    # -- transitions -------------------------------------------------------
    @staticmethod
    def ode_step(z: torch.Tensor, v: torch.Tensor,
                 t: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """Deterministic Euler step from time t to time s < t."""
        dt = _broadcast(t - s, z)
        return z - dt * v

    @staticmethod
    def sde_transition(z: torch.Tensor, v: torch.Tensor, t: torch.Tensor,
                       s: torch.Tensor, eta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gaussian transition ``p(z_s | z_t)`` for the stochastic sampler.

        Re-noising is done in endpoint space: recover ``x_hat`` and
        ``noise_hat`` exactly, then rebuild ``z_s`` with part of the original
        noise replaced by fresh noise. Because
        ``sqrt(s^2 - sigma^2) * noise_hat + sigma * new_noise`` has variance
        ``s^2``, the marginal ``z_s = (1-s) x_hat + s * N(0, I)`` is preserved
        exactly -- the noise buys exploration without biasing the sampler.

        Returns ``(mean, std)``; std is a scalar per batch element.
        """
        t_b, s_b = _broadcast(t, z), _broadcast(s, z)
        x_hat = z - t_b * v
        noise_hat = z + (1.0 - t_b) * v

        # DDIM-style ancestral noise, rewritten for the rectified-flow schedule.
        # sigma <= s always, and vanishes at both s=t and s=0.
        safe_t = torch.clamp(t_b, min=1e-5)
        sigma = eta * (s_b / safe_t) * torch.sqrt(torch.clamp(t_b ** 2 - s_b ** 2, min=0.0))
        sigma = torch.minimum(sigma, s_b)

        keep = torch.sqrt(torch.clamp(s_b ** 2 - sigma ** 2, min=0.0))
        mean = (1.0 - s_b) * x_hat + keep * noise_hat
        return mean, sigma


def gaussian_log_prob(sample: torch.Tensor, mean: torch.Tensor,
                      std: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-sample log N(sample; mean, std^2 I), summed over all non-batch dims."""
    std = torch.clamp(std, min=eps)
    var = std ** 2
    logp = -0.5 * ((sample - mean) ** 2 / var) - torch.log(std) - 0.5 * math.log(2 * math.pi)
    return logp.flatten(1).sum(-1)


VelocityFn = Callable[[torch.Tensor, Optional[torch.Tensor], torch.Tensor],
                      Tuple[torch.Tensor, Optional[torch.Tensor]]]


@torch.no_grad()
def sample_ode(
    velocity_fn: VelocityFn,
    video_shape: Tuple[int, ...],
    audio_shape: Optional[Tuple[int, ...]],
    flow: FlowMatching,
    num_steps: int,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Deterministic Euler sampler; used for evaluation and final outputs."""
    z_v = torch.randn(video_shape, device=device, generator=generator)
    z_a = (torch.randn(audio_shape, device=device, generator=generator)
           if audio_shape is not None else None)

    grid = flow.timestep_grid(num_steps, device)
    batch = video_shape[0]
    for i in range(num_steps):
        t = grid[i].expand(batch)
        s = grid[i + 1].expand(batch)
        v_v, v_a = velocity_fn(z_v, z_a, t)
        z_v = FlowMatching.ode_step(z_v, v_v, t, s)
        if z_a is not None and v_a is not None:
            z_a = FlowMatching.ode_step(z_a, v_a, t, s)
    return z_v, z_a


@torch.no_grad()
def sample_sde(
    velocity_fn: VelocityFn,
    video_shape: Tuple[int, ...],
    audio_shape: Optional[Tuple[int, ...]],
    flow: FlowMatching,
    num_steps: int,
    device: torch.device,
    eta: float,
    generator: Optional[torch.Generator] = None,
):
    """Stochastic sampler that records the full trajectory.

    Flow-GRPO needs each denoising step to be a *sampled action* with a known
    log-probability, so this returns the states, means and stds at every step
    rather than just the final latents.
    """
    z_v = torch.randn(video_shape, device=device, generator=generator)
    z_a = (torch.randn(audio_shape, device=device, generator=generator)
           if audio_shape is not None else None)

    grid = flow.timestep_grid(num_steps, device)
    batch = video_shape[0]
    traj = {
        "video_prev": [], "video_next": [], "video_mean": [], "video_std": [],
        "audio_prev": [], "audio_next": [], "audio_mean": [], "audio_std": [],
        "t": [], "s": [],
    }

    for i in range(num_steps):
        t = grid[i].expand(batch)
        s = grid[i + 1].expand(batch)
        v_v, v_a = velocity_fn(z_v, z_a, t)

        mean_v, std_v = FlowMatching.sde_transition(z_v, v_v, t, s, eta)
        next_v = mean_v + std_v * torch.randn(mean_v.shape, device=device,
                                              generator=generator)
        traj["video_prev"].append(z_v)
        traj["video_mean"].append(mean_v)
        traj["video_std"].append(std_v)
        traj["video_next"].append(next_v)

        if z_a is not None and v_a is not None:
            mean_a, std_a = FlowMatching.sde_transition(z_a, v_a, t, s, eta)
            next_a = mean_a + std_a * torch.randn(mean_a.shape, device=device,
                                                  generator=generator)
            traj["audio_prev"].append(z_a)
            traj["audio_mean"].append(mean_a)
            traj["audio_std"].append(std_a)
            traj["audio_next"].append(next_a)
            z_a = next_a

        traj["t"].append(t)
        traj["s"].append(s)
        z_v = next_v

    stacked = {k: (torch.stack(v, dim=1) if v else None) for k, v in traj.items()}
    return z_v, z_a, stacked


def _broadcast(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Reshape a per-sample scalar to broadcast against ``like``."""
    return t.view(-1, *([1] * (like.dim() - 1))).to(like.dtype)
