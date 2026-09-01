"""Flow-matching identities and sampler properties."""

from __future__ import annotations

import math

import pytest
import torch

from nanoseedance.config import FlowConfig
from nanoseedance.flow import (
    FlowMatching,
    gaussian_log_prob,
    resolution_shift,
    shift_timesteps,
)


def test_endpoint_recovery():
    """x_hat and noise_hat must invert the forward process exactly."""
    flow = FlowMatching(FlowConfig())
    x = torch.randn(8, 4, 3, 4, 4)
    noise = torch.randn_like(x)
    t = torch.rand(8).clamp(0.05, 0.95)

    z = flow.add_noise(x, noise, t)
    v = flow.velocity_target(x, noise)

    assert torch.allclose(FlowMatching.to_x0(z, v, t), x, atol=1e-5)
    assert torch.allclose(FlowMatching.to_noise(z, v, t), noise, atol=1e-5)


def test_boundary_conditions():
    flow = FlowMatching(FlowConfig())
    x = torch.randn(4, 2, 2)
    noise = torch.randn_like(x)
    assert torch.allclose(flow.add_noise(x, noise, torch.zeros(4)), x, atol=1e-6)
    assert torch.allclose(flow.add_noise(x, noise, torch.ones(4)), noise, atol=1e-6)


def test_ode_step_is_exact_for_linear_path():
    """The path is linear in t, so one Euler step with the true velocity is exact."""
    flow = FlowMatching(FlowConfig())
    x = torch.randn(4, 3, 5)
    noise = torch.randn_like(x)
    t = torch.full((4,), 0.9)
    s = torch.full((4,), 0.2)
    z_t = flow.add_noise(x, noise, t)
    v = flow.velocity_target(x, noise)
    assert torch.allclose(FlowMatching.ode_step(z_t, v, t, s),
                          flow.add_noise(x, noise, s), atol=1e-5)


def test_shift_is_monotonic_and_fixes_endpoints():
    t = torch.linspace(0, 1, 64)
    for shift in (0.5, 1.0, 1.6, 3.0):
        shifted = shift_timesteps(t, shift)
        assert torch.all(shifted[1:] >= shifted[:-1] - 1e-6)
        assert abs(float(shifted[0])) < 1e-6
        assert abs(float(shifted[-1]) - 1.0) < 1e-6


def test_resolution_shift_increases_with_tokens():
    cfg = FlowConfig()
    small = resolution_shift(cfg.base_tokens, cfg)
    large = resolution_shift(cfg.max_tokens, cfg)
    assert large > small


def test_sde_transition_preserves_marginal():
    """z_s built by the stochastic step must have the variance the schedule says.

    If this drifts, the SDE sampler is no longer sampling the same
    distribution as the ODE sampler and GRPO optimises the wrong model.
    """
    torch.manual_seed(0)
    n = 60000
    x = torch.zeros(n, 1)          # data at the origin ...
    noise = torch.randn(n, 1)      # ... so z_t is pure noise scaled by t
    t = torch.full((n,), 0.8)
    s = torch.full((n,), 0.5)

    z_t = FlowMatching.add_noise(x, noise, t)
    v = FlowMatching.velocity_target(x, noise)
    mean, std = FlowMatching.sde_transition(z_t, v, t, s, eta=1.0)
    z_s = mean + std * torch.randn_like(mean)

    # Marginal should be N(0, s^2).
    assert abs(float(z_s.mean())) < 0.02
    assert abs(float(z_s.std()) - 0.5) < 0.02


def test_sde_reduces_to_ode_when_eta_zero():
    flow = FlowMatching(FlowConfig())
    z = torch.randn(4, 3)
    v = torch.randn(4, 3)
    t, s = torch.full((4,), 0.7), torch.full((4,), 0.3)
    mean, std = FlowMatching.sde_transition(z, v, t, s, eta=0.0)
    assert torch.allclose(std, torch.zeros_like(std))
    assert torch.allclose(mean, FlowMatching.ode_step(z, v, t, s), atol=1e-5)


def test_gaussian_log_prob_matches_reference():
    mean = torch.randn(5, 3, 4)
    std = torch.rand(5, 1, 1) + 0.1
    sample = mean + std * torch.randn_like(mean)

    expected = torch.distributions.Normal(mean, std.expand_as(mean)).log_prob(sample)
    assert torch.allclose(gaussian_log_prob(sample, mean, std),
                          expected.flatten(1).sum(-1), atol=1e-4)
