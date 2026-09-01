"""Regression test for a silent MPS correctness bug.

``torch.nn.functional.scaled_dot_product_attention`` on the MPS backend was
observed (torch 2.8.0) to return results differing from CPU by O(1) when
autograd is disabled, while agreeing to ~1e-6 with autograd enabled.

Everything that matters runs under ``torch.no_grad``: encoding video to
latents for DiT training, every sampling step, and RL rollouts. So the failure
mode was training on corrupted latents and computing meaningless importance
ratios, with nothing raised. ``models.layers.attention`` routes MPS around the
bad kernel; these tests fail if that workaround is removed.
"""

from __future__ import annotations

import pytest
import torch

from nanoseedance.config import Config
from nanoseedance.pipeline import SeedancePipeline

mps_only = pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="requires the MPS backend",
)


@pytest.fixture(scope="module")
def cfg() -> Config:
    return Config.load("configs/smoke.yaml")


def _velocity(pipe, device, zv, za, t, captions, grad: bool):
    text, pooled, mask = pipe.encode_prompts(captions, device)
    with torch.enable_grad() if grad else torch.no_grad():
        v, _ = pipe.velocity(zv.to(device), za.to(device), t.to(device),
                             text, pooled, mask, guidance_scale=2.0)
    return v.detach().cpu()


@mps_only
def test_mps_matches_cpu_and_is_grad_mode_invariant(cfg):
    torch.manual_seed(0)
    reference = SeedancePipeline.build(cfg).eval()
    state = reference.dit_state()

    captions = ["a red circle moving left", "a cyan square moving up"]
    zv = torch.randn(2, *cfg.video_latent_shape)
    za = torch.randn(2, *cfg.audio_latent_shape)
    t = torch.rand(2)

    outputs = {}
    for name in ("cpu", "mps"):
        device = torch.device(name)
        pipe = SeedancePipeline.build(cfg)
        pipe.load_dit_state(state)
        pipe = pipe.to(device).eval()
        outputs[name] = (
            _velocity(pipe, device, zv, za, t, captions, grad=False),
            _velocity(pipe, device, zv, za, t, captions, grad=True),
        )

    scale = float(outputs["cpu"][0].std())
    tolerance = 1e-3 * max(scale, 1.0)

    # The bug produced errors of order the signal itself, so a loose tolerance
    # still catches it while allowing normal float32 accumulation differences.
    assert (outputs["mps"][0] - outputs["cpu"][0]).abs().max() < tolerance
    assert (outputs["mps"][1] - outputs["cpu"][1]).abs().max() < tolerance
    assert (outputs["mps"][0] - outputs["mps"][1]).abs().max() < tolerance


@mps_only
def test_attention_helper_is_grad_mode_invariant():
    from nanoseedance.models.layers import attention

    torch.manual_seed(0)
    q, k, v = (torch.randn(6, 1, 4, 64, device="mps") for _ in range(3))
    with torch.no_grad():
        without = attention(q, k, v).cpu()
    with_grad = attention(q, k, v).detach().cpu()
    assert torch.allclose(without, with_grad, atol=1e-5)
