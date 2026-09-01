"""Transformer primitives shared by the video and audio branches.

Linear layers are given stable names (``qkv``, ``proj``, ``to_q``, ``to_kv``,
``fc1``, ``fc2``) because the LoRA injector in ``sft.py`` targets them by name.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# norms / embeddings
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal embedding of continuous flow time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    args = t.float().view(-1, 1) * freqs.view(1, -1) * 1000.0
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_dim))


# --------------------------------------------------------------------------
# rotary position embeddings
# --------------------------------------------------------------------------
def _axis_angles(positions: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """(L,) positions -> (L, dim // 2) rotation angles."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32,
                                          device=positions.device) / dim))
    return positions.float().unsqueeze(1) * freqs.unsqueeze(0)


def split_dim(head_dim: int, num_axes: int) -> List[int]:
    """Split head_dim across axes, keeping every part even."""
    if head_dim % 2:
        raise ValueError("head_dim must be even for rotary embeddings")
    pairs = head_dim // 2
    base, extra = divmod(pairs, num_axes)
    parts = [(base + (1 if i < extra else 0)) * 2 for i in range(num_axes)]
    return parts


def rope_params_nd(sizes: Sequence[int], head_dim: int, theta: float,
                   device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Axial N-D rotary tables.

    Returns ``(cos, sin)`` of shape ``(prod(sizes), head_dim // 2)``, laid out
    so that the flattened token order matches row-major order over ``sizes``.
    """
    parts = split_dim(head_dim, len(sizes))
    per_axis: List[torch.Tensor] = []
    for axis, (length, dim) in enumerate(zip(sizes, parts)):
        angles = _axis_angles(torch.arange(length, device=device), dim, theta)  # (L, dim/2)
        # Broadcast this axis over the others to build the full grid.
        shape = [1] * len(sizes) + [dim // 2]
        shape[axis] = length
        expanded = angles.view(shape).expand(list(sizes) + [dim // 2])
        per_axis.append(expanded.reshape(-1, dim // 2))
    grid = torch.cat(per_axis, dim=-1)  # (N, head_dim/2)
    return torch.cos(grid), torch.sin(grid)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate interleaved pairs of ``x`` (B, heads, N, head_dim)."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos.to(x.dtype).view(1, 1, *cos.shape)
    sin = sin.to(x.dtype).view(1, 1, *sin.shape)
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


# --------------------------------------------------------------------------
# attention / mlp
# --------------------------------------------------------------------------
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Scaled dot-product attention, avoiding a wrong MPS kernel.

    On the MPS backend ``F.scaled_dot_product_attention`` can return results
    that differ from CPU by O(1) when autograd is disabled, while agreeing to
    ~1e-6 with it enabled. That combination is particularly nasty here: every
    VAE encode and every sampling step runs under ``torch.no_grad``, so
    training would consume corrupted latents and RL importance ratios would be
    garbage, all without raising anything. The explicit implementation below
    agrees with CPU in both modes, so MPS uses it.

    See ``tests/test_backend_parity.py``, which fails if this is removed.
    """
    if q.device.type == "mps":
        scores = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)
        return scores.softmax(dim=-1) @ v
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q = apply_rope(q, rope[0], rope[1])
            k = apply_rope(k, rope[0], rope[1])
        out = attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class CrossAttention(nn.Module):
    """Queries from one stream, keys/values from another (text, or the other branch)."""

    def __init__(self, dim: int, context_dim: int, num_heads: int, qk_norm: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_kv = nn.Linear(context_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, n, _ = x.shape
        m = context.shape[1]
        q = self.to_q(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.to_kv(context).view(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q, k = self.q_norm(q), self.k_norm(k)

        attn_mask = None
        if context_mask is not None:
            attn_mask = context_mask.view(b, 1, 1, m).expand(b, self.num_heads, n, m)
        out = attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


# --------------------------------------------------------------------------
# adaptive layer norm (DiT-style conditioning)
# --------------------------------------------------------------------------
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class AdaLNModulation(nn.Module):
    """Produces ``num_groups`` x (shift, scale, gate) triplets from a conditioning vector."""

    def __init__(self, dim: int, num_groups: int):
        super().__init__()
        self.num_groups = num_groups
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * num_groups * dim))
        # Zero-init => every block starts as the identity, which is what makes
        # deep DiTs trainable from scratch without warmup tricks.
        nn.init.zeros_(self.net[1].weight)
        nn.init.zeros_(self.net[1].bias)

    def forward(self, cond: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        params = self.net(cond).chunk(3 * self.num_groups, dim=-1)
        return [
            (params[3 * i], params[3 * i + 1], params[3 * i + 2])
            for i in range(self.num_groups)
        ]


class FinalLayer(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.linear = nn.Linear(dim, out_dim)
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.modulation(cond).chunk(2, dim=-1)
        return self.linear(modulate(self.norm(x), shift, scale))
