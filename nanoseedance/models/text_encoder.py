"""Text encoders for prompt conditioning.

``TinyTextEncoder`` is trained jointly with the DiT over the closed toy
vocabulary. ``HFTextEncoder`` swaps in a frozen T5/CLIP encoder for real
prompts; both expose the same ``(tokens, pooled, mask)`` interface, so the DiT
does not care which one it is given.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .layers import FeedForward, RMSNorm, SelfAttention, rope_params_nd


class TextBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = SelfAttention(dim, num_heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = FeedForward(dim, mlp_ratio)

    def forward(self, x: torch.Tensor,
                rope: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope=rope)
        return x + self.mlp(self.norm2(x))


class TinyTextEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, depth: int, num_heads: int,
                 max_length: int, rope_theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_length = max_length
        self.rope_theta = rope_theta
        self.head_dim = dim // num_heads
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([TextBlock(dim, num_heads) for _ in range(depth)])
        self.norm = RMSNorm(dim)
        nn.init.normal_(self.embed.weight, std=0.02)
        self._rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _rope(self, length: int, device: torch.device):
        if self._rope_cache is None or self._rope_cache[0].shape[0] != length \
                or self._rope_cache[0].device != device:
            self._rope_cache = rope_params_nd((length,), self.head_dim,
                                              self.rope_theta, device)
        return self._rope_cache

    def forward(self, ids: torch.Tensor,
                mask: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.embed(ids)
        rope = self._rope(ids.shape[1], ids.device)
        for block in self.blocks:
            x = block(x, rope)
        x = self.norm(x)

        if mask is None:
            mask = torch.ones(ids.shape, dtype=torch.bool, device=ids.device)
        weights = mask.float().unsqueeze(-1)
        pooled = (x * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        return x, pooled, mask


class HFTextEncoder(nn.Module):
    """Frozen HuggingFace encoder (T5/CLIP) for real prompts.

    Kept out of the default path so the repo has no hard transformers
    dependency; enable with ``dit.text_encoder=hf``.
    """

    def __init__(self, model_name: str = "google/flan-t5-small", max_length: int = 64):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        self.encoder = getattr(model, "encoder", model)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.max_length = max_length
        self.dim = self.encoder.config.hidden_size

    @torch.no_grad()
    def forward(self, captions, device: Optional[torch.device] = None):
        batch = self.tokenizer(list(captions), padding="max_length", truncation=True,
                               max_length=self.max_length, return_tensors="pt")
        batch = {k: v.to(device or self.encoder.device) for k, v in batch.items()}
        out = self.encoder(**batch).last_hidden_state
        mask = batch["attention_mask"].bool()
        weights = mask.float().unsqueeze(-1)
        pooled = (out * weights).sum(1) / weights.sum(1).clamp(min=1e-6)
        return out, pooled, mask
