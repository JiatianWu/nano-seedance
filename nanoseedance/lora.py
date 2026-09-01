"""Low-rank adapters for the SFT stage.

Fine-tuning a diffusion transformer on a small curated set is the classic
overfitting trap: full fine-tuning at high LR erases pretraining. LoRA
constrains the update to a low-rank subspace, which both regularises and cuts
optimiser memory. Adapters are merged back into the base weights on save, so
every downstream stage just loads an ordinary state dict.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / rank
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_a.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_b.weight)  # start as an exact no-op
        self.base.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(x)) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """Fold BA into the base weight and return a plain Linear."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        delta = (self.lora_b.weight @ self.lora_a.weight) * self.scaling
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def inject_lora(module: nn.Module, targets: Sequence[str], rank: int = 8,
                alpha: float = 16.0) -> List[nn.Parameter]:
    """Replace every ``nn.Linear`` whose attribute name is in ``targets``.

    Returns the newly created trainable parameters.
    """
    created: List[nn.Parameter] = []
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            wrapper = LoRALinear(child, rank, alpha)
            setattr(module, name, wrapper)
            created.extend([wrapper.lora_a.weight, wrapper.lora_b.weight])
        else:
            created.extend(inject_lora(child, targets, rank, alpha))
    return created


@torch.no_grad()
def merge_lora(module: nn.Module) -> int:
    """Recursively replace every LoRALinear with its merged Linear."""
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged_linear())
            merged += 1
        else:
            merged += merge_lora(child)
    return merged


def lora_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in module.state_dict().items() if "lora_" in k}
