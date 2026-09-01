"""Word-level tokenizer for the toy caption grammar.

Real systems condition on a frozen T5/CLIP encoder. At this scale a learned
embedding over a closed vocabulary is both sufficient and far cheaper, and it
keeps the repo dependency-free. ``models.text_encoder.HFTextEncoder`` is the
drop-in for a real encoder.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .synthetic import BACKGROUND_NAMES, COLOR_NAMES, DIRECTIONS, SHAPES, SHAPE_TIMBRE, SPEEDS

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

_GRAMMAR_WORDS: List[str] = [
    "a", "an", "the", "moving", "on", "background", "with", "tone", "and",
    "of", "in", "to", "shape", "colour", "color", "sound", "video",
]


def build_vocab() -> List[str]:
    words: List[str] = [PAD, BOS, EOS, UNK]
    seen = set(words)
    for group in (
        _GRAMMAR_WORDS,
        list(SHAPES),
        list(COLOR_NAMES),
        list(BACKGROUND_NAMES),
        list(DIRECTIONS),
        list(SPEEDS),
        sorted(set(SHAPE_TIMBRE.values())),
    ):
        for word in group:
            if word not in seen:
                seen.add(word)
                words.append(word)
    return words


class CaptionTokenizer:
    """Deterministic word-level tokenizer with a fixed, closed vocabulary."""

    def __init__(self, max_length: int = 24):
        self.itos = build_vocab()
        self.stoi: Dict[str, int] = {w: i for i, w in enumerate(self.itos)}
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    def encode(self, text: str) -> List[int]:
        tokens = [self.stoi[BOS]]
        for word in text.lower().replace(",", " ").split():
            tokens.append(self.stoi.get(word, self.stoi[UNK]))
        tokens.append(self.stoi[EOS])
        return tokens[: self.max_length]

    def decode(self, ids: Sequence[int]) -> str:
        specials = {self.stoi[PAD], self.stoi[BOS], self.stoi[EOS]}
        return " ".join(self.itos[i] for i in ids if int(i) not in specials)

    def batch_encode(self, texts: Sequence[str],
                     device: Optional[torch.device] = None):
        """Return (ids, mask) of shape (B, max_length); mask is 1 on real tokens."""
        encoded = [self.encode(t) for t in texts]
        ids = torch.full((len(texts), self.max_length), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(texts), self.max_length), dtype=torch.bool)
        for row, seq in enumerate(encoded):
            ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[row, : len(seq)] = True
        if device is not None:
            ids, mask = ids.to(device), mask.to(device)
        return ids, mask
