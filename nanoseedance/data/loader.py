"""Dataset/dataloader construction, dispatching on ``data.loader``."""

from __future__ import annotations

from typing import List, Optional

from torch.utils.data import DataLoader, Dataset

from ..config import Config, DataConfig
from .synthetic import SyntheticAVDataset, collate


def build_dataset(cfg: DataConfig, split: str = "train", caption_style: str = "simple",
                  indices: Optional[List[int]] = None) -> Dataset:
    if cfg.loader == "synthetic":
        return SyntheticAVDataset(cfg, split=split, caption_style=caption_style,
                                  indices=indices)
    if cfg.loader in ("folder", "webdataset"):
        from .video_dataset import RealVideoDataset

        return RealVideoDataset(cfg, split=split, caption_style=caption_style,
                                indices=indices)
    raise ValueError("unknown data.loader %r" % cfg.loader)


def build_dataloader(cfg: Config, split: str = "train", batch_size: Optional[int] = None,
                     caption_style: str = "simple", indices: Optional[List[int]] = None,
                     shuffle: Optional[bool] = None, num_workers: Optional[int] = None
                     ) -> DataLoader:
    dataset = build_dataset(cfg.data, split=split, caption_style=caption_style,
                            indices=indices)
    return DataLoader(
        dataset,
        batch_size=batch_size or cfg.train.batch_size,
        shuffle=(split == "train") if shuffle is None else shuffle,
        num_workers=cfg.train.num_workers if num_workers is None else num_workers,
        collate_fn=collate,
        drop_last=(split == "train"),
        persistent_workers=False,
    )
