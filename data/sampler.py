from __future__ import annotations

"""Deterministic episode-order sampler; an episode is never split across videos."""

from collections.abc import Iterator, Sized
import numpy as np


class EpisodeSampler:
    def __init__(self, data_source: Sized, *, shuffle: bool = True, seed: int = 0) -> None:
        self.data_source = data_source
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(len(self.data_source), dtype=np.int64)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        return iter(indices.tolist())

    def __len__(self) -> int:
        return len(self.data_source)
