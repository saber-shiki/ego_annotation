from __future__ import annotations

"""Read-only lazy episode dataset with source-video-disjoint splits."""

from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any
import json

from data.build_index import INDEX_SCHEMA
from data.bundle import EpisodeBundle, file_sha256


def source_video_split(source_video_id: str, *, seed: int = 0, test_fraction: float = 0.2) -> str:
    if not 0.0 <= float(test_fraction) <= 1.0:
        raise ValueError("test_fraction must be in [0, 1]")
    digest = sha256(f"{int(seed)}:{source_video_id}".encode()).digest()
    uniform = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "test" if uniform < float(test_fraction) else "train"


class EpisodeDataset(Sequence[EpisodeBundle]):
    """Metadata stays resident; bundle arrays are loaded on first item use."""

    def __init__(
        self,
        index: str | Path | Mapping[str, Any],
        *,
        split: str = "all",
        seed: int = 0,
        test_fraction: float = 0.2,
        verify_manifests: bool = True,
    ) -> None:
        if split not in {"all", "train", "test"}:
            raise ValueError("split must be all/train/test")
        if isinstance(index, Mapping):
            payload = json.loads(json.dumps(index, allow_nan=False))
        else:
            payload = json.loads(Path(index).read_text(encoding="utf-8"))
        if payload.get("schema") != INDEX_SCHEMA or not isinstance(payload.get("episodes"), list):
            raise ValueError("invalid episode index")
        records = []
        for row in payload["episodes"]:
            if not isinstance(row, dict):
                raise ValueError("invalid episode index row")
            row_split = source_video_split(str(row["source_video_id"]), seed=seed, test_fraction=test_fraction)
            if split != "all" and row_split != split:
                continue
            path = Path(row["manifest"]).expanduser().resolve()
            if verify_manifests and file_sha256(path) != row["manifest_sha256"]:
                raise ValueError(f"manifest changed after indexing: {path}")
            records.append({**row, "manifest": path, "split": row_split})
        self._records = tuple(records)
        self._verify = bool(verify_manifests)
        self._cache: dict[int, EpisodeBundle] = {}

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> EpisodeBundle:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index not in self._cache:
            bundle = EpisodeBundle.open(self._records[index]["manifest"], verify=self._verify)
            if bundle.manifest.get("preprocessing_only") is True:
                raise ValueError("preprocessing-only v2 bundle requires a future explicit training adapter; use EpisodeBundle for observation inspection")
            self._cache[index] = bundle
        return self._cache[index]

    @property
    def source_video_ids(self) -> frozenset[str]:
        return frozenset(str(row["source_video_id"]) for row in self._records)

    @property
    def records(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(row) for row in self._records)
