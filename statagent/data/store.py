"""In-memory dataset store shared by the UI, the agent pipeline and tools."""
from __future__ import annotations

import pandas as pd

from statagent.data.profiler import profile_frame
from statagent.models.dataset import DatasetProfile


class DatasetStore:
    """Holds DataFrames by id and caches their profiles."""

    def __init__(self) -> None:
        self._frames: dict[str, pd.DataFrame] = {}
        self._sources: dict[str, str] = {}
        self._profiles: dict[str, DatasetProfile] = {}
        self._counter = 0

    def add(self, df: pd.DataFrame, name: str) -> str:
        self._counter += 1
        dataset_id = f"ds{self._counter}"
        self._frames[dataset_id] = df.copy()
        self._sources[dataset_id] = name
        self._profiles.pop(dataset_id, None)
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame:
        if dataset_id not in self._frames:
            raise KeyError(
                f"Unknown dataset '{dataset_id}'. Available: {sorted(self._frames)}"
            )
        return self._frames[dataset_id]

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._frames

    def profile(self, dataset_id: str) -> DatasetProfile:
        if dataset_id not in self._frames:
            raise KeyError(f"Unknown dataset '{dataset_id}'.")
        if dataset_id not in self._profiles:
            self._profiles[dataset_id] = profile_frame(
                self._frames[dataset_id], dataset_id, self._sources[dataset_id]
            )
        return self._profiles[dataset_id]

    def invalidate(self, dataset_id: str) -> None:
        self._profiles.pop(dataset_id, None)

    def remove(self, dataset_id: str) -> None:
        self._frames.pop(dataset_id, None)
        self._sources.pop(dataset_id, None)
        self._profiles.pop(dataset_id, None)

    def ids(self) -> list[str]:
        return list(self._frames)

    def source_name(self, dataset_id: str) -> str:
        return self._sources.get(dataset_id, dataset_id)

    def list_profiles(self) -> list[DatasetProfile]:
        return [self.profile(i) for i in self._frames]

    def __len__(self) -> int:
        return len(self._frames)
