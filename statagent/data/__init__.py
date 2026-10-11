"""Data layer: file loading, automatic profiling, dataset store, synthetic data."""
from statagent.data.loader import DataLoadError, load_dataframe
from statagent.data.profiler import profile_frame
from statagent.data.store import DatasetStore

__all__ = ["DataLoadError", "DatasetStore", "load_dataframe", "profile_frame"]
