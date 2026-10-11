"""JSON serialization helpers.

Statistical libraries return numpy scalars which are not directly JSON
serializable. Everything persisted (reports, benchmark logs) passes through
:func:`to_jsonable` first.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert an object into plain JSON-safe Python values."""
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return to_jsonable(obj.item())
    return str(obj)


def json_dumps(obj: Any, **kwargs: Any) -> str:
    kwargs.setdefault("indent", 2)
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(to_jsonable(obj), **kwargs)


def json_loads(text: str) -> Any:
    return json.loads(text)
