"""Bootstrap confidence intervals with a user-settable random seed."""
from __future__ import annotations

from typing import Literal

import numpy as np
import plotly.graph_objects as go
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.common import COLOR, COLOR_2, base_figure, clean_pair, code_snippet, get_series, to_numeric
from statagent.stats.engine import Params, ToolRegistry


class BootstrapParams(Params):
    dataset_id: str
    column: str
    statistic: Literal["mean", "median", "std", "correlation"] = "mean"
    # For statistic="correlation": the second column to correlate with.
    x: str | None = None
    y: str | None = None
    method: Literal["pearson", "spearman"] = "pearson"
    n_boot: int = 2000
    seed: int = 42
    alpha: float = 0.05


@ToolRegistry.register(
    "bootstrap_ci",
    "Bootstrap confidence interval for a statistic (mean, median, standard "
    "deviation, or correlation) by resampling with replacement. "
    "Nonparametric: no distributional assumptions. The random seed is "
    "configurable for reproducible runs.",
    BootstrapParams,
    "Returns the observed estimate, bootstrap bias, bootstrap SE, and a "
    "percentile CI, plus a histogram of the bootstrap distribution.",
)
def bootstrap_ci(store: DatasetStore, params: BootstrapParams) -> dict:
    if not (100 <= params.n_boot <= 50000):
        raise ValueError("n_boot must be between 100 and 50000.")
    if not (0.0 < params.alpha < 0.5):
        raise ValueError("alpha must be between 0 and 0.5.")

    df = store.get(params.dataset_id)
    if params.statistic == "correlation":
        if not params.x or not params.y:
            raise ValueError("statistic='correlation' requires both x and y.")
        x, y, n = clean_pair(df, params.x, params.y)
        data = np.column_stack([x, y])

        def stat_fn(idx: np.ndarray) -> float:
            sub = data[idx]
            if params.method == "pearson":
                r, _ = stats.pearsonr(sub[:, 0], sub[:, 1])
            else:
                r, _ = stats.spearmanr(sub[:, 0], sub[:, 1])
            return float(r)

        estimate = stat_fn(np.arange(n))
        label = f"{params.method} correlation({params.x}, {params.y})"
    else:
        s = get_series(store, params.dataset_id, params.column)
        num = to_numeric(s, params.column).dropna().to_numpy(float)
        n = len(num)
        if n < 5:
            raise ValueError("Bootstrap requires at least 5 observations.")
        stat_fn_map = {
            "mean": np.mean,
            "median": np.median,
            "std": lambda a: np.std(a, ddof=1),
        }
        stat_fn = stat_fn_map[params.statistic]
        estimate = float(stat_fn(num))
        data = num
        label = f"{params.statistic}({params.column})"

    rng = np.random.default_rng(params.seed)
    boot = np.empty(params.n_boot)
    for i in range(params.n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stat_fn(idx)

    lo, hi = np.percentile(boot, [100 * params.alpha / 2, 100 * (1 - params.alpha / 2)])
    bias = float(boot.mean() - estimate)
    se = float(boot.std(ddof=1))

    fig = base_figure(f"Bootstrap distribution of {label}")
    fig.add_trace(go.Histogram(x=boot, nbinsx=40, marker_color=COLOR, name="bootstrap"))
    for v, name in ((lo, "2.5%"), (hi, "97.5%"), (estimate, "observed")):
        fig.add_vline(x=v, line=dict(color=COLOR_2, dash="dash"), annotation_text=name)
    fig.update_xaxes(title="statistic")

    def _r(v):
        return round(float(v), 6)

    return {
        "summary": {
            "statistic": label,
            "estimate": _r(estimate),
            "ci95": [_r(lo), _r(hi)],
            "bias": _r(bias),
            "bootstrap_se": _r(se),
            "n_boot": params.n_boot,
            "seed": params.seed,
            "n": n,
        },
        "details": {
            "estimate": _r(estimate),
            "ci95": [_r(lo), _r(hi)],
            "bias": _r(bias),
            "bootstrap_se": _r(se),
        },
        "figures": [fig],
        "notes": [
            f"Bootstrap percentile CI for {label}: ({lo:.4g}, {hi:.4g}) with "
            f"{params.n_boot} resamples, seed {params.seed}."
        ],
        "code": code_snippet(
            "statagent.stats.bootstrap.bootstrap_ci",
            {
                "dataset_id": params.dataset_id,
                "column": params.column,
                "statistic": params.statistic,
                "x": params.x,
                "y": params.y,
                "n_boot": params.n_boot,
                "seed": params.seed,
                "alpha": params.alpha,
            },
        ),
        "n": n,
    }
