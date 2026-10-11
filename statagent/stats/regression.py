"""OLS linear regression with residual diagnostics.

Implemented with numpy/scipy formulas (verified against statsmodels in the
test suite) so the engine works on any supported Python version. Categorical
predictors are one-hot encoded with the first level as reference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.common import COLOR, COLOR_2, base_figure, code_snippet, fmt_p
from statagent.stats.engine import Params, ToolRegistry


class RegressionParams(Params):
    dataset_id: str
    y: str
    x: list[str]
    categorical: list[str] = []


def _vif(x_matrix: np.ndarray, names: list[str]) -> dict[str, float]:
    """Variance inflation factors for non-intercept columns."""
    out: dict[str, float] = {}
    k = x_matrix.shape[1]
    for j in range(1, k):
        y_j = x_matrix[:, j]
        others = np.delete(x_matrix, j, axis=1)
        beta, *_ = np.linalg.lstsq(others, y_j, rcond=None)
        resid = y_j - others @ beta
        ss_res = float(resid @ resid)
        ss_tot = float(((y_j - y_j.mean()) ** 2).sum())
        r2j = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out[names[j]] = 1e6 if r2j >= 0.9999 else round(1 / (1 - r2j), 3)
    return out


def _qq_figure(resid: np.ndarray, title: str) -> go.Figure:
    n = len(resid)
    observed = np.sort(resid)
    expected = stats.norm.ppf((np.arange(1, n + 1) - 0.375) / (n + 0.25))
    fig = base_figure(title)
    fig.add_trace(go.Scatter(x=expected, y=observed, mode="markers", marker=dict(color=COLOR, size=6)))
    lo, hi = expected.min(), expected.max()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo * observed.std(ddof=1), hi * observed.std(ddof=1)],
            mode="lines",
            line=dict(color=COLOR_2, dash="dash"),
            name="normal line",
        )
    )
    fig.update_xaxes(title="Theoretical normal quantiles")
    fig.update_yaxes(title="Residual quantiles")
    return fig


@ToolRegistry.register(
    "linear_regression",
    "Ordinary least squares regression of a numeric outcome y on one or "
    "more predictors x. Categorical predictors (list their names in "
    "'categorical') are one-hot encoded automatically. Reports coefficients, "
    "standard errors, p-values, 95% CIs, R-squared, adjusted R-squared, "
    "F-test, AIC/BIC, and residual diagnostics (Durbin-Watson, "
    "Jarque-Bera, Breusch-Pagan, VIF, Cook's distance).",
    RegressionParams,
    "Returns the model table and diagnostics plus diagnostic plots. "
    "Regression describes ASSOCIATION, not causation, unless the design is "
    "experimental and confounders are controlled.",
)
def linear_regression(store: DatasetStore, params: RegressionParams) -> dict:
    df = store.get(params.dataset_id)
    if params.y not in df.columns:
        raise ValueError(f"Outcome '{params.y}' not found. Available: {list(df.columns)}")
    if not params.x:
        raise ValueError("linear_regression needs at least one predictor in x.")
    for c in params.x:
        if c not in df.columns:
            raise ValueError(f"Predictor '{c}' not found. Available: {list(df.columns)}")

    numeric_x = [c for c in params.x if c not in params.categorical]
    parts: list[pd.Series] = []
    dummy_map: dict[str, str] = {}
    for c in numeric_x:
        parts.append(pd.to_numeric(df[c], errors="coerce").rename(c))
    for c in params.categorical:
        dummies = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=True, dtype=float)
        for col in dummies.columns:
            dummy_map[col] = c
        parts += [dummies[col] for col in dummies.columns]
    if not parts:
        raise ValueError("No usable predictors after processing.")

    X_df = pd.concat(parts, axis=1)
    data = pd.concat([X_df, pd.to_numeric(df[params.y], errors="coerce").rename(params.y)], axis=1)
    data = data.dropna()
    n = len(data)
    names = ["Intercept"] + list(X_df.columns)
    X = np.column_stack([np.ones(n), data[X_df.columns].to_numpy(float)])
    yv = data[params.y].to_numpy(float)
    k = X.shape[1]
    if n <= k + 1:
        raise ValueError(f"Not enough complete observations for regression: n={n}, parameters={k}.")

    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    df_resid = n - k
    sigma2 = float(resid @ resid / df_resid)
    try:
        cov = np.linalg.inv(X.T @ X) * sigma2
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(X.T @ X) * sigma2
    se = np.sqrt(np.diag(cov))
    t_vals = beta / se
    p_vals = 2 * stats.t.sf(np.abs(t_vals), df_resid)
    crit = stats.t.ppf(0.975, df_resid)
    ci_lo, ci_hi = beta - crit * se, beta + crit * se

    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid
    f_stat = (r2 / (k - 1)) / ((1 - r2) / df_resid) if r2 < 1 else float("inf")
    p_f = float(stats.f.sf(f_stat, k - 1, df_resid)) if np.isfinite(f_stat) else 0.0
    aic = n * np.log(ss_res / n) + 2 * k
    bic = n * np.log(ss_res / n) + np.log(n) * k

    # --- diagnostics ---
    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid**2))
    s_skew, s_kurt = float(stats.skew(resid)), float(stats.kurtosis(resid, fisher=True))
    jb = n / 6 * (s_skew**2 + s_kurt**2 / 4)
    p_jb = float(stats.chi2.sf(jb, 2))
    aux_beta, *_ = np.linalg.lstsq(X, resid**2, rcond=None)
    aux_resid = resid**2 - X @ aux_beta
    ss_aux = float(aux_resid @ aux_resid)
    ss_tot_aux = float(((resid**2 - (resid**2).mean()) ** 2).sum())
    r2_aux = 1 - ss_aux / ss_tot_aux if ss_tot_aux > 0 else 0.0
    bp = n * r2_aux
    p_bp = float(stats.chi2.sf(bp, k - 1))
    vif = _vif(X, names)
    h = np.diag(X @ np.linalg.pinv(X.T @ X) @ X.T)
    cook = resid**2 / (k * sigma2) * (h / (1 - h) ** 2)

    def _r(v):
        return round(float(v), 6)

    coef_rows = []
    for i, name in enumerate(names):
        row = {
            "variable": name,
            "coefficient": _r(beta[i]),
            "se": _r(se[i]),
            "t": _r(t_vals[i]),
            "p_value": _r(p_vals[i]),
            "ci95": [_r(ci_lo[i]), _r(ci_hi[i])],
        }
        coef_rows.append(row)
        if dummy_map.get(name):
            row["encodes"] = dummy_map[name]

    notes: list[str] = []
    if p_bp < 0.05:
        notes.append("Breusch-Pagan p < 0.05: heteroscedastic residuals (non-constant variance).")
    if p_jb < 0.05:
        notes.append("Jarque-Bera p < 0.05: residuals deviate from normality.")
    high_vif = {k_: v for k_, v in vif.items() if v > 10}
    if high_vif:
        notes.append(f"High VIF (collinearity): {high_vif}.")
    if dummy_map:
        notes.append("Categorical predictors one-hot encoded; first level is the reference.")

    # --- figures ---
    figures: list[go.Figure] = []
    if len(numeric_x) == 1 and not params.categorical:
        xvals = data[numeric_x[0]].to_numpy(float)
        fig = base_figure(f"Regression: {params.y} ~ {numeric_x[0]}")
        order = np.argsort(xvals)
        fig.add_trace(
            go.Scatter(
                x=xvals, y=yv, mode="markers", marker=dict(color=COLOR, size=7, opacity=0.6),
                name="observations",
            )
        )
        xs = np.linspace(xvals.min(), xvals.max(), 200)
        Xs = np.column_stack([np.ones(200), xs])
        fit = Xs @ beta
        hh = np.diag(Xs @ np.linalg.pinv(X.T @ X) @ Xs.T)
        band = crit * np.sqrt(sigma2 * hh)
        fig.add_trace(
            go.Scatter(x=xs, y=fit, mode="lines", line=dict(color=COLOR_2, width=2), name="fit")
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([xs, xs[::-1]]),
                y=np.concatenate([fit + band, (fit - band)[::-1]]),
                fill="toself", fillcolor="rgba(232,89,12,0.15)", line=dict(width=0),
                name="95% CI",
            )
        )
        fig.update_xaxes(title=numeric_x[0])
        fig.update_yaxes(title=params.y)
        figures.append(fig)
    else:
        fig = base_figure("Regression coefficients (95% CI)")
        non_int = [r for r in coef_rows if r["variable"] != "Intercept"]
        fig.add_trace(
            go.Bar(
                x=[r["coefficient"] for r in non_int],
                y=[r["variable"] for r in non_int],
                orientation="h",
                marker_color=COLOR,
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[r["coefficient"] - r["ci95"][0] for r in non_int],
                    arrayminus=[r["ci95"][1] - r["coefficient"] for r in non_int],
                ),
            )
        )
        fig.update_xaxes(title="coefficient")
        figures.append(fig)

    fit_vals = X @ beta
    fig_res = base_figure("Residuals vs fitted values")
    fig_res.add_trace(
        go.Scatter(x=fit_vals, y=resid, mode="markers", marker=dict(color=COLOR, size=6, opacity=0.6))
    )
    fig_res.add_hline(y=0, line=dict(color=COLOR_2, dash="dash"))
    fig_res.update_xaxes(title="Fitted values")
    fig_res.update_yaxes(title="Residuals")
    figures.append(fig_res)
    figures.append(_qq_figure(resid, "Normal Q-Q plot of residuals"))

    summary = {
        "n": n,
        "k": k,
        "r_squared": _r(r2),
        "adj_r_squared": _r(adj_r2),
        "f_stat": _r(f_stat),
        "p_f": _r(p_f),
        "aic": _r(aic),
        "bic": _r(bic),
        "coefficients": {
            r["variable"]: {
                "coefficient": r["coefficient"],
                "se": r["se"],
                "t": r["t"],
                "p_value": r["p_value"],
                "ci95": r["ci95"],
            }
            for r in coef_rows
        },
    }
    details = {
        "coefficients": coef_rows,
        "r_squared": _r(r2),
        "adj_r_squared": _r(adj_r2),
        "f_stat": _r(f_stat),
        "p_f": _r(p_f),
        "p_f_formatted": fmt_p(p_f),
        "aic": _r(aic),
        "bic": _r(bic),
        "diagnostics": {
            "durbin_watson": _r(dw),
            "jarque_bera": _r(jb),
            "jarque_bera_p": _r(p_jb),
            "breusch_pagan": _r(bp),
            "breusch_pagan_p": _r(p_bp),
            "vif": vif,
            "cook_max": _r(float(cook.max())),
        },
    }
    return {
        "summary": summary,
        "details": details,
        "figures": figures,
        "notes": notes,
        "code": code_snippet(
            "statagent.stats.regression.linear_regression",
            {
                "dataset_id": params.dataset_id,
                "y": params.y,
                "x": params.x,
                "categorical": params.categorical,
            },
        ),
        "n": n,
    }
