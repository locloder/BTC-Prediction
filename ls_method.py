import math
from datetime import timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ---------- Helpers ----------

_INTERVAL_TO_STEP = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "5d": timedelta(days=5),
    "1wk": timedelta(weeks=1),
    "1mo": timedelta(days=30),
}

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DateTimeIndex.")
    d = df.copy()
    if d.index.tz is not None:
        d.index = d.index.tz_convert('UTC').tz_localize(None)
    if "Close" not in d.columns:
        raise ValueError("DataFrame must contain a 'Close' column.")
    return d

def _as_1d(a) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1)

def _xy_from_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    d = _ensure_datetime_index(df)
    t0 = d.index[0]
    x = _as_1d((d.index - t0).total_seconds()) / 3600.0
    y = _as_1d(d["Close"].values)
    return x, y, t0

def _future_times(last_time: pd.Timestamp, n_future: int, interval: str) -> List[pd.Timestamp]:
    step = _INTERVAL_TO_STEP.get(interval)
    if step is None:
        raise ValueError(f"Unsupported interval '{interval}'")
    n_future = max(1, int(n_future))
    return [last_time + step * (i + 1) for i in range(n_future)]

# ---------- Models ----------

def _fit_poly(x, y, deg, x_eval):
    x = _as_1d(x); y = _as_1d(y); x_eval = _as_1d(x_eval)
    coeffs = np.polyfit(x, y, deg=int(deg))
    return np.polyval(coeffs, x_eval), {"degree": int(deg), "coeffs": coeffs}

def _fit_exp(x, y, x_eval):
    x = _as_1d(x); y = _as_1d(y); x_eval = _as_1d(x_eval)
    mask = y > 0
    if mask.sum() < 2:
        return np.full_like(x_eval, y.mean(), dtype=float), {"note": "insufficient positive y"}
    X = np.vstack([np.ones(mask.sum()), x[mask]]).T
    beta, *_ = np.linalg.lstsq(X, np.log(y[mask]), rcond=None)
    ln_a, b = beta
    a = np.exp(ln_a)
    return a * np.exp(b * x_eval), {"a": float(a), "b": float(b)}

def _fit_log(x, y, x_eval, eps=1.0):
    x = _as_1d(x); y = _as_1d(y); x_eval = _as_1d(x_eval)
    xs = x - x.min() + eps
    X = np.vstack([np.ones_like(xs), np.log(xs)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta
    xs_eval = x_eval - x.min() + eps
    return a + b * np.log(xs_eval), {"a": float(a), "b": float(b), "eps": float(eps)}

def _fit_power(x, y, x_eval, eps=1.0):
    x = _as_1d(x); y = _as_1d(y); x_eval = _as_1d(x_eval)
    xs = x - x.min() + eps
    mask = (y > 0) & (xs > 0)
    if mask.sum() < 2:
        return np.full_like(x_eval, y.mean(), dtype=float), {"note": "insufficient y/x for log"}
    X = np.vstack([np.ones(mask.sum()), np.log(xs[mask])]).T
    beta, *_ = np.linalg.lstsq(X, np.log(y[mask]), rcond=None)
    ln_a, b = beta
    a = np.exp(ln_a)
    xs_eval = x_eval - x.min() + eps
    return a * (xs_eval ** b), {"a": float(a), "b": float(b), "eps": float(eps)}

# ---------- Public API ----------

DEFAULT_CONFIG = {
    "poly_deg": 3,
    "enable": {
        "polynomial": True,
        "exponential": True,
        "logarithmic": True,
        "powerlaw": True,
    }
}

def fit_and_forecast(
    df: pd.DataFrame,
    interval: str,
    n_future: int,
    config: Dict = None,
) -> Dict:
    if config is None:
        config = DEFAULT_CONFIG
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    enabled = {**DEFAULT_CONFIG["enable"], **cfg.get("enable", {})}

    d = _ensure_datetime_index(df)
    x_hist, y_hist, _ = _xy_from_df(d)
    last_time = d.index[-1]
    f_times = _future_times(last_time, n_future, interval)

    step_hours = _INTERVAL_TO_STEP[interval].total_seconds() / 3600.0
    x_last = float(x_hist[-1])
    x_future = np.arange(x_last + step_hours, x_last + step_hours * (int(n_future) + 1), step_hours, dtype=float)

    out = {"future_times": f_times, "hist_times": list(d.index), "models": {}}

    if enabled.get("polynomial", False):
        hist, _ = _fit_poly(x_hist, y_hist, cfg["poly_deg"], x_hist)
        fut, _ = _fit_poly(x_hist, y_hist, cfg["poly_deg"], x_future)
        out["models"]["Polynomial"] = {"hist": hist, "future": fut}

    if enabled.get("exponential", False):
        hist, _ = _fit_exp(x_hist, y_hist, x_hist)
        fut, _ = _fit_exp(x_hist, y_hist, x_future)
        out["models"]["Exponential"] = {"hist": hist, "future": fut}

    if enabled.get("logarithmic", False):
        hist, _ = _fit_log(x_hist, y_hist, x_hist)
        fut, _ = _fit_log(x_hist, y_hist, x_future)
        out["models"]["Logarithmic"] = {"hist": hist, "future": fut}

    if enabled.get("powerlaw", False):
        hist, _ = _fit_power(x_hist, y_hist, x_hist)
        fut, _ = _fit_power(x_hist, y_hist, x_future)
        out["models"]["Power-law"] = {"hist": hist, "future": fut}

    return out


def draw_approximations(ax, df: pd.DataFrame, interval: str, result: Dict):
    hist_times = pd.to_datetime(result["hist_times"])
    fut_times = pd.to_datetime(result["future_times"])

    # Define consistent style per model
    base_styles = {
        "Polynomial":  dict(color="#FFFFFF", lw=1.5),
        "Exponential": dict(color="#2ca02c", lw=1.5),
        "Logarithmic": dict(color="#1f77b4", lw=1.5),
        "Power-law":   dict(color="#d62728", lw=1.5),
    }

    for name, d in result["models"].items():
        st = base_styles.get(name, {"lw": 1.2})

        # Draw the fit and forecast using same color
        fit_line, = ax.plot(hist_times, d["hist"], linestyle="-", **st)
        ax.plot(fut_times, d["future"], linestyle="--", color=fit_line.get_color(), lw=st.get("lw", 1.2))

        # Only add one legend entry (solid line)
        fit_line.set_label(name)