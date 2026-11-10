from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def score_distribution(scores: Iterable[float], labels: Iterable[int]) -> Dict:
    s = np.asarray(list(scores), dtype=float)
    y = np.asarray(list(labels), dtype=int)
    out = {}
    for name, cls in [("normal", 0), ("anomaly", 1)]:
        vals = s[y == cls]
        out[f"score_mean_{name}"] = float(np.mean(vals)) if vals.size else float("nan")
        out[f"score_std_{name}"] = float(np.std(vals)) if vals.size else float("nan")
    out["score_gap"] = out["score_mean_anomaly"] - out["score_mean_normal"]
    return out


def rmse(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2))) if mask.any() else float("nan")
