from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, List

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(values) -> str:
    arr = np.asarray(values)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def safe_corr(x: Iterable[float], y: Iterable[float], method: str) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    a = a[mask]
    b = b[mask]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    if method == "pearson":
        return float(np.corrcoef(a, b)[0, 1])
    if method == "spearman":
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).correlation)
    raise ValueError(method)


def mae(x: Iterable[float], y: Iterable[float]) -> float:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float("nan")
