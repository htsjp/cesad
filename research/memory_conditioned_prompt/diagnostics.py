import numpy as np


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "max": float("nan"), "std": float("nan")}
    return {"mean": float(arr.mean()), "max": float(arr.max()), "std": float(arr.std())}


def metric_value(metrics, key):
    return float(metrics.get("mean", {}).get(key, float("nan")))

