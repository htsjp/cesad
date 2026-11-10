import numpy as np


def gini(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0
    arr = np.sort(np.abs(arr))
    n = arr.size
    return float((2 * np.arange(1, n + 1) @ arr) / (n * arr.sum()) - (n + 1) / n)


def summarize_usage(top1, labels, memory_size, region_count, variant="", direction="", seed=0, split=""):
    top1 = np.asarray(top1, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    proto_counts = np.bincount(top1, minlength=memory_size) if top1.size else np.zeros(memory_size)
    region_top = top1 // 4 if top1.size else np.zeros(0, dtype=np.int64)
    region_counts = np.bincount(region_top, minlength=region_count) if top1.size else np.zeros(region_count)
    return {
        "seed": seed,
        "direction": direction,
        "split": split,
        "variant": variant,
        "unique_prototypes_used": int((proto_counts > 0).sum()),
        "unique_regions_used": int((region_counts > 0).sum()),
        "never_used_prototype_fraction": float((proto_counts == 0).mean()),
        "never_used_region_fraction": float((region_counts == 0).mean()),
        "usage_gini": gini(region_counts),
        "maximum_region_usage": int(region_counts.max()) if region_counts.size else 0,
        "top1_anomaly_fraction": float(labels[top1].mean()) if top1.size else float("nan"),
    }

