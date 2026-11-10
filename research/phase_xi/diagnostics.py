def cohen_d(pos, neg):
    import numpy as np

    pos = np.asarray(pos, dtype=float)
    neg = np.asarray(neg, dtype=float)
    pooled = ((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0) ** 0.5 if len(pos) > 1 and len(neg) > 1 else 0.0
    return 0.0 if pooled == 0.0 else float((pos.mean() - neg.mean()) / pooled)
