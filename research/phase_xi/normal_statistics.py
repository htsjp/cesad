import numpy as np

from phase_xi.shared_anomaly_basis import pca_basis


def category_normal_stats(features_by_category, rank):
    stats = {}
    for category, values in features_by_category.items():
        x = np.asarray(values, dtype=np.float32)
        mu = x.mean(axis=0).astype(np.float32)
        basis, eig, evr = pca_basis(x - mu, rank)
        stats[category] = {"mu": mu, "basis": basis, "eig": eig, "evr": evr}
    return stats
