import numpy as np
import torch


def fit_kmeans(mu, k, seed=0, tol=1e-4, n_init=10):
    from sklearn.cluster import KMeans

    x = mu.detach().float().cpu().numpy()
    k = min(int(k), x.shape[0])
    km = KMeans(n_clusters=k, init="k-means++", random_state=int(seed), n_init=n_init, tol=float(tol))
    assign = km.fit_predict(x)
    return {
        "codebook": torch.from_numpy(km.cluster_centers_).float(),
        "assignments": torch.from_numpy(assign).long(),
        "inertia": float(km.inertia_),
        "tol": float(tol),
        "n_init": int(n_init),
    }


def reconstruct_codebook(fit):
    return fit["codebook"][fit["assignments"]]


def random_assignment(assignments, seed):
    gen = torch.Generator().manual_seed(int(seed))
    return assignments[torch.randperm(assignments.numel(), generator=gen)]


def random_codebook(mu, k, seed):
    gen = torch.Generator().manual_seed(int(seed))
    k = min(int(k), mu.shape[0])
    idx = torch.randperm(mu.shape[0], generator=gen)[:k]
    codebook = mu[idx].float().clone()
    dist = torch.cdist(mu.float(), codebook.float())
    assignments = dist.argmin(dim=1)
    return {"codebook": codebook, "assignments": assignments}


def fit_offset(mu, base_mu, max_rank):
    from shared_residual_memory.covariance import covariance_pca

    residual = mu.float() - base_mu.float()
    basis, top, all_eigs = covariance_pca(residual, max_rank)
    coeff = residual @ basis
    return {"basis": basis, "coeff": coeff, "eigvals": all_eigs, "top_eigvals": top}


def add_offset(base_mu, offset_fit, rank):
    basis = offset_fit["basis"][:, :rank]
    coeff = offset_fit["coeff"][:, :rank]
    return base_mu.float() + coeff @ basis.T


def utilization(assignments, k):
    counts = torch.bincount(assignments.cpu(), minlength=int(k)).numpy().astype(np.float64)
    used = int((counts > 0).sum())
    if counts.sum() == 0:
        gini = float("nan")
    else:
        sorted_counts = np.sort(counts)
        n = len(sorted_counts)
        gini = float((2 * np.arange(1, n + 1) @ sorted_counts) / (n * sorted_counts.sum()) - (n + 1) / n)
    return {"used_codewords": used, "empty_codewords": int(k - used), "usage_gini": gini}

