import numpy as np
import torch
import torch.nn.functional as F


def centroid_diagnostics(name, true_mu, pred_mu, family, rank="", k=""):
    true = true_mu.float()
    pred = pred_mu.float()
    cos = F.cosine_similarity(true, pred, dim=-1)
    return {
        "variant": name,
        "family": family,
        "rank": rank,
        "K": k,
        "centroid_mse": float((true - pred).pow(2).mean().item()),
        "centroid_cosine": float(cos.mean().item()),
        "worst_centroid_cosine": float(cos.min().item()),
    }


def virtual_diagnostics(name, full_keys, pred_keys):
    cos = F.cosine_similarity(full_keys.float(), pred_keys.float(), dim=-1)
    sims = pred_keys.float().reshape(-1, 4, pred_keys.shape[-1])
    pair = []
    for i in range(4):
        for j in range(i + 1, 4):
            pair.append(F.cosine_similarity(sims[:, i], sims[:, j], dim=-1))
    pair = torch.stack(pair, dim=1)
    return {
        "variant": name,
        "virtual_prototype_cosine_to_fullfps": float(cos.mean().item()),
        "worst_virtual_prototype_cosine_to_fullfps": float(cos.min().item()),
        "inter_prototype_diversity": float((1.0 - pair).mean().item()),
        "near_duplicate_ratio": float((pair > 0.999).float().mean().item()),
    }


def metric_delta(row, full_row):
    out = dict(row)
    for key in ("I-AUROC", "P-AUROC", "PRO"):
        out[f"dFull {key.split('-')[0] if key != 'PRO' else 'PRO'}"] = float(row[key]) - float(full_row[key])
    return out


def control_summary(rows, candidate, metric="PRO"):
    vals = np.asarray([float(r[metric]) for r in rows if np.isfinite(float(r[metric]))], dtype=np.float64)
    if vals.size == 0:
        return {"metric": metric, "control_count": 0}
    cand = float(candidate[metric])
    return {
        "metric": metric,
        "control_count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "median": float(np.median(vals)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "candidate_percentile": float((vals <= cand).mean()),
        "empirical_one_sided_p": float((1 + (vals >= cand).sum()) / (vals.size + 1)),
    }

