import itertools

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, dim=-1)


def repeat_to_k(x: torch.Tensor, k: int) -> torch.Tensor:
    if x.shape[0] == k:
        return x
    if x.shape[0] > k:
        return x[:k]
    reps = []
    idx = 0
    while len(reps) < k:
        reps.append(x[idx % x.shape[0]])
        idx += 1
    return torch.stack(reps, dim=0)


def mechanism_metrics(tokens: torch.Tensor, prototypes: torch.Tensor) -> dict:
    tokens = normalize(tokens.float())
    prototypes = normalize(prototypes.float())
    sim = tokens @ prototypes.T
    nearest_sim = sim.max(dim=1).values
    distances = 1.0 - nearest_sim

    if prototypes.shape[0] > 1:
        pair_sims = []
        duplicate_pairs = 0
        total_pairs = 0
        for i, j in itertools.combinations(range(prototypes.shape[0]), 2):
            s = float((prototypes[i] * prototypes[j]).sum().item())
            pair_sims.append(1.0 - s)
            duplicate_pairs += int(s > 0.999)
            total_pairs += 1
        pairwise_diversity = float(sum(pair_sims) / len(pair_sims))
        duplication_ratio = float(duplicate_pairs / max(1, total_pairs))
    else:
        pairwise_diversity = 0.0
        duplication_ratio = 0.0

    mean_proto = normalize(tokens.mean(dim=0, keepdim=True))
    total = (1.0 - (tokens @ mean_proto.T).squeeze(-1)).mean().item()
    recon = distances.mean().item()
    if total <= 1e-12:
        variance_coverage = 1.0 if recon <= 1e-12 else 0.0
    else:
        variance_coverage = max(0.0, min(1.0, 1.0 - recon / total))

    return {
        "mean_reconstruction_distance": float(recon),
        "maximum_coverage_radius": float(distances.max().item()),
        "pairwise_prototype_diversity": pairwise_diversity,
        "region_variance_coverage": float(variance_coverage),
        "prototype_duplication_ratio": duplication_ratio,
    }
