import hashlib
from typing import Dict

import torch


def tensor_checksum(x: torch.Tensor) -> str:
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def partition_memory(memory: Dict, labels_override: torch.Tensor = None) -> Dict:
    keys = memory["patch_keys"]
    labels = labels_override if labels_override is not None else memory["patch_values"].argmax(dim=-1)
    labels = labels.to(keys.device).long()
    normal_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()
    anomaly_idx = torch.nonzero(labels == 1, as_tuple=False).flatten()
    recon_idx = torch.cat([normal_idx, anomaly_idx], dim=0)
    order = torch.argsort(recon_idx)
    reconstructed = torch.cat([keys[normal_idx], keys[anomaly_idx]], dim=0)[order]
    return {
        "normal_keys": keys[normal_idx],
        "anomaly_keys": keys[anomaly_idx],
        "normal_indices": normal_idx,
        "anomaly_indices": anomaly_idx,
        "labels": labels,
        "normal_count": int(normal_idx.numel()),
        "anomaly_count": int(anomaly_idx.numel()),
        "feature_checksum": tensor_checksum(keys),
        "label_checksum": tensor_checksum(labels.cpu()),
        "reconstructs_original": bool(torch.allclose(reconstructed, keys, atol=1e-6)),
        "intersection_empty": bool(set(normal_idx.detach().cpu().tolist()).isdisjoint(set(anomaly_idx.detach().cpu().tolist()))),
    }


def random_label_partition(memory: Dict, seed: int) -> torch.Tensor:
    labels = memory["patch_values"].argmax(dim=-1).detach().cpu()
    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(labels.numel(), generator=g)
    shuffled = labels[perm].to(memory["patch_keys"].device)
    return shuffled


def matched_capacity_indices(partition: Dict, seed: int):
    n_idx = partition["normal_indices"].detach().cpu()
    a_idx = partition["anomaly_indices"].detach().cpu()
    m = min(int(n_idx.numel()), int(a_idx.numel()))
    g = torch.Generator().manual_seed(int(seed))
    n_sel = n_idx[torch.randperm(n_idx.numel(), generator=g)[:m]]
    a_sel = a_idx[torch.randperm(a_idx.numel(), generator=g)[:m]]
    return n_sel.to(partition["normal_indices"].device), a_sel.to(partition["anomaly_indices"].device)
