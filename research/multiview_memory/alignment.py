import hashlib
from typing import Dict, List

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def fused_baseline_view(t24: torch.Tensor, v12: torch.Tensor, v18: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    value = normalize(v12 + v18)
    return normalize(normalize(t24) + float(alpha) * value)


def fps_indices(tokens: torch.Tensor, k: int) -> torch.Tensor:
    x = normalize(tokens)
    if x.shape[0] <= k:
        repeats = [i % x.shape[0] for i in range(k)]
        return torch.tensor(repeats, device=x.device, dtype=torch.long)
    selected = [0]
    dist = 1.0 - (x @ x[0:1].T).squeeze(-1)
    for _ in range(1, k):
        idx = int(torch.argmax(dist).item())
        selected.append(idx)
        dist = torch.minimum(dist, 1.0 - (x @ x[idx : idx + 1].T).squeeze(-1))
    return torch.tensor(selected, device=x.device, dtype=torch.long)


def checksum_tensor(x: torch.Tensor) -> str:
    data = x.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def build_alignment_entry(
    memory_id: int,
    sample_id: str,
    region_id: int,
    region_type: str,
    patch_index: int,
    side: int,
    label: int,
) -> Dict:
    return {
        "memory_id": int(memory_id),
        "source_id": str(sample_id),
        "region_id": int(region_id),
        "region_type": str(region_type),
        "patch_index": int(patch_index),
        "patch_y": int(patch_index // side),
        "patch_x": int(patch_index % side),
        "label": int(label),
    }


def summarize_alignment(memory: Dict, entries: List[Dict]) -> Dict:
    labels = memory["patch_values"].argmax(dim=-1)
    return {
        "memory_entries": int(memory["T24"].shape[0]),
        "entry_records": int(len(entries)),
        "labels_match_records": bool(len(entries) == labels.numel() and all(int(e["label"]) == int(labels[i]) for i, e in enumerate(entries))),
        "ordering": "baseline fused FPS_K4 order within source loader, region order anomaly then normal",
        "views": ["T24", "V12", "V18"],
        "all_views_l2_normalized": bool(
            all(torch.allclose(memory[k].norm(dim=-1), torch.ones(memory[k].shape[0], device=memory[k].device), atol=2e-3, rtol=2e-3) for k in ("T24", "V12", "V18"))
        ),
        "checksums": {k: checksum_tensor(memory[k]) for k in ("T24", "V12", "V18", "patch_values")},
    }

