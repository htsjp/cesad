import hashlib
import json
import os
from pathlib import Path
from typing import Dict

import torch


def tensor_checksum(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()


def cache_path(root: Path, dataset: str, split: str, sample_id: str, layer: int, rep: str) -> Path:
    safe = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:16]
    return root / dataset / split / f"layer_{layer}" / rep / f"{safe}.pt"


def save_feature_cache(root: Path, dataset: str, split: str, sample_id: str, layer: int, rep: str, tensor: torch.Tensor, normalized: bool):
    path = cache_path(root, dataset, split, sample_id, layer, rep)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_image_id": sample_id,
        "layer": int(layer),
        "feature_type": rep,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "normalization_status": bool(normalized),
        "checksum": tensor_checksum(tensor),
        "tensor": tensor.detach().cpu(),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def verify_cache(path: Path, tensor: torch.Tensor, atol: float = 1e-6) -> Dict:
    payload = torch.load(path, map_location="cpu")
    cached = payload["tensor"]
    diff = float((cached.float() - tensor.detach().cpu().float()).abs().max().item())
    checksum_match = payload.get("checksum") == tensor_checksum(cached)
    return {"path": str(path), "max_abs_diff": diff, "checksum_match": bool(checksum_match), "match": diff <= atol and checksum_match}


def cache_size_mb(root: Path) -> float:
    if not root.exists():
        return 0.0
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total / (1024 * 1024)
