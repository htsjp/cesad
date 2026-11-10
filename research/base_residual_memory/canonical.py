import hashlib
from typing import Dict

import torch
import torch.nn.functional as F


FORMULA_NAME = "F1_PhaseIID_T24_plus_mean_normalized_V12_V18_alpha0.5"
FORMULA_TEXT = "normalize(normalize(T24) + alpha * normalize((normalize(V12) + normalize(V18)) / 2))"


def tensor_checksum(x: torch.Tensor, n: int = 16) -> str:
    arr = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()[:n]


def canonical_fuse(t24: torch.Tensor, v12: torch.Tensor, v18: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    anchor = F.normalize(t24.float(), dim=-1)
    value = F.normalize((F.normalize(v12.float(), dim=-1) + F.normalize(v18.float(), dim=-1)) / 2.0, dim=-1)
    return F.normalize(anchor + float(alpha) * value, dim=-1)


def ivc_head_formula(t24: torch.Tensor, v12: torch.Tensor, v18: torch.Tensor, alpha: float = 0.5, a12: float = 1.0, a18: float = 1.0) -> torch.Tensor:
    residual = F.normalize(float(a12) * F.normalize(v12.float(), dim=-1) + float(a18) * F.normalize(v18.float(), dim=-1), dim=-1)
    return F.normalize(F.normalize(t24.float(), dim=-1) + float(alpha) * residual, dim=-1)


def compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor) -> Dict:
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    diff = (af - bf).abs()
    flat_a = af.reshape(-1, af.shape[-1]) if af.ndim > 1 else af.reshape(1, -1)
    flat_b = bf.reshape(-1, bf.shape[-1]) if bf.ndim > 1 else bf.reshape(1, -1)
    cos = F.cosine_similarity(flat_a, flat_b, dim=-1)
    unequal = (diff > 0).float().mean().item() if diff.numel() else 0.0
    return {
        "name": name,
        "shape_a": list(af.shape),
        "shape_b": list(bf.shape),
        "dtype_a": str(a.dtype),
        "dtype_b": str(b.dtype),
        "checksum_a": tensor_checksum(af),
        "checksum_b": tensor_checksum(bf),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
        "mean_cosine": float(cos.mean().item()) if cos.numel() else float("nan"),
        "min_cosine": float(cos.min().item()) if cos.numel() else float("nan"),
        "fraction_unequal": float(unequal),
    }

