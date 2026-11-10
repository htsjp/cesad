import torch
import torch.nn.functional as F


def mean_one_softmax(logits: torch.Tensor) -> torch.Tensor:
    weights = torch.softmax(logits.float(), dim=-1)
    return weights * weights.shape[-1]


def fuse_t24_value(t24: torch.Tensor, v12: torch.Tensor, v18: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    residual = F.normalize(F.normalize(v12.float(), dim=-1) + F.normalize(v18.float(), dim=-1), dim=-1)
    return F.normalize(F.normalize(t24.float(), dim=-1) + float(alpha) * residual, dim=-1)


def value_only(v12: torch.Tensor, v18: torch.Tensor) -> torch.Tensor:
    return F.normalize(F.normalize(v12.float(), dim=-1) + F.normalize(v18.float(), dim=-1), dim=-1)

