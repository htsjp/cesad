from typing import Dict

import torch

from .residual_fusion import mean_one_softmax


def random_static_like(weights_by_layer: Dict[int, torch.Tensor], seed: int) -> Dict[int, torch.Tensor]:
    out = {}
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 1009)
    for layer, w in weights_by_layer.items():
        logits = torch.randn(w.shape, generator=generator)
        out[layer] = mean_one_softmax(logits)
    return out


def permute_weights(weights_by_layer: Dict[int, torch.Tensor], seed: int) -> Dict[int, torch.Tensor]:
    out = {}
    generator = torch.Generator()
    generator.manual_seed(int(seed) + 2003)
    for layer, w in weights_by_layer.items():
        perm = torch.randperm(w.numel(), generator=generator)
        out[layer] = w.detach().cpu()[perm]
    return out


def topk_binary_weights(scores_by_layer: Dict[int, torch.Tensor], keep_fraction: float, largest: bool = True) -> Dict[int, torch.Tensor]:
    out = {}
    for layer, scores in scores_by_layer.items():
        scores = scores.float()
        k = max(1, int(round(scores.numel() * float(keep_fraction))))
        idx = torch.topk(scores, k=k, largest=largest).indices
        w = torch.zeros_like(scores)
        w[idx] = scores.numel() / float(k)
        out[layer] = w
    return out

