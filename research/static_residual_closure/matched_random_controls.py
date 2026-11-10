import torch

from .static_head_gate import mean_one_softmax


def random_static_weights(heads, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 43_211)
    return {12: mean_one_softmax(torch.randn(heads, generator=gen)), 18: mean_one_softmax(torch.randn(heads, generator=gen))}


def permuted_weights(weights, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 19_919)
    out = {}
    for layer, w in weights.items():
        out[layer] = w.detach().cpu()[torch.randperm(w.numel(), generator=gen)]
    return out


def random_subset_weights(heads, keep_fraction, seed):
    gen = torch.Generator()
    gen.manual_seed(int(seed) + 9_001)
    out = {}
    k = max(1, int(round(heads * float(keep_fraction))))
    for layer in (12, 18):
        idx = torch.randperm(heads, generator=gen)[:k]
        w = torch.zeros(heads)
        w[idx] = heads / float(k)
        out[layer] = w
    return out

