import torch
import torch.nn.functional as F


def random_unit_like(x, seed):
    gen = torch.Generator(device=x.device).manual_seed(int(seed))
    r = torch.randn(x.shape, generator=gen, device=x.device, dtype=x.dtype)
    return F.normalize(r, dim=-1)


def shuffle_contexts(contexts, seed):
    gen = torch.Generator(device=contexts.device).manual_seed(int(seed))
    return contexts[torch.randperm(contexts.shape[0], generator=gen, device=contexts.device)]

