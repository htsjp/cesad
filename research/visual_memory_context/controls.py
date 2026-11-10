import torch
import torch.nn.functional as F


def apply_context_control(context, mode, seed=0):
    if mode == "cross_shuffle":
        gen = torch.Generator(device=context.device).manual_seed(int(seed))
        return context[torch.randperm(context.shape[0], generator=gen, device=context.device)]
    if mode == "random":
        gen = torch.Generator(device=context.device).manual_seed(int(seed))
        rand = torch.randn(context.shape, generator=gen, device=context.device, dtype=context.dtype)
        scale = context.norm(dim=-1, keepdim=True).mean().clamp_min(1e-12)
        return F.normalize(rand, dim=-1) * scale
    if mode == "swap":
        out = context.clone()
        out[:, 0, :], out[:, 1, :] = context[:, 1, :], context[:, 0, :]
        out[:, 2, :] = F.normalize(out[:, 1, :] - out[:, 0, :], dim=-1)
        return out
    return context

