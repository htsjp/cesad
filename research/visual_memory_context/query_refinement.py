import torch.nn.functional as F


def direct_refine(q0, delta, alpha):
    return F.normalize(q0.float() + float(alpha) * delta.float(), dim=-1)

