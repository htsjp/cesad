import torch


def same_label_shuffle_coeff(coeff, labels, seed):
    labels = torch.as_tensor(labels).long()
    out = coeff.clone()
    gen = torch.Generator().manual_seed(int(seed))
    for lab in [0, 1]:
        idx = torch.nonzero(labels == lab, as_tuple=False).squeeze(1)
        if idx.numel() > 1:
            out[idx] = out[idx[torch.randperm(idx.numel(), generator=gen)]]
    return out


def same_label_shuffle_scale(scales, labels, seed):
    labels = torch.as_tensor(labels).long()
    out = scales.clone()
    gen = torch.Generator().manual_seed(int(seed))
    for lab in [0, 1]:
        idx = torch.nonzero(labels == lab, as_tuple=False).squeeze(1)
        if idx.numel() > 1:
            out[idx] = out[idx[torch.randperm(idx.numel(), generator=gen)]]
    return out

