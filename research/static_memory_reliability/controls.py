import torch

from static_memory_reliability.prior_standardization import standardize_within_label


def same_label_prior_shuffle(prior, labels, seed):
    labels = torch.as_tensor(labels).long()
    out = prior.clone()
    gen = torch.Generator().manual_seed(int(seed))
    for lab in [0, 1]:
        idx = torch.nonzero(labels == lab, as_tuple=False).squeeze(1)
        if idx.numel() > 1:
            out[idx] = out[idx[torch.randperm(idx.numel(), generator=gen)]]
    return out


def random_matched_prior(prior, labels, seed):
    labels = torch.as_tensor(labels).long()
    gen = torch.Generator().manual_seed(int(seed))
    out = torch.zeros_like(prior)
    for lab in [0, 1]:
        idx = torch.nonzero(labels == lab, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        vals = prior[idx]
        rand = torch.randn(idx.numel(), generator=gen, dtype=prior.dtype)
        rand = (rand - rand.mean()) / rand.std(unbiased=False).clamp_min(1e-6)
        out[idx] = (rand * vals.std(unbiased=False).clamp_min(1e-6) + vals.mean()).clamp(-3, 3)
    out, _ = standardize_within_label(out, labels)
    return out


def cross_label_prior_shuffle(prior, seed):
    gen = torch.Generator().manual_seed(int(seed))
    return prior[torch.randperm(prior.numel(), generator=gen)]


def category_mean_prior(prior, categories):
    out = prior.clone()
    for cat in sorted(set(categories)):
        idx = torch.tensor([i for i, c in enumerate(categories) if c == cat], dtype=torch.long)
        if idx.numel():
            out[idx] = prior[idx].mean()
    return out


def label_frequency_prior(labels):
    labels = torch.as_tensor(labels).long()
    out = torch.zeros(labels.shape[0], dtype=torch.float32)
    n0 = float((labels == 0).sum().item())
    n1 = float((labels == 1).sum().item())
    total = max(1.0, n0 + n1)
    out[labels == 0] = n0 / total
    out[labels == 1] = n1 / total
    return out

