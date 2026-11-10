import torch


def same_sample_mask(provenance, sample_id, device):
    return torch.tensor([p["sample_id"] == int(sample_id) for p in provenance], dtype=torch.bool, device=device)


def apply_self_exclusion(logits, provenance, sample_id):
    mask = same_sample_mask(provenance, sample_id, logits.device)
    excluded = int(mask.sum().item())
    if excluded:
        logits = logits.masked_fill(mask.unsqueeze(0), float("-inf"))
    return logits, excluded
