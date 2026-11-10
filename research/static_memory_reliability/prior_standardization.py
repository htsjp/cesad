import torch


def standardize_within_label(raw, labels, clip=3.0, eps=1e-6):
    raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
    labels = torch.as_tensor(labels).long()
    out = torch.zeros_like(raw)
    stats = {}
    for lab in [0, 1]:
        idx = labels == lab
        vals = raw[idx]
        if vals.numel() == 0:
            continue
        mean = vals.mean()
        std = vals.std(unbiased=False).clamp_min(eps)
        out[idx] = ((vals - mean) / std).clamp(-float(clip), float(clip))
        stats[f"label_{lab}_mean"] = float(out[idx].mean())
        stats[f"label_{lab}_std"] = float(out[idx].std(unbiased=False))
    return out, stats

