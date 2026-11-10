import torch


def regional_rms_scale(coeff):
    return coeff.float().pow(2).sum(dim=-1).mean(dim=1).sqrt()


def scale_summary(scales, labels):
    labels = torch.as_tensor(labels).long()
    out = {
        "scale_mean": float(scales.mean()),
        "scale_std": float(scales.std(unbiased=False)),
        "scale_min": float(scales.min()),
        "scale_max": float(scales.max()),
    }
    for lab, name in [(0, "normal"), (1, "anomaly")]:
        vals = scales[labels == lab]
        if vals.numel():
            out[f"{name}_scale_median"] = float(vals.median())
            out[f"{name}_scale_mean"] = float(vals.mean())
    return out


def median_scale(scales, labels, mode):
    if mode == "global":
        return torch.full_like(scales, float(scales.median()))
    out = scales.clone()
    labels = torch.as_tensor(labels).long()
    for lab in [0, 1]:
        vals = scales[labels == lab]
        if vals.numel():
            out[labels == lab] = vals.median()
    return out

