import torch
import torch.nn.functional as F


def tensor_stats(name, official, candidate):
    a = official.detach().float().cpu()
    b = candidate.detach().float().cpu()
    diff = (a - b).abs()
    flat_a = a.reshape(-1)
    flat_b = b.reshape(-1)
    cosine = F.cosine_similarity(flat_a, flat_b, dim=0).item() if flat_a.numel() else 1.0
    return {
        "tensor": name,
        "shape": "x".join(str(v) for v in a.shape),
        "dtype_official": str(official.dtype),
        "dtype_candidate": str(candidate.dtype),
        "mean_abs_diff": float(diff.mean()) if diff.numel() else 0.0,
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "cosine": float(cosine),
        "nan_count": int(torch.isnan(b).sum().item()),
        "inf_count": int(torch.isinf(b).sum().item()),
    }


def compare_visual_outputs(official, candidate):
    rows = [tensor_stats("returned_cls_features", official[0], candidate[0])]
    for idx, (a, b) in enumerate(zip(official[1], candidate[1])):
        rows.append(tensor_stats(f"returned_patch_features_layer_{idx}", a, b))
    for idx, (a, b) in enumerate(zip(official[2], candidate[2])):
        rows.append(tensor_stats(f"returned_cls_tokens_layer_{idx}", a, b))
    for idx, (a, b) in enumerate(zip(official[3], candidate[3])):
        rows.append(tensor_stats(f"returned_patch_proj_layer_{idx}", a, b))
    return rows
