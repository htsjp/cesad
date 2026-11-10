import math

import torch
import torch.nn.functional as F


def grid_side(tokens):
    side = int(math.sqrt(tokens.shape[1]))
    if side * side != tokens.shape[1]:
        raise ValueError(f"Token count {tokens.shape[1]} is not a square grid")
    return side


def neighborhood_mean(tokens, kernel=3, mode="reflect"):
    b, n, c = tokens.shape
    side = grid_side(tokens)
    x = tokens.transpose(1, 2).reshape(b, c, side, side)
    pad = kernel // 2
    if mode == "reflect":
        xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        pooled = F.avg_pool2d(xp, kernel_size=kernel, stride=1)
    else:
        weight = torch.ones((1, 1, kernel, kernel), device=x.device, dtype=x.dtype)
        xp = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0)
        pooled_sum = F.conv2d(xp, weight.expand(c, 1, kernel, kernel), groups=c)
        valid = F.pad(torch.ones((b, 1, side, side), device=x.device, dtype=x.dtype), (pad, pad, pad, pad), mode="constant", value=0)
        count = F.conv2d(valid, weight).clamp_min(1.0)
        pooled = pooled_sum / count
    return pooled.reshape(b, c, n).transpose(1, 2)


def aggregate_feature(tokens, eta3=0.5, eta5=0.0, boundary="reflect"):
    out = tokens.float()
    if eta3:
        out = out + float(eta3) * neighborhood_mean(tokens, 3, boundary)
    if eta5:
        out = out + float(eta5) * neighborhood_mean(tokens, 5, boundary)
    return F.normalize(out, dim=-1)


def anomaly_map_sharpness(amap):
    gx = amap[..., :, 1:] - amap[..., :, :-1]
    gy = amap[..., 1:, :] - amap[..., :-1, :]
    return float((gx.abs().mean() + gy.abs().mean()).item())

