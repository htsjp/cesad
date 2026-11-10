import torch


def mild_style_view(image):
    out = image * 1.06 + 0.015
    out = torch.clamp(out, 0.0, 1.0)
    return out
