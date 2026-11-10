"""Fixed nonoverlapping2x2 local re-observation, matching the sealed runner."""

import torch
import torch.nn.functional as F


def crop_boxes(width, height):
    return [
        (0, 0, width // 2, height // 2),
        (width // 2, 0, width, height // 2),
        (0, height // 2, width // 2, height),
        (width // 2, height // 2, width, height),
    ]


def fuse_views(q):
    """[heads,global/TL/TR/BL/BR,37,37] -> [heads,74,74]."""
    if q.ndim != 4 or tuple(q.shape[1:]) != (5, 37, 37):
        raise ValueError("Expected5 ordered37x37 views")
    global_map = F.interpolate(
        q[:, 0:1], size=(74, 74), mode="bilinear", align_corners=True
    )[:, 0]
    local = torch.empty_like(global_map)
    local[:, :37, :37] = q[:, 1]
    local[:, :37, 37:] = q[:, 2]
    local[:, 37:, :37] = q[:, 3]
    local[:, 37:, 37:] = q[:, 4]
    return 0.5 * global_map + 0.5 * local
