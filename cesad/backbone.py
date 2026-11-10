"""Image encoder adapter around the original frozen visual implementation."""

import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .config import research_root
from .lrp import crop_boxes


class FrozenEncoder:
    def __init__(self, base_weights, ancillary_checkpoint, device):
        sys.path.insert(0, str(research_root()))
        import AnomalyCLIP_lib.model_load as loader
        from experiments.phase_x0a_mandatory_gates import load_components
        from utils.transforms import get_transform

        original = loader._download

        def pinned(url, cache_dir=None):
            if url != loader._MODELS["ViT-L/14@336px"]:
                raise RuntimeError("Unexpected base model")
            return str(base_weights)

        loader._download = pinned
        try:
            self.model, *_ = load_components(str(ancillary_checkpoint), device)
        finally:
            loader._download = original
        self.model.eval()
        self.model.requires_grad_(False)
        self.device = device
        position = self.model.visual.positional_embedding
        with torch.inference_mode():
            if position.shape != (577, 1024):
                raise RuntimeError("Expected original24x24 positional grid")
            grid = position[1:].reshape(-1, 24, 24, 1024).permute(0, 3, 1, 2)
            grid = F.interpolate(grid, (37, 37), mode="bilinear", align_corners=False)
            position.data = torch.cat(
                [position[:1], grid.reshape(-1, 1024, 1369).transpose(1, 2)[0]], dim=0
            )
        if self.model.dtype != torch.float32:
            raise RuntimeError("Expected FP32 visual model")
        self.preprocess = get_transform(argparse.Namespace(image_size=518))[0]

    def extract(self, path):
        from phase_xa.official_task_forward import (
            visual_outputs,
            normalized_patch_feature,
        )

        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            views = [image] + [image.crop(box) for box in crop_boxes(width, height)]
            tensor = torch.stack([self.preprocess(view) for view in views])
        output = []
        with torch.inference_mode(), torch.autocast(
            device_type=str(self.device).split(":")[0], enabled=False
        ):
            for batch in (tensor[:1], tensor[1:]):
                feature = normalized_patch_feature(
                    visual_outputs(
                        self.model, batch.to(self.device), adapter=None, grad=False
                    )
                )
                output.append(feature.float().cpu().numpy())
        result = np.concatenate(output)
        if result.shape != (5, 1369, 1024):
            raise RuntimeError("Unexpected frozen feature shape")
        return result
