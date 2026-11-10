from typing import Dict

import torch


def materialized_memory(base_memory: Dict[str, torch.Tensor], patch_keys: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "image_keys": base_memory["image_keys"],
        "image_values": base_memory["image_values"],
        "patch_keys": patch_keys,
        "patch_values": base_memory["patch_values"],
    }

