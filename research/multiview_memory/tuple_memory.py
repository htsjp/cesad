from typing import Dict

import torch


VIEW_KEYS = ("T24", "V12", "V18")


def to_device(memory: Dict, device: str) -> Dict:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in memory.items()}


def cpu_state(memory: Dict) -> Dict:
    return {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in memory.items()}


def memory_storage_mb(memory: Dict) -> float:
    total = 0
    for key, value in memory.items():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return float(total / (1024 * 1024))


def repeated_t24(memory: Dict) -> Dict:
    out = dict(memory)
    out["V12"] = memory["T24"]
    out["V18"] = memory["T24"]
    return out


def shuffled_memory_views(memory: Dict, seed: int) -> Dict:
    generator = torch.Generator(device=memory["T24"].device)
    generator.manual_seed(int(seed) + 17_071)
    out = dict(memory)
    for key, offset in (("V12", 0), ("V18", 1)):
        generator.manual_seed(int(seed) + 17_071 + offset)
        perm = torch.randperm(memory[key].shape[0], generator=generator, device=memory[key].device)
        out[key] = memory[key][perm]
    return out


def label_shuffle(memory: Dict, seed: int) -> Dict:
    generator = torch.Generator(device=memory["patch_values"].device)
    generator.manual_seed(int(seed) + 91_337)
    perm = torch.randperm(memory["patch_values"].shape[0], generator=generator, device=memory["patch_values"].device)
    out = dict(memory)
    out["patch_values"] = memory["patch_values"][perm]
    return out

