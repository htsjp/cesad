import torch
import torch.nn.functional as F


def shuffled_profiles(profiles: torch.Tensor, seed: int):
    generator = torch.Generator(device=profiles.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(profiles.shape[0], generator=generator, device=profiles.device)
    return profiles[perm]


def shuffled_attribute_dimensions(profiles: torch.Tensor, seed: int):
    generator = torch.Generator(device=profiles.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(profiles.shape[-1], generator=generator, device=profiles.device)
    return profiles[..., perm]


def random_text_bank_like(bank, seed: int):
    embeddings = bank["embeddings"]
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(seed))
    random = torch.randn(embeddings.shape, generator=generator, device=embeddings.device, dtype=embeddings.dtype)
    random = F.normalize(random, dim=-1)
    metadata = dict(bank["metadata"])
    metadata["random_seed"] = int(seed)
    return {"embeddings": random, "group_ids": bank["group_ids"].clone(), "labels": bank["labels"].clone(), "metadata": metadata}


def label_only_memory_profiles(values: torch.Tensor):
    labels = values.argmax(dim=-1)
    out = torch.zeros(labels.shape[0], 2, device=values.device, dtype=torch.float32)
    out[torch.arange(labels.shape[0], device=values.device), labels] = 1.0
    return out


def label_only_query_profiles(scalar_score: torch.Tensor):
    p_anom = scalar_score.reshape(-1).clamp(0.0, 1.0)
    return torch.stack([1.0 - p_anom, p_anom], dim=-1)

