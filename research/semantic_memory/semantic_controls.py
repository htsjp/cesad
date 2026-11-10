import torch
import torch.nn.functional as F


def shuffled_semantics(k_semantic: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=k_semantic.device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(k_semantic.numel(), generator=generator, device=k_semantic.device)
    return k_semantic[perm]


def random_text_bank(reference_bank, seed: int):
    embeddings = reference_bank["embeddings"]
    generator = torch.Generator(device=embeddings.device)
    generator.manual_seed(int(seed))
    random = torch.randn(embeddings.shape, generator=generator, device=embeddings.device, dtype=embeddings.dtype)
    random = F.normalize(random, dim=-1)
    metadata = dict(reference_bank["metadata"])
    metadata["random_anchor_seed"] = int(seed)
    return {"embeddings": random, "labels": reference_bank["labels"].clone(), "metadata": metadata}


def label_only_semantics(patch_values: torch.Tensor) -> torch.Tensor:
    labels = patch_values.argmax(dim=-1)
    return torch.where(labels == 1, torch.ones_like(labels, dtype=torch.float32), -torch.ones_like(labels, dtype=torch.float32))
