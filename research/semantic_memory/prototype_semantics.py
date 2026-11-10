import torch
import torch.nn.functional as F


def _top_mean(sim: torch.Tensor, m: int) -> torch.Tensor:
    k = min(int(m), sim.shape[-1])
    return torch.topk(sim, k=k, dim=-1).values.mean(dim=-1)


def semantic_scores(features: torch.Tensor, text_bank, m: int = 3) -> torch.Tensor:
    features = F.normalize(features.float(), dim=-1)
    anchors = F.normalize(text_bank["embeddings"].to(features.device).float(), dim=-1)
    labels = text_bank["labels"].to(features.device)
    sim = features @ anchors.T
    normal = _top_mean(sim[..., labels == 0], m)
    anomaly = _top_mean(sim[..., labels == 1], m)
    return anomaly - normal


@torch.no_grad()
def attach_prototype_semantics(memory: dict, text_bank, m: int = 3) -> dict:
    enriched = dict(memory)
    enriched["patch_semantic"] = semantic_scores(memory["patch_keys"], text_bank, m=m)
    return enriched
