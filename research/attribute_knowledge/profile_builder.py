import torch
import torch.nn.functional as F


def attribute_profiles(features: torch.Tensor, bank, tau_attr: float):
    logits = features.float() @ bank["embeddings"].to(features.device).float().T
    return F.softmax(logits / float(tau_attr), dim=-1)


def binary_attribute_profiles(features: torch.Tensor, bank, tau_attr: float):
    labels = bank["labels"].to(features.device)
    embeddings = bank["embeddings"].to(features.device)
    normal = F.normalize(embeddings[labels == 0].mean(dim=0, keepdim=True), dim=-1)
    anomaly = F.normalize(embeddings[labels == 1].mean(dim=0, keepdim=True), dim=-1)
    binary = torch.cat([normal, anomaly], dim=0)
    logits = features.float() @ binary.float().T
    return F.softmax(logits / float(tau_attr), dim=-1)

