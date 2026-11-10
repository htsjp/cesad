from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


class PrototypeReconstructor:
    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def reconstruct(self, query_features: torch.Tensor, selected_features: torch.Tensor, selected_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        query = F.normalize(query_features.float(), dim=-1)
        features = F.normalize(selected_features.float(), dim=-1)
        values = selected_values.float()
        labels = values.argmax(dim=-1)
        sim = query @ features.T
        weights = torch.softmax(sim / self.temperature, dim=-1)

        protos = []
        available = []
        for label in [0, 1]:
            mask = labels == label
            available.append(bool(mask.any()))
            if mask.any():
                w = weights[:, mask]
                w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                proto = F.normalize(w @ features[mask], dim=-1)
            else:
                proto = torch.zeros_like(query)
            protos.append(proto)

        normal_sim = (query * protos[0]).sum(dim=-1)
        anomaly_sim = (query * protos[1]).sum(dim=-1)
        pair = torch.stack([normal_sim, anomaly_sim], dim=-1)
        probs = torch.softmax(pair / self.temperature, dim=-1)
        return {"logits": probs, "prototypes": torch.stack(protos, dim=1), "available": available}
