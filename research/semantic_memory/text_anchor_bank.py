import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from experiments.mrad_audit_utils import encode_plain_text, tokenize_texts


def expand_templates(attributes: Iterable[str], templates: Iterable[str]) -> List[List[str]]:
    groups = []
    for attribute in attributes:
        groups.append([template.format(attribute) for template in templates])
    return groups


@torch.no_grad()
def build_text_anchor_bank(model, cfg: Dict, device: str):
    normal = list(cfg["text_bank"]["normal"])
    anomaly = list(cfg["text_bank"]["anomaly"])
    templates = list(cfg["text_bank"]["templates"])
    attributes = normal + anomaly
    groups = expand_templates(attributes, templates)
    flat_texts = [text for group in groups for text in group]
    tokens = tokenize_texts(flat_texts).to(device)
    flat = encode_plain_text(model, tokens).float()
    flat = F.normalize(flat, dim=-1)
    per_attribute = flat.reshape(len(attributes), len(templates), -1).mean(dim=1)
    per_attribute = F.normalize(per_attribute, dim=-1)
    labels = torch.tensor([0] * len(normal) + [1] * len(anomaly), device=device, dtype=torch.long)
    metadata = {
        "normal_attributes": normal,
        "anomaly_attributes": anomaly,
        "templates": templates,
        "embedding_dim": int(per_attribute.shape[-1]),
        "normal_count": len(normal),
        "anomaly_count": len(anomaly),
        "all_attributes": attributes,
    }
    return {"embeddings": per_attribute, "labels": labels, "metadata": metadata}


def save_text_anchor_bank(bank, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": bank["embeddings"].detach().cpu(),
            "labels": bank["labels"].detach().cpu(),
        },
        output_dir / "text_anchor_embeddings.pt",
    )
    (output_dir / "text_anchor_metadata.json").write_text(
        json.dumps(bank["metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
