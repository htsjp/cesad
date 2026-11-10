import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F

from experiments.mrad_audit_utils import encode_plain_text, tokenize_texts


def _template_groups(attributes: Iterable[str], templates: Iterable[str]) -> List[List[str]]:
    return [[template.format(attribute) for template in templates] for attribute in attributes]


def _cosine_stats(embeddings: torch.Tensor, group_ids: torch.Tensor):
    sim = embeddings @ embeddings.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    same = group_ids[:, None] == group_ids[None, :]
    within = sim[same & ~eye]
    between = sim[~same]
    near = []
    attrs = embeddings.shape[0]
    for i in range(attrs):
        for j in range(i + 1, attrs):
            value = float(sim[i, j].item())
            if value >= 0.98:
                near.append({"i": i, "j": j, "cosine": value})
    return {
        "within_group_cosine_mean": float(within.mean().item()) if within.numel() else 0.0,
        "between_group_cosine_mean": float(between.mean().item()) if between.numel() else 0.0,
        "near_duplicate_pairs": near,
    }


@torch.no_grad()
def build_attribute_bank(model, cfg: Dict, device: str):
    groups_cfg = cfg["attribute_bank"]["groups"]
    templates = list(cfg["attribute_bank"]["templates"])
    attributes, group_names, group_ids, binary_labels = [], [], [], []
    for gid, (group_name, group_attrs) in enumerate(groups_cfg.items()):
        for attr in group_attrs:
            attributes.append(attr)
            group_names.append(group_name)
            group_ids.append(gid)
            binary_labels.append(0 if gid == 0 else 1)

    prompts = [text for group in _template_groups(attributes, templates) for text in group]
    tokens = tokenize_texts(prompts).to(device)
    flat = F.normalize(encode_plain_text(model, tokens).float(), dim=-1)
    embeddings = flat.reshape(len(attributes), len(templates), -1).mean(dim=1)
    embeddings = F.normalize(embeddings, dim=-1)

    group_ids_t = torch.tensor(group_ids, device=device, dtype=torch.long)
    labels_t = torch.tensor(binary_labels, device=device, dtype=torch.long)
    stats = _cosine_stats(embeddings, group_ids_t)
    metadata = {
        "attributes": attributes,
        "groups": group_names,
        "group_names": list(groups_cfg.keys()),
        "group_ids": group_ids,
        "binary_labels": binary_labels,
        "templates": templates,
        "num_attributes": len(attributes),
        "embedding_dim": int(embeddings.shape[-1]),
        **stats,
    }
    return {"embeddings": embeddings, "group_ids": group_ids_t, "labels": labels_t, "metadata": metadata}


def save_attribute_bank(bank, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": bank["embeddings"].detach().cpu(),
            "group_ids": bank["group_ids"].detach().cpu(),
            "labels": bank["labels"].detach().cpu(),
        },
        output_dir / "text_attribute_bank.pt",
    )
    (output_dir / "text_attribute_metadata.json").write_text(
        json.dumps(bank["metadata"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

