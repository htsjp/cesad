from typing import Dict

import torch
import torch.nn.functional as F


def cosine(query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
    return torch.matmul(F.normalize(query.float(), dim=-1), F.normalize(memory.float(), dim=-1).T)


def view_similarities(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "T24": cosine(query_views["T24"], memory["T24"]),
        "V12": cosine(query_views["V12"], memory["V12"]),
        "V18": cosine(query_views["V18"], memory["V18"]),
    }


def mismatch_similarities(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "T24": cosine(query_views["T24"], memory["V12"]),
        "V12": cosine(query_views["V12"], memory["V18"]),
        "V18": cosine(query_views["V18"], memory["T24"]),
    }

def maxsim_scores(query_views: Dict[str, torch.Tensor], memory: Dict[str, torch.Tensor]) -> torch.Tensor:
    qkeys = ("T24", "V12", "V18")
    mkeys = ("T24", "V12", "V18")
    parts = []
    for qk in qkeys:
        sims = torch.stack([cosine(query_views[qk], memory[mk]) for mk in mkeys], dim=0)
        parts.append(sims.max(dim=0).values)
    return torch.stack(parts, dim=0).sum(dim=0) / float(len(parts))

