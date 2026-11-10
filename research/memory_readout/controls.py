import torch
import torch.nn.functional as F

from .adaptive_prototype import weighted_topk_prototype
from .kernel_evidence import safe_topk


def static_centroid_logit(query, normal_keys, anomaly_keys, tau_cls=1.0):
    mu_n = F.normalize(normal_keys.mean(dim=0, keepdim=True), dim=-1)
    mu_a = F.normalize(anomaly_keys.mean(dim=0, keepdim=True), dim=-1)
    rn = (query * mu_n).sum(dim=-1)
    ra = (query * mu_a).sum(dim=-1)
    return (ra - rn) / float(tau_cls), rn, ra


def global_topk_then_split(query, memory, k, tau, tau_cls, fallback="centroid"):
    keys = memory["patch_keys"]
    labels = memory["patch_values"].argmax(dim=-1)
    sim = query @ keys.T
    vals, idx = safe_topk(sim, k)
    selected = keys[idx]
    selected_labels = labels[idx]
    outs = []
    for b in range(query.shape[0]):
        q = query[b : b + 1]
        protos = []
        scores = []
        for label in (0, 1):
            mask = selected_labels[b] == label
            if mask.any():
                weights = torch.softmax(vals[b][mask] / float(tau), dim=-1)
                proto = F.normalize((weights[:, None] * selected[b][mask]).sum(dim=0, keepdim=True), dim=-1)
            else:
                class_keys = keys[labels == label]
                proto = F.normalize(class_keys.mean(dim=0, keepdim=True), dim=-1)
            protos.append(proto)
            scores.append((q * proto).sum(dim=-1))
        outs.append((scores[1] - scores[0]) / float(tau_cls))
    return torch.cat(outs, dim=0)


def topk_label_voting(query, memory, k, tau):
    sim = query @ memory["patch_keys"].T
    vals, idx = safe_topk(sim, k)
    weights = torch.softmax(vals / float(tau), dim=-1)
    labels = memory["patch_values"].float()[idx]
    return (weights.unsqueeze(-1) * labels).sum(dim=-2)


def shared_adaptive_prototype(query, memory, k, tau):
    keys = memory["patch_keys"]
    labels = memory["patch_values"].argmax(dim=-1).float()
    sim = query @ keys.T
    vals, idx = safe_topk(sim, 2 * int(k))
    selected = keys[idx]
    weights = torch.softmax(vals / float(tau), dim=-1)
    proto = F.normalize((weights.unsqueeze(-1) * selected).sum(dim=-2), dim=-1)
    sim_proto = (query * proto).sum(dim=-1)
    label_prop = (weights * labels[idx]).sum(dim=-1)
    return torch.sigmoid(sim_proto) * label_prop
