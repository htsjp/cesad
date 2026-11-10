import torch


def mean_one_softmax(logits):
    return torch.softmax(logits.float(), dim=-1) * logits.numel()


def uniform_weights(heads):
    return {12: torch.ones(heads), 18: torch.ones(heads)}


def low_variance_weights(head_rows, keep_fraction=0.25, largest=False):
    out = {}
    for layer in (12, 18):
        rows = [r for r in head_rows if int(r["layer"]) == layer]
        scores = torch.tensor([float(r["spatial_variance"]) for r in rows])
        k = max(1, int(round(len(rows) * float(keep_fraction))))
        idx = torch.topk(scores, k=k, largest=largest).indices
        w = torch.zeros(len(rows))
        w[idx] = len(rows) / float(k)
        out[layer] = w
    return out


def redundancy_subset_weights(head_rows, keep_fraction=0.25):
    out = {}
    for layer in (12, 18):
        rows = [r for r in head_rows if int(r["layer"]) == layer]
        scores = -torch.tensor([float(r["mean_head_cka_redundancy"]) for r in rows])
        k = max(1, int(round(len(rows) * float(keep_fraction))))
        idx = torch.topk(scores, k=k, largest=True).indices
        w = torch.zeros(len(rows))
        w[idx] = len(rows) / float(k)
        out[layer] = w
    return out

