import torch


def lazy_logits(query: torch.Tensor, grouped_keys: torch.Tensor, values: torch.Tensor, batch_groups: int = 256):
    chunks = []
    flat_q = query.reshape(-1, query.shape[-1])
    for start in range(0, grouped_keys.shape[0], int(batch_groups)):
        keys = grouped_keys[start : start + int(batch_groups)].reshape(-1, grouped_keys.shape[-1])
        chunks.append(flat_q @ keys.T)
    sim = torch.cat(chunks, dim=1).reshape(*query.shape[:-1], -1)
    weights = torch.softmax(sim, dim=-1)
    logits = weights @ values.float()
    return logits, sim, weights

