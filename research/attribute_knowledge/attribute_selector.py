import torch


def top_r_indices(profile: torch.Tensor, r: int):
    r = min(int(r), profile.shape[-1])
    return torch.topk(profile, k=r, dim=-1).indices


def group_conditioned_top_r(profile: torch.Tensor, group_ids: torch.Tensor, r: int):
    device = profile.device
    group_ids = group_ids.to(device)
    groups = torch.unique(group_ids, sorted=True)
    group_scores = torch.stack([(profile[..., group_ids == gid]).sum(dim=-1) for gid in groups], dim=-1)
    selected_group = groups[group_scores.argmax(dim=-1)]
    flat_profile = profile.reshape(-1, profile.shape[-1])
    flat_group = selected_group.reshape(-1)
    out = []
    for row, gid in zip(flat_profile, flat_group):
        candidates = torch.nonzero(group_ids == gid, as_tuple=False).reshape(-1)
        k = min(int(r), int(candidates.numel()))
        local = torch.topk(row[candidates], k=k, dim=-1).indices
        chosen = candidates[local]
        if k < int(r):
            pad = chosen[:1].repeat(int(r) - k)
            chosen = torch.cat([chosen, pad], dim=0)
        out.append(chosen)
    return torch.stack(out, dim=0).reshape(*profile.shape[:-1], int(r))

