import torch


def materialized_similarity(query, keys):
    return torch.matmul(query, keys.T)


def check_materialized_match(query, keys_a, keys_b, atol=1e-6):
    sim_a = materialized_similarity(query, keys_a)
    sim_b = materialized_similarity(query, keys_b)
    diff = float((sim_a - sim_b).abs().max().item())
    return {"max_abs_diff": diff, "match": diff <= atol}

