import torch


def permute_memory_values(cache, seed=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(cache["values"].shape[0], generator=generator)
    return {
        "keys": cache["keys"].clone(),
        "values": cache["values"][perm].clone(),
        "provenance": list(cache["provenance"]),
        "permutation": perm.tolist(),
    }
