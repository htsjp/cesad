def guided_storage(regions, dim, rank, learned_prompt_scalars=0):
    r, d, k = int(regions), int(dim), int(rank)
    scalars = d + k * d + r * k + 2 * d + 8 * r + 4 * r + int(learned_prompt_scalars) + 2 * d
    full = r * 4 * d
    return {
        "feature_scalars": scalars,
        "storage_ratio": scalars / full,
        "fp32_bytes": scalars * 4,
        "fp16_bytes": scalars * 2,
        "basis_scalars": k * d + 2 * d,
        "coefficient_scalars": r * k + 8 * r,
        "prompt_scalars": int(learned_prompt_scalars) + 2 * d,
        "norm_scalars": 4 * r,
        "metadata_bytes": r * 16,
    }

