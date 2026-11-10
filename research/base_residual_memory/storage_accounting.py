def feature_storage_scalars(dim: int, rank: int, groups: int, virtual_per_group: int = 4):
    full = groups * virtual_per_group * dim
    local = groups * (dim + rank * dim + virtual_per_group * rank)
    return {
        "full_feature_scalars": int(full),
        "compressed_feature_scalars": int(local),
        "feature_scalar_ratio": float(local / full) if full else float("nan"),
        "fp32_feature_bytes": int(local * 4),
        "fp16_feature_bytes": int(local * 2),
        "full_fp32_feature_bytes": int(full * 4),
        "full_fp16_feature_bytes": int(full * 2),
    }

