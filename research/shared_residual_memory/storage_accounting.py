def global_shared_storage(groups: int, dim: int, rank: int, label_conditioned=False):
    full = groups * 4 * dim
    basis_count = 2 if label_conditioned else 1
    compressed = groups * dim + basis_count * rank * dim + groups * 4 * rank
    return {
        "full_feature_scalars": int(full),
        "compressed_feature_scalars": int(compressed),
        "feature_scalar_ratio": float(compressed / full),
        "fp32_feature_bytes": int(compressed * 4),
        "fp16_feature_bytes": int(compressed * 2),
        "full_fp32_feature_bytes": int(full * 4),
        "full_fp16_feature_bytes": int(full * 2),
        "basis_count": int(basis_count),
    }

