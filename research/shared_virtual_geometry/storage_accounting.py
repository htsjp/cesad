def original_coeff_storage(regions, dim, rank=8):
    r, d, k = int(regions), int(dim), int(rank)
    scalars = d + k * d + r * k + 2 * d + 8 * r + 4 * r
    return {"feature_scalars": scalars, "storage_ratio": scalars / (r * 4 * d), "fp32_bytes": scalars * 4, "fp16_bytes": scalars * 2}


def shared_geometry_storage(regions, dim, rank=8, label_specific=True):
    r, d, k = int(regions), int(dim), int(rank)
    geom = 16 if label_specific else 8
    scalars = d + k * d + r * k + 2 * d + r + geom + 4 * r
    return {
        "feature_scalars": scalars,
        "storage_ratio": scalars / (r * 4 * d),
        "fp32_bytes": scalars * 4,
        "fp16_bytes": scalars * 2,
        "geometry_scalars": geom,
        "scale_scalars": r,
        "storage_ratio_vs_original_coeff": scalars / max(1, original_coeff_storage(r, d, k)["feature_scalars"]),
    }

