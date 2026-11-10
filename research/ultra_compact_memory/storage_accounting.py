import json
from pathlib import Path


def _bytes(scalars, bytes_per):
    return int(scalars) * int(bytes_per)


def fullfps_storage(regions, dim):
    scalars = int(regions) * 4 * int(dim)
    return {"feature_scalars": scalars, "storage_ratio": 1.0, "fp32_bytes": _bytes(scalars, 4), "fp16_bytes": _bytes(scalars, 2)}


def shared_rank2_storage(regions, dim):
    r, d = int(regions), int(dim)
    scalars = r * d + 2 * d + 8 * r
    return {"feature_scalars": scalars, "storage_ratio": scalars / (r * 4 * d), "fp32_bytes": _bytes(scalars, 4), "fp16_bytes": _bytes(scalars, 2)}


def centroid_pca_storage(regions, dim, rank):
    r, d, s = int(regions), int(dim), int(rank)
    scalars = d + s * d + r * s + 2 * d + 8 * r
    return {
        "feature_scalars": scalars,
        "storage_ratio": scalars / (r * 4 * d),
        "basis_scalars": s * d + 2 * d,
        "coefficient_scalars": r * s + 8 * r,
        "fp32_bytes": _bytes(scalars, 4),
        "fp16_bytes": _bytes(scalars, 2),
    }


def codebook_storage(regions, dim, k, offset_rank=0):
    r, d, k, q = int(regions), int(dim), int(k), int(offset_rank)
    scalars = k * d + 2 * d + 8 * r
    if q:
        scalars += q * d + q * r
    index_bytes = r * 2 if k <= 65536 else r * 4
    fp32 = _bytes(scalars, 4) + index_bytes
    return {
        "feature_scalars": scalars,
        "index_bytes": index_bytes,
        "storage_ratio": fp32 / max(1, r * 4 * d * 4),
        "basis_scalars": 2 * d + q * d,
        "codebook_scalars": k * d,
        "coefficient_scalars": 8 * r + q * r,
        "fp32_bytes": fp32,
        "fp16_bytes": _bytes(scalars, 2) + index_bytes,
    }


def serialized_bytes(path):
    path = Path(path)
    return path.stat().st_size if path.exists() else 0


def write_json_artifact(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return serialized_bytes(path)

