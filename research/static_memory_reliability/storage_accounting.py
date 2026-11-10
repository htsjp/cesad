def add_reliability_storage(base_storage, regions, full_scalars=None):
    out = dict(base_storage)
    prior_scalars = int(regions)
    out["reliability_prior_scalars"] = prior_scalars
    out["adapter_parameters"] = 0
    out["feature_scalars_without_prior"] = int(out.get("feature_scalars", 0))
    out["feature_scalars"] = int(out.get("feature_scalars", 0)) + prior_scalars
    out["fp32_bytes"] = int(out["feature_scalars"]) * 4
    out["fp16_bytes"] = int(out["feature_scalars"]) * 2
    if full_scalars:
        out["storage_ratio"] = float(out["feature_scalars"]) / float(full_scalars)
    out["reliability_prior_bytes_fp32"] = prior_scalars * 4
    out["reliability_prior_bytes_fp16"] = prior_scalars * 2
    return out

