from .static_head_gate import low_variance_weights, redundancy_subset_weights


def subset_variants(head_rows, fractions):
    variants = []
    for frac in fractions:
        variants.append({"name": f"D1_LowVariance_keep{int(frac*100)}", "family": "subset", "weights": low_variance_weights(head_rows, frac, largest=False), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
        variants.append({"name": f"D2_HighVariance_keep{int(frac*100)}", "family": "subset", "weights": low_variance_weights(head_rows, frac, largest=True), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
        variants.append({"name": f"D3_LowRedundancy_keep{int(frac*100)}", "family": "subset", "weights": redundancy_subset_weights(head_rows, frac), "alpha": 0.5, "a12": 1.0, "a18": 1.0})
    return variants

