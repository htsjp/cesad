def assign_outcome(rows, sanity):
    if not sanity.get("source_protocol_verified", False):
        return "K_SOURCE_PROTOCOL_FAILURE"
    if not sanity.get("factorized_equivalence_ok", False):
        return "J_FACTORIZED_RETRIEVAL_FAILURE"
    if not sanity.get("row_permutation_invariant", False) or not sanity.get("collapsed_matches_centroid_only", False):
        return "L_IMPLEMENTATION_FAILURE"
    gains = []
    random_matches = False
    directions = sorted({r["direction"] for r in rows})
    for d in directions:
        ds = [r for r in rows if r["direction"] == d]
        v2 = next((r for r in ds if r["variant"] == "V2_CompactPCA_OriginalCoefficients"), None)
        v8 = next((r for r in ds if r["variant"] == "V8_LearnedLabelSpecificGeometry"), None)
        if v2 and v8:
            gains.append(float(v8["P-AUROC"]) - float(v2["P-AUROC"]))
        ctrls = [r for r in ds if r["variant"].startswith("C1_RandomGeometry")]
        if v8 and ctrls:
            med = sorted(float(x["P-AUROC"]) for x in ctrls)[len(ctrls) // 2]
            random_matches = med >= float(v8["P-AUROC"]) - 1e-6
    if gains and all(g > 0.003 for g in gains) and not random_matches:
        return "A_DISCRIMINATIVE_SHARED_GEOMETRY_SUPPORTED"
    if random_matches:
        return "F_GEOMETRY_IDENTITY_NOT_SPECIFIC"
    if gains and all(abs(g) <= 0.002 for g in gains):
        return "E_SHARED_GEOMETRY_COMPRESSION_ONLY"
    if any(g > 0.0015 for g in gains):
        return "H_DATASET_SPECIFIC_GEOMETRY_GAIN"
    return "G_REGIONAL_COEFFICIENTS_REQUIRED"

