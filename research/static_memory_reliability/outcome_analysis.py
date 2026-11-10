def assign_outcome(rows, selection, sanity):
    if not sanity.get("retrieval_path_verified", False):
        return "I_RETRIEVAL_PATH_FAILURE"
    if not sanity.get("source_protocol_verified", False):
        return "J_SOURCE_PROTOCOL_FAILURE"
    if not sanity.get("p0_reproduces_v2", False) or not sanity.get("c9_reproduces_v2", False):
        return "K_IMPLEMENTATION_FAILURE"
    if not selection.get("selected_is_non_uniform", False):
        return "H_NO_STATIC_RELIABILITY_GAIN"
    target_rows = [r for r in rows if r.get("split") == "target"]
    gains = []
    random_or_shuffle_matches = False
    for d in sorted({r["direction"] for r in target_rows}):
        ds = [r for r in target_rows if r["direction"] == d]
        v2 = next((r for r in ds if r["variant"] == "V2_CompactPCA_Uniform"), None)
        sel = next((r for r in ds if r["variant"] == "V5_CompactPCA_CompositeReliability"), None)
        if not v2 or not sel:
            continue
        gains.append((float(sel["I-AUROC"]) - float(v2["I-AUROC"]), float(sel["P-AUROC"]) - float(v2["P-AUROC"]), float(sel["PRO"]) - float(v2["PRO"])))
        controls = [r for r in ds if r["variant"].startswith("C1_") or r["variant"].startswith("C2_")]
        if controls:
            med = sorted(float(r["P-AUROC"]) for r in controls)[len(controls) // 2]
            random_or_shuffle_matches = random_or_shuffle_matches or med >= float(sel["P-AUROC"]) - 1e-6
    if random_or_shuffle_matches:
        return "F_PRIOR_IDENTITY_NOT_SPECIFIC"
    if gains and all(g[0] > 0.003 or g[1] > 0.003 for g in gains):
        return "A_STATIC_MEMORY_RELIABILITY_SUPPORTED"
    if gains and all(g[0] > -0.002 and g[1] > -0.002 for g in gains) and any(g[0] > 0.0015 or g[1] > 0.0015 or g[2] > 0.003 for g in gains):
        return "B_PARTIAL_STATIC_RELIABILITY_GAIN"
    if gains and sum(1 for g in gains if g[0] > 0 or g[1] > 0) == 1:
        return "G_DATASET_SPECIFIC_RELIABILITY_GAIN"
    return "H_NO_STATIC_RELIABILITY_GAIN"

