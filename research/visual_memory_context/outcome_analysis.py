def assign_outcome(rows, sanity):
    if not sanity.get("reference_code_ok", False):
        return "J_REFERENCE_CODE_FAILURE"
    if not sanity.get("source_protocol_verified", False):
        return "K_SOURCE_PROTOCOL_FAILURE"
    if not sanity.get("c7_reproduces_v2", False) or not sanity.get("c11_reproduces_v2", False):
        return "L_IMPLEMENTATION_FAILURE"
    directions = sorted({r["direction"] for r in rows})
    gains = []
    direct_matches = False
    random_matches = False
    for d in directions:
        ds = [r for r in rows if r["direction"] == d]
        v2 = next((r for r in ds if r["variant"] == "V2_CompactPCA"), None)
        v7 = next((r for r in ds if r["variant"].startswith("V7_")), None)
        v4 = next((r for r in ds if r["variant"].startswith("V4_")), None)
        if v2 and v7:
            gains.append(float(v7["P-AUROC"]) - float(v2["P-AUROC"]))
        if v4 and v7 and float(v4["P-AUROC"]) >= float(v7["P-AUROC"]) - 1e-6:
            direct_matches = True
        ctrls = [r for r in ds if r["variant"].startswith(("C1_", "C3_"))]
        if v7 and ctrls:
            med = sorted(float(r["P-AUROC"]) for r in ctrls)[len(ctrls) // 2]
            random_matches = med >= float(v7["P-AUROC"]) - 1e-6
    if gains and all(g >= 0.003 for g in gains) and not direct_matches and not random_matches:
        return "A_PATCH_MEMORY_VISUAL_CONTEXT_SUPPORTED"
    if direct_matches and any(g > 0.0015 for g in gains):
        return "C_DIRECT_VISUAL_RESIDUAL_ONLY"
    if random_matches:
        return "E_CONTEXT_IDENTITY_NOT_SPECIFIC"
    if any(g > 0.0015 for g in gains):
        return "H_DATASET_SPECIFIC_VISUAL_CONTEXT_GAIN"
    return "I_NO_VISUAL_CONTEXT_GAIN"

