def assign_outcome(rows, sanity):
    if not sanity.get("reference_code_ok", False):
        return "I_REFERENCE_CODE_FAILURE"
    if not sanity.get("source_protocol_verified", False):
        return "J_SOURCE_PROTOCOL_FAILURE"
    if not sanity.get("factorized_equivalence_ok", False):
        return "H_FACTORIZED_RETRIEVAL_FAILURE"
    directions = sorted({r["direction"] for r in rows})
    gains = []
    label_explains = True
    random_matches = False
    for d in directions:
        drows = [r for r in rows if r["direction"] == d]
        pca = next((r for r in drows if r["variant"].startswith("V2_CompactPCA")), None)
        v7 = next((r for r in drows if r["variant"].startswith("V7_LabelPlusLearnedTextBasis")), None)
        v3 = next((r for r in drows if r["variant"].startswith("V3_LabelAnchoredCompactBasis")), None)
        if pca and v7:
            gains.append(float(v7["P-AUROC"]) - float(pca["P-AUROC"]))
        if v3 and v7 and float(v7["P-AUROC"]) > float(v3["P-AUROC"]) + 0.0015:
            label_explains = False
        ctrls = [r for r in drows if r["variant"].startswith("C1_RandomTextDirection")]
        if v7 and ctrls:
            med = sorted(float(r["P-AUROC"]) for r in ctrls)[len(ctrls) // 2]
            random_matches = med >= float(v7["P-AUROC"]) - 1e-6
    if gains and all(g > 0.003 for g in gains) and not label_explains and not random_matches:
        return "A_TEXT_ANCHORED_COMPACT_MEMORY_SUPPORTED"
    if label_explains and any(g > 0.0015 for g in gains):
        return "C_LABEL_GUIDED_MEMORY_ONLY"
    if random_matches:
        return "E_TEXT_DIRECTION_NOT_SPECIFIC"
    if any(g > 0.0015 for g in gains):
        return "F_DATASET_SPECIFIC_TEXT_BASIS_GAIN"
    return "G_NO_TEXT_GUIDED_MEMORY_GAIN"

