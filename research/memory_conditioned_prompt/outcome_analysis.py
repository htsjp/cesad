def assign_outcome(rows, sanity):
    if not sanity.get("reference_code_ok") or not sanity.get("source_protocol_verified"):
        return "J_IMPLEMENTATION_OR_PROTOCOL_FAILURE"
    if not sanity.get("clip_frozen_ok") or not sanity.get("c11_reproduces_v0"):
        return "J_IMPLEMENTATION_OR_PROTOCOL_FAILURE"
    dirs = sorted({r["direction"] for r in rows})
    gains = []
    for d in dirs:
        dr = [r for r in rows if r["direction"] == d]
        v0 = next((r for r in dr if r["variant"] == "V0_CompactMemoryBaseline"), None)
        v6 = next((r for r in dr if r["variant"] == "V6_MemoryConditionedPromptQueryRefinement"), None)
        if v0 and v6:
            gains.append((float(v6["target_I-AUROC"]) - float(v0["target_I-AUROC"]), float(v6["target_P-AUROC"]) - float(v0["target_P-AUROC"]), float(v6["target_PRO"]) - float(v0["target_PRO"])))
    if not gains:
        return "K_INCONCLUSIVE"
    mean_i = sum(g[0] for g in gains) / len(gains)
    mean_p = sum(g[1] for g in gains) / len(gains)
    mean_pro = sum(g[2] for g in gains) / len(gains)
    if (mean_i >= 0.005 or mean_p >= 0.005) and all(g[2] >= -0.002 for g in gains):
        return "A_MEMORY_CONDITIONED_MULTIMODAL_PROMPT_SUPPORTED"
    if (mean_i >= 0.002 or mean_p >= 0.002 or mean_pro >= 0.003) and all(g[0] > -0.01 and g[1] > -0.01 for g in gains):
        return "B_PARTIAL_MEMORY_PROMPT_GAIN"
    if any((g[0] > 0 or g[1] > 0 or g[2] > 0) for g in gains):
        return "G_DATASET_SPECIFIC_PROMPT_GAIN"
    return "H_NO_MEMORY_PROMPT_GAIN"

