def assign_outcome(rows, sanity):
    if not sanity.get("baseline_ok") or not sanity.get("grouping_ok") or not sanity.get("lazy_materialized_match"):
        return "IMPLEMENTATION_OR_METADATA_FAILURE"
    by = {r["variant"]: r for r in rows}
    full = by.get("V0_FullFPS")
    centroid = by.get("V1_FPSCentroidRepeated4")
    g2 = by.get("V3_GlobalSharedPCA_Rank2")
    if not full or not centroid or not g2:
        return "INCONCLUSIVE"
    close = (
        float(full["I-AUROC"]) - float(g2["I-AUROC"]) <= 0.003
        and float(full["P-AUROC"]) - float(g2["P-AUROC"]) <= 0.003
        and float(full["PRO"]) - float(g2["PRO"]) <= 0.003
    )
    storage_ok = float(g2.get("storage_ratio", 1.0)) <= 0.40
    better_centroid = float(g2["PRO"]) > float(centroid["PRO"]) and float(g2["P-AUROC"]) > float(centroid["P-AUROC"])
    if close and storage_ok and better_centroid:
        return "PARTIAL_SHARED_BASIS_COMPRESSION"
    return "INCONCLUSIVE"

