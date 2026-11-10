def assign_outcome(rows, sanity):
    if not sanity.get("consistency_resolved") or not sanity.get("grouping_ok") or not sanity.get("rank3_exact"):
        return "IMPLEMENTATION_OR_METADATA_FAILURE"
    by = {r["variant"]: r for r in rows}
    full = by.get("V0_FullFPS")
    rank1 = by.get("V3_BaseResidual_Rank1")
    rank2 = by.get("V4_BaseResidual_Rank2")
    if not full or not rank1 or not rank2:
        return "INCONCLUSIVE"
    close = []
    for r in (rank1, rank2):
        close.append(
            abs(float(r["P-AUROC"]) - float(full["P-AUROC"])) <= 0.002
            and abs(float(r["PRO"]) - float(full["PRO"])) <= 0.002
            and abs(float(r["I-AUROC"]) - float(full["I-AUROC"])) <= 0.002
        )
    if any(close):
        return "PARTIAL_RESIDUAL_COMPRESSION"
    return "INCONCLUSIVE"

