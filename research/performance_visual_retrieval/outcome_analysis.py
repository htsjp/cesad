def assign_viii_a_outcome(rows, sanity):
    if not sanity.get("factorized_equivalence_ok", False):
        return "A8_IMPLEMENTATION_FAILURE"
    if not sanity.get("orthogonal_reproduces_identity", False):
        return "A8_IMPLEMENTATION_FAILURE"
    target = [r for r in rows if r.get("split") == "target"]
    gains = []
    random_matches = False
    for d in sorted({r["direction"] for r in target}):
        ds = [r for r in target if r["direction"] == d]
        base = next((r for r in ds if r["variant"] == "A0_IdentityMetric"), None)
        sym = next((r for r in ds if r["variant"] == "A_SelectedSymmetricMetric"), None)
        if base and sym:
            gains.append((float(sym["I-AUROC"]) - float(base["I-AUROC"]), float(sym["P-AUROC"]) - float(base["P-AUROC"]), float(sym["PRO"]) - float(base["PRO"])))
        controls = [r for r in ds if r["variant"].startswith("A8_RandomFrozenPSD")]
        if sym and controls:
            med = sorted(float(r["P-AUROC"]) for r in controls)[len(controls) // 2]
            random_matches = random_matches or med >= float(sym["P-AUROC"]) - 1e-6
    if random_matches:
        return "A5_METRIC_IDENTITY_NOT_SPECIFIC"
    if gains and all(g[0] > 0.005 or g[1] > 0.005 for g in gains):
        return "A1_SYMMETRIC_METRIC_SUPPORTED"
    if gains and all(g[0] > -0.002 and g[1] > -0.002 and g[2] > -0.002 for g in gains) and any(g[0] > 0.002 or g[1] > 0.002 for g in gains):
        return "A2_PARTIAL_SYMMETRIC_METRIC_GAIN"
    if gains and sum(1 for g in gains if g[0] > 0 or g[1] > 0) == 1:
        return "A6_DATASET_SPECIFIC_METRIC_GAIN"
    return "A7_NO_METRIC_GAIN"

