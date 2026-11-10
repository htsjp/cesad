def assign_outcome(summary_rows):
    by_name = {r["variant"]: r for r in summary_rows}
    base = by_name.get("V0_UniformResidualBaseline")
    if not base:
        return {"Outcome": "INCONCLUSIVE", "HeadIdentitySupported": False, "ScaleOrLayerCalibrationOnly": False, "NoRobustStaticImprovement": False}
    metric = "target_P-AUROC" if "target_P-AUROC" in base else "P-AUROC"
    random_best = max([r for r in summary_rows if r["variant"].startswith("E") or r["variant"].startswith("D")], key=lambda r: r.get(metric, float("-inf")), default=None)
    learned = by_name.get("C1_LearnedStaticHeadGate_proxy")
    if learned and random_best and random_best.get(metric, 0) >= learned.get(metric, 0):
        return {"Outcome": "NoRobustStaticImprovement", "HeadIdentitySupported": False, "ScaleOrLayerCalibrationOnly": False, "NoRobustStaticImprovement": True}
    return {"Outcome": "INCONCLUSIVE", "HeadIdentitySupported": False, "ScaleOrLayerCalibrationOnly": False, "NoRobustStaticImprovement": False}
