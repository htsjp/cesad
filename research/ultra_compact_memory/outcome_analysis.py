def assign_outcome(summary_rows, source_selection, sanity):
    if not sanity.get("both_transfer_directions_completed"):
        return "G_IMPLEMENTATION_OR_DATASET_FAILURE"
    if not sanity.get("shared_rank2_generalization_supported"):
        return "F_SHARED_RANK2_NOT_CROSS_DOMAIN_STABLE"
    five = source_selection.get("selected_5pct_by_direction", {})
    ten = source_selection.get("selected_10pct_by_direction", {})
    if len(five) == 2 and all(v for v in five.values()) and sanity.get("five_pct_primary_success"):
        return "A_CROSS_DOMAIN_95_PERCENT_COMPRESSION_SUPPORTED"
    if len(five) == 2 and all(v for v in five.values()) and sanity.get("five_pct_partial_success"):
        return "B_CROSS_DOMAIN_PARTIAL_95_PERCENT_COMPRESSION"
    if len(ten) == 2 and all(v for v in ten.values()) and sanity.get("ten_pct_compact_success"):
        return "C_CROSS_DOMAIN_90_PERCENT_COMPRESSION_SUPPORTED"
    if sanity.get("one_direction_compression_success"):
        return "D_DATASET_SPECIFIC_ULTRA_COMPRESSION"
    if sanity.get("shared_rank2_generalization_supported"):
        return "E_SHARED_RANK2_GENERALIZES_BUT_CENTROID_COMPRESSION_FAILS"
    return "H_INCONCLUSIVE"

