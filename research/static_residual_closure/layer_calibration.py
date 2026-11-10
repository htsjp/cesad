def layer_ratio_variants(ratios, alphas):
    out = []
    for a12, a18 in ratios:
        for alpha in alphas:
            out.append({
                "name": f"B1_LayerRatio_a12{float(a12):.2f}_a18{float(a18):.2f}_alpha{float(alpha):.2f}",
                "alpha": float(alpha),
                "a12": float(a12),
                "a18": float(a18),
                "family": "layer_ratio",
            })
    return out

