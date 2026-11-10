def alpha_variants(values):
    return [{"name": f"A1_AlphaOnly_alpha{float(a):.2f}", "alpha": float(a), "a12": 1.0, "a18": 1.0, "family": "alpha"} for a in values]

