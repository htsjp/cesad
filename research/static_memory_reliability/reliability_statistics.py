def reliability_rows(stats, prior=None, direction="", seed=0, backend=""):
    rows = []
    labels = stats["labels"]
    prior = prior if prior is not None else stats["D"] * 0
    for i in range(labels.shape[0]):
        rows.append(
            {
                "seed": seed,
                "direction": direction,
                "backend": backend,
                "region": i,
                "label": int(labels[i]),
                "category": stats["categories"][i],
                "A": float(stats["A"][i]),
                "B": float(stats["B"][i]),
                "U": float(stats["U"][i]),
                "D": float(stats["D"][i]),
                "H": float(stats["H"][i]),
                "sigma": float(stats["sigma"][i]),
                "T": float(stats["T"][i]),
                "coverage": float(stats["coverage"][i]),
                "density": float(stats["density"][i]),
                "scale": float(stats["scale"][i]),
                "prior": float(prior[i]),
                "support_a": float(stats["support_a"][i]),
                "support_b": float(stats["support_b"][i]),
                "imputed": bool(stats["missing"][i]),
                "bootstrap_low": float(stats["bootstrap_low"][i]),
                "bootstrap_high": float(stats["bootstrap_high"][i]),
            }
        )
    return rows

