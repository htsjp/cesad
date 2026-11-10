def bootstrap_rows(stats, direction="", seed=0, backend=""):
    rows = []
    for i in range(stats["labels"].shape[0]):
        rows.append(
            {
                "seed": seed,
                "direction": direction,
                "backend": backend,
                "region": i,
                "sigma": float(stats["sigma"][i]),
                "bootstrap_low": float(stats["bootstrap_low"][i]),
                "bootstrap_high": float(stats["bootstrap_high"][i]),
                "coefficient_of_variation": float(stats["sigma"][i] / (abs(float(stats["D"][i])) + 1e-6)),
            }
        )
    return rows

