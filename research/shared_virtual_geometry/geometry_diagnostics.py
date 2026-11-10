import itertools
import torch


def polygon_area(g):
    x, y = g[:, 0], g[:, 1]
    return 0.5 * torch.abs(torch.dot(x, torch.roll(y, -1)) - torch.dot(y, torch.roll(x, -1)))


def geometry_rows(name, geoms, seed=0, direction=""):
    rows = []
    for lab, lname in [(0, "normal"), (1, "anomaly")]:
        g = geoms[lab].detach().cpu()
        dists = torch.cdist(g, g)
        cov = (g.T @ g) / max(1, g.shape[0] - 1)
        eig = torch.linalg.eigvalsh(cov).sort(descending=True).values
        rows.append(
            {
                "seed": seed,
                "direction": direction,
                "variant": name,
                "label": lname,
                "center_error": float(g.mean(dim=0).norm()),
                "rms_energy": float(g.pow(2).sum(dim=-1).mean()),
                "min_pairwise_distance": float(dists[dists > 0].min()) if bool((dists > 0).any()) else 0.0,
                "max_pairwise_distance": float(dists.max()),
                "polygon_area": float(polygon_area(g)),
                "eig1": float(eig[0]),
                "eig2": float(eig[1]) if eig.numel() > 1 else 0.0,
                "coords": g.tolist(),
            }
        )
    return rows

