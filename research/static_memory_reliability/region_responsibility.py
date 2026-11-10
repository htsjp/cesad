from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def infer_category(source_id):
    parts = [p for p in Path(str(source_id).replace("\\", "/")).parts if p not in {"", "/"}]
    lowered = [p.lower() for p in parts]
    if "test" in lowered:
        idx = lowered.index("test")
        if idx > 0:
            return parts[idx - 1]
    if "train" in lowered:
        idx = lowered.index("train")
        if idx > 0:
            return parts[idx - 1]
    return parts[-3] if len(parts) >= 3 else str(source_id)


def region_metadata(memory):
    entries = memory.get("entries", [])
    regions = memory["patch_keys"].shape[0] // 4
    labels = memory["patch_values"].argmax(dim=-1).reshape(regions, 4)[:, 0].long()
    source_ids, categories = [], []
    for r in range(regions):
        e = entries[r * 4] if entries else {}
        sid = str(e.get("source_id", f"region_{r}"))
        source_ids.append(sid)
        categories.append(infer_category(sid))
    return {"labels": labels, "source_ids": source_ids, "categories": categories}


def _masked_region_weights(query, keys, query_source_ids, region_source_ids, batch_size=128, temperature=1.0):
    region_count = keys.shape[0] // 4
    out = []
    for start in tqdm(range(0, query.shape[0], batch_size), desc="source responsibility weights", leave=False):
        q = query[start : start + batch_size].float()
        sim = q @ keys.float().T / float(temperature)
        batch_sources = query_source_ids[start : start + q.shape[0]]
        for i, sid in enumerate(batch_sources):
            mask_regions = [r for r, rsid in enumerate(region_source_ids) if rsid == sid]
            if mask_regions:
                idx = torch.tensor([4 * r + j for r in mask_regions for j in range(4)], dtype=torch.long, device=sim.device)
                sim[i, idx] = -1e4
        w = F.softmax(sim, dim=-1).reshape(q.shape[0], region_count, 4).sum(dim=-1)
        out.append(w.detach().cpu())
    return torch.cat(out, dim=0)


def compute_region_responsibility(memory, keys, scales=None, bootstrap_seeds=range(10), batch_size=128, eps=1e-6):
    meta = region_metadata(memory)
    labels = meta["labels"]
    region_source_ids = meta["source_ids"]
    region_categories = meta["categories"]
    query = memory["patch_keys"].float()
    query_labels = memory["patch_values"].argmax(dim=-1).long()
    query_source_ids = [memory["entries"][i].get("source_id", f"query_{i}") for i in range(query.shape[0])]
    query_categories = [infer_category(x) for x in query_source_ids]
    W = _masked_region_weights(query, keys, query_source_ids, region_source_ids, batch_size=batch_size)

    image_groups = defaultdict(list)
    for i, (cat, sid, y) in enumerate(zip(query_categories, query_source_ids, query_labels.tolist())):
        image_groups[(cat, sid, int(y))].append(i)
    image_rows = []
    for (cat, sid, y), idxs in image_groups.items():
        image_rows.append((cat, sid, y, W[idxs].mean(dim=0)))

    def balanced_mean(condition):
        by_cat = defaultdict(list)
        for cat, _sid, y, vec in image_rows:
            if condition(y):
                by_cat[cat].append(vec)
        if not by_cat:
            return torch.zeros(W.shape[1]), torch.zeros(W.shape[1])
        cat_means = torch.stack([torch.stack(v).mean(dim=0) for v in by_cat.values()])
        return cat_means.mean(dim=0), torch.tensor([len(v) for v in by_cat.values()], dtype=torch.float32)

    A = torch.zeros(W.shape[1])
    B = torch.zeros(W.shape[1])
    support_a = torch.zeros(W.shape[1])
    support_b = torch.zeros(W.shape[1])
    for lab in [0, 1]:
        idx = labels == lab
        a, ac = balanced_mean(lambda y, lab=lab: y == lab)
        b, bc = balanced_mean(lambda y, lab=lab: y != lab)
        A[idx] = a[idx]
        B[idx] = b[idx]
        support_a[idx] = float(ac.sum().item()) if ac.numel() else 0.0
        support_b[idx] = float(bc.sum().item()) if bc.numel() else 0.0
    U, _ = balanced_mean(lambda _y: True)
    D = torch.log((A + eps) / (B + eps))
    H = torch.log(U + eps)

    boot = []
    gen_rows = list(image_rows)
    for seed in bootstrap_seeds:
        g = torch.Generator().manual_seed(int(seed))
        by_cat = defaultdict(list)
        for row in gen_rows:
            by_cat[row[0]].append(row)
        sampled = []
        for rows in by_cat.values():
            if not rows:
                continue
            idx = torch.randint(0, len(rows), (len(rows),), generator=g)
            sampled.extend([rows[int(i)] for i in idx])
        old = image_rows
        image_rows = sampled
        Ab = torch.zeros(W.shape[1])
        Bb = torch.zeros(W.shape[1])
        for lab in [0, 1]:
            idx = labels == lab
            a, _ = balanced_mean(lambda y, lab=lab: y == lab)
            b, _ = balanced_mean(lambda y, lab=lab: y != lab)
            Ab[idx] = a[idx]
            Bb[idx] = b[idx]
        boot.append(torch.log((Ab + eps) / (Bb + eps)))
        image_rows = old
    boot = torch.stack(boot) if boot else D[None, :]
    sigma = boot.std(dim=0, unbiased=False)
    T = D / (sigma + eps)
    coverage = (A / A.sum().clamp_min(eps)).clamp_min(eps)
    density = -torch.cdist(keys.reshape(-1, 4, keys.shape[-1]).mean(dim=1), keys.reshape(-1, 4, keys.shape[-1]).mean(dim=1)).topk(k=min(6, W.shape[1]), largest=False).values[:, 1:].mean(dim=1)
    stats = {
        "A": A,
        "B": B,
        "U": U,
        "D": D,
        "H": H,
        "sigma": sigma,
        "T": T,
        "coverage": coverage,
        "density": density,
        "scale": torch.ones_like(D) if scales is None else scales.float(),
        "labels": labels,
        "categories": region_categories,
        "support_a": support_a,
        "support_b": support_b,
        "missing": ((support_a == 0) | (support_b == 0)).float(),
        "bootstrap_low": boot.quantile(0.025, dim=0),
        "bootstrap_high": boot.quantile(0.975, dim=0),
    }
    return stats

