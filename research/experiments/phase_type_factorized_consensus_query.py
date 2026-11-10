import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import CHECKPOINT, EPOCHS, LR, SEED, SIDE, dice_loss, make_mvec_target, make_source_records, project_patch_image
from experiments.phase_liv_memory_support_modes import eval_maps, external_cell
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from models.mlp import Projector


OUT = Path("results/phase_type_factorized_consensus_query")


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def load_patch_proj(device):
    patch_proj = Projector(1024, 768, length=2).to(device).eval()
    ckpt = torch.load(CHECKPOINT, map_location=device)
    patch_proj.load_state_dict(ckpt["patch_proj"])
    for p in patch_proj.parameters():
        p.requires_grad_(False)
    return patch_proj


def projected_source_cache(out, records, patch, patch_proj, device):
    path = out / "source_projected_records.float16.npy"
    shape = (len(records), SIDE * SIDE, 768)
    if path.exists():
        arr = np.load(path, mmap_mode="r")
        if arr.shape == shape:
            return arr
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
    with torch.no_grad():
        for i, rec in enumerate(records):
            if i % 100 == 0:
                print(f"[TypeConsensus] project source {i}/{len(records)}", flush=True)
            z = project_patch_image(patch_proj, patch[rec["pos"]], device).detach().cpu().numpy().astype(np.float16)
            arr[i] = z
    del arr
    return np.load(path, mmap_mode="r")


class QuerySet(nn.Module):
    def __init__(self, n, dim=768):
        super().__init__()
        torch.manual_seed(SEED)
        self.q = nn.Parameter(torch.randn(n, dim) * 0.02)

    def forward(self):
        return self.q / self.q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def train_queries(z_cache, records, groups, normal_ids, out, tag, device):
    qset = QuerySet(len(groups)).to(device)
    opt = torch.optim.AdamW(qset.parameters(), lr=LR)
    rng = random.Random(SEED)
    max_pos = max(len(v) for v in groups.values())
    group_names = list(groups.keys())
    curves = []
    for epoch in range(EPOCHS):
        total = 0.0
        steps = 0
        for gi, name in enumerate(group_names):
            pos_ids = groups[name]
            for _ in range(max_pos):
                rid_pos = rng.choice(pos_ids)
                rid_norm = rng.choice(normal_ids)
                q = qset()[gi]
                zp = torch.tensor(np.asarray(z_cache[rid_pos], dtype=np.float32), device=device)
                zn = torch.tensor(np.asarray(z_cache[rid_norm], dtype=np.float32), device=device)
                yp = torch.tensor(records[rid_pos]["mask"], dtype=torch.float32, device=device)
                logp = zp @ q
                logn = zn @ q
                loss = F.binary_cross_entropy_with_logits(logp, yp) + dice_loss(logp, yp)
                loss = loss + F.binary_cross_entropy_with_logits(logn, torch.zeros_like(logn)) + dice_loss(logn, torch.zeros_like(logn))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.detach().cpu())
                steps += 1
        row = {"tag": tag, "epoch": epoch + 1, "loss": total / max(1, steps), "groups": len(groups), "steps": steps}
        curves.append(row)
        print(f"[TypeConsensus] {tag} epoch {epoch+1}/{EPOCHS} loss={row['loss']:.4f}", flush=True)
    with torch.no_grad():
        q = qset().detach().cpu().numpy().astype(np.float32)
    return q, curves


def train_shared_query(z_cache, records, normal_ids, anomaly_ids, device):
    return train_queries(z_cache, records, {"shared": anomaly_ids}, normal_ids, OUT / "tmp", "B0_SharedQuery", device)


def true_groups(records):
    groups = defaultdict(list)
    normal = []
    anomalies = []
    for i, r in enumerate(records):
        if r["normal"]:
            normal.append(i)
        else:
            groups[r["defect_name"]].append(i)
            anomalies.append(i)
    return dict(sorted(groups.items())), normal, anomalies


def random_groups_like(groups, seed=SEED):
    rng = random.Random(seed)
    all_ids = [i for ids in groups.values() for i in ids]
    rng.shuffle(all_ids)
    out = {}
    pos = 0
    for name, ids in groups.items():
        n = len(ids)
        out[f"pseudo_{name}"] = all_ids[pos:pos + n]
        pos += n
    return out


def consensus(q):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    m = q.mean(axis=0)
    strength = float(np.linalg.norm(m))
    return (m / max(strength, 1e-12)).astype(np.float32), strength


def query_geometry(q, q_cons, prefix):
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    sims = q @ q.T
    tri = sims[np.triu_indices_from(sims, k=1)]
    rows = [{
        "variant": prefix,
        "stat": "summary",
        "num_queries": int(q.shape[0]),
        "pairwise_cosine_mean": float(np.mean(tri)) if tri.size else float("nan"),
        "pairwise_cosine_std": float(np.std(tri)) if tri.size else float("nan"),
        "pairwise_cosine_min": float(np.min(tri)) if tri.size else float("nan"),
        "pairwise_cosine_max": float(np.max(tri)) if tri.size else float("nan"),
        "ConsensusStrength": float(np.linalg.norm(q.mean(axis=0))),
    }]
    cq = q @ q_cons
    for i, v in enumerate(cq):
        rows.append({"variant": prefix, "stat": "query_to_consensus", "query_id": i, "cos_q_to_consensus": float(v)})
    return rows


def target_cells(smoke):
    target_dataset, target_indices, target_patch, _vocab = make_mvec_target(smoke)
    cells = [("MVTec", "MVTec", target_dataset, target_indices, target_patch, "source_dataset")]
    cells.append(("BTAD", *external_cell("BTAD", smoke)[1:]))
    cells.append(("CVC-ClinicDB", *external_cell("CVC-ClinicDB", smoke)[1:]))
    return cells


def score_query_on_target(q_cons, patch, patch_proj, device, smoke):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    q = torch.tensor(q_cons, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n):
            z = project_patch_image(patch_proj, patch[i], device)
            s = (z @ q).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            maps[i] = s.astype(np.float16)
            image[i] = top1_image_score(s)
    return image, maps


def evaluate_all(queries, patch_proj, device, smoke):
    raw, percat = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, q in queries.items():
            print(f"[TypeConsensus] eval {variant} on {cell_id}", flush=True)
            image, maps = score_query_on_target(q, patch, patch_proj, device, smoke)
            metric, per = eval_maps(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat


def summarize_partition(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in sorted({r["dataset"] for r in raw}):
        b0 = by[(ds, "B0_SharedQuery")]
        e1 = by[(ds, "E1_TrueTypeConsensus")]
        e2 = by[(ds, "E2_RandomPartitionConsensus")]
        rows.append({
            "dataset": ds,
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
        })
    return rows


def decide(raw):
    rows = summarize_partition(raw)
    e1_b0 = [r["E1_minus_B0_G"] for r in rows]
    e1_e2 = [r["E1_minus_E2_G"] for r in rows]
    if sum(v > 0 for v in e1_b0) >= 2 and float(np.mean(e1_b0)) >= 0.5 and min(e1_b0) >= -1.0:
        if float(np.mean(e1_e2)) >= 0.5:
            return "TYPE_FACTORIZED_CONSENSUS_SUPPORTED", rows
        return "CONSENSUS_GAIN_NOT_SEMANTIC", rows
    return "TYPE_FACTORIZATION_NO_SIGNAL", rows


def write_report(out, outcome, partition_rows, raw, geom):
    lines = ["# Type-Factorized Consensus Anomaly Query", "", f"Outcome: `{outcome}`", "", "| Dataset | Variant | P-AUROC | P-AP | G_pixel |", "|---|---|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['dataset']} | {r['variant']} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} | {float(r['G_pixel']):.4f} |")
    lines.extend(["", "## Partition Deltas", "", "| Dataset | E1-B0 G | E2-B0 G | E1-E2 G |", "|---|---:|---:|---:|"])
    for r in partition_rows:
        lines.append(f"| {r['dataset']} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    summaries = [r for r in geom if r.get("stat") == "summary"]
    lines.extend(["", "## Query Geometry", ""])
    for r in summaries:
        lines.append(f"- {r['variant']}: ConsensusStrength={float(r['ConsensusStrength']):.4f}, pairwise mean={float(r['pairwise_cosine_mean']):.4f}.")
    lines.append("")
    lines.append("STOP: no weighted mean, PCA/SVD query, multi-query inference, max ensemble, defect text, prompt, cross-attention, Memory, DINO, or target tuning was started.")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[TypeConsensus] start smoke={args.smoke} device={device}", flush=True)
    patch_proj = load_patch_proj(device)
    _ds, _indices, source_patch, records, _source_names = make_source_records(args.smoke)
    z_cache = projected_source_cache(out, records, source_patch, patch_proj, device)
    groups, normal_ids, anomaly_ids = true_groups(records)
    count_rows = [{"partition": "true", "group": k, "count": len(v)} for k, v in groups.items()]
    b0_q, b0_curve = train_shared_query(z_cache, records, normal_ids, anomaly_ids, device)
    e1_q, e1_curve = train_queries(z_cache, records, groups, normal_ids, out, "E1_TrueTypeConsensus", device)
    rgroups = random_groups_like(groups)
    count_rows.extend({"partition": "random", "group": k, "count": len(v)} for k, v in rgroups.items())
    e2_q, e2_curve = train_queries(z_cache, records, rgroups, normal_ids, out, "E2_RandomPartitionConsensus", device)
    b0_cons = b0_q[0] / max(np.linalg.norm(b0_q[0]), 1e-12)
    e1_cons, e1_strength = consensus(e1_q)
    e2_cons, e2_strength = consensus(e2_q)
    geom = []
    geom.extend(query_geometry(e1_q, e1_cons, "E1_TrueTypeConsensus"))
    geom.extend(query_geometry(e2_q, e2_cons, "E2_RandomPartitionConsensus"))
    geom.append({"variant": "B0_SharedQuery", "stat": "summary", "num_queries": 1, "ConsensusStrength": 1.0, "pairwise_cosine_mean": float("nan")})
    queries = {
        "B0_SharedQuery": b0_cons,
        "E1_TrueTypeConsensus": e1_cons,
        "E2_RandomPartitionConsensus": e2_cons,
    }
    raw, percat = evaluate_all(queries, patch_proj, device, args.smoke)
    outcome, partition_rows = decide(raw)
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_CATEGORY.csv", percat)
    write_csv(out / "QUERY_GEOMETRY.csv", geom)
    write_csv(out / "TRUE_VS_RANDOM_PARTITION.csv", count_rows + partition_rows)
    write_csv(out / "TRAIN_CURVE.csv", b0_curve + e1_curve + e2_curve)
    write_json(out / "OUTCOME.json", {
        "outcome": outcome,
        "smoke": args.smoke,
        "runtime_sec": time.time() - start,
        "strict_source_only": True,
        "target_defect_names_used": False,
        "text_used": False,
        "query_strength": {"E1": e1_strength, "E2": e2_strength},
    })
    write_report(out, outcome, partition_rows, raw, geom)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
