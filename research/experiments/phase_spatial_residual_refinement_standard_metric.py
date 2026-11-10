import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, load_payload
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_spatial_supervision_transfer_audit import target_cells
from experiments.phase_standard_zsad_benchmark_infrastructure import build_sota_reference, build_sota_gap
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from experiments.standard_zsad_metrics import evaluate_zsad_standard, resize_mask_flat


OUT = Path("results/phase_spatial_residual_refinement_standard_metric/formal")
STD_OUT = Path("results/phase_standard_zsad_benchmark_infrastructure/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
EPOCHS = 10
LR = 0.001
TOP_NEG_FRAC = 0.10
HIST_B0 = {
    "MVTec": {"P-AUROC": 90.3345, "P-AP": 32.5154, "P-F1max": 38.1099, "PRO": 83.2430, "G_pixel": 61.4250},
    "BTAD": {"P-AUROC": 93.8786, "P-AP": 51.3092, "P-F1max": 54.5789, "PRO": 64.0710, "G_pixel": 72.5939},
    "CVC-ClinicDB": {"P-AUROC": 86.6497, "P-AP": 43.5498, "P-F1max": 46.8085, "PRO": 72.7344, "G_pixel": 65.0998},
}


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


class SpatialResidualRefiner(nn.Module):
    def __init__(self, dim=1024, hidden=32, spatial=True):
        super().__init__()
        torch.manual_seed(SEED)
        k = 3 if spatial else 1
        p = 1 if spatial else 0
        self.local = nn.Conv2d(dim, dim, kernel_size=k, padding=p, groups=dim, bias=True)
        self.fc1 = nn.Conv2d(dim, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, dim, kernel_size=1)
        self.ln = nn.LayerNorm(dim)
        nn.init.normal_(self.local.weight, std=0.01)
        nn.init.zeros_(self.local.bias)
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        # z: B,H,W,D
        x = z.permute(0, 3, 1, 2).contiguous()
        r = self.fc2(F.gelu(self.fc1(F.gelu(self.local(x)))))
        y = x + r
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.ln(y)
        return F.normalize(y, dim=-1)


def fixed_smooth(z):
    x = z.permute(0, 3, 1, 2).contiguous()
    m = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    y = 0.5 * x + 0.5 * m
    return F.normalize(y.permute(0, 2, 3, 1).contiguous(), dim=-1)


def load_base(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    v_hn = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32))
    align_model = make_model(hybrid_states(m_c, alpha * v_hn), corrupt_payload, device)
    for p in align_model.parameters():
        p.requires_grad_(False)
    align_model.eval()
    return q_pc1, align_model, m_c.astype(np.float32), alpha, v_hn.astype(np.float32)


@torch.no_grad()
def aligned_grid(patch_image, q_pc1, align_model, device):
    z0 = remove_pc1(np.asarray(patch_image, dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    x = torch.tensor(z0, dtype=torch.float32, device=device)
    z = align_model.adapt(x).reshape(SIDE, SIDE, 1024)
    return z


def make_split_slots(slots, labels_by_pos):
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    out = {}
    for cat, positions in slots.items():
        normal, anomaly = [], []
        for pos in positions:
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
            if y.max() > 0:
                anomaly.append(int(pos))
            else:
                normal.append(int(pos))
        out[cat] = {0: normal, 1: anomaly}
    return out


def sample_positions(split_slots, rng, step):
    positions = []
    for ci, cat in enumerate(sorted(split_slots)):
        want = (step + ci) % 2
        pool = split_slots[cat][want] or split_slots[cat][1 - want]
        if not pool:
            continue
        positions.append(int(rng.choice(pool)))
    return positions


def batch_tensors(positions, patch, labels_by_pos, q_pc1, align_model, device):
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    zs, ys = [], []
    for pos in positions:
        zs.append(aligned_grid(patch[int(pos)], q_pc1, align_model, device))
        ys.append(torch.tensor(np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.float32).reshape(SIDE, SIDE), device=device))
    return torch.stack(zs, dim=0), torch.stack(ys, dim=0)


def logits_from_z(z, m, alpha, v):
    return contrast_logits(z.reshape(-1, z.shape[-1]), m, alpha, v).reshape(z.shape[0], SIDE, SIDE)


def topk_set(logits, y, frac=TOP_NEG_FRAC):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    k = max(1, int(np.ceil(float(len(neg)) * frac)))
    chosen = neg[torch.topk(logits.reshape(-1).detach()[neg], k=k, largest=True).indices]
    return chosen


def random_set(y, count, rng, device):
    neg = torch.nonzero(y.reshape(-1) < 0.5, as_tuple=False).flatten()
    ids = rng.choice(len(neg), size=count, replace=len(neg) < count)
    return neg[torch.tensor(ids, dtype=torch.long, device=device)]


def train_refiner(variant, spatial, split_slots, labels_by_pos, patch, q_pc1, align_model, m_np, alpha, v_np, device, smoke=False):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model = SpatialResidualRefiner(spatial=spatial).to(device)
        model.load_state_dict(payload["model"], strict=True)
        return model.eval(), payload["curves"], payload["overlap"], payload["feature_rows"], True
    print(f"[SRR] train {variant}", flush=True)
    model = SpatialResidualRefiner(spatial=spatial).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    m = torch.tensor(m_np, dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    rng = np.random.default_rng(SEED)
    total_samples = len(split_slots) * 3 * 512
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 2)
    curves, overlap_rows, feature_rows = [], [], []
    for epoch in range(1, EPOCHS + 1):
        vals, ovs, coss, rnorms = [], [], [], []
        for step in range(steps_per_epoch):
            positions = sample_positions(split_slots, rng, step + epoch * 997)
            z, y = batch_tensors(positions, patch, labels_by_pos, q_pc1, align_model, device)
            base_logits = logits_from_z(z, m, alpha, v)
            z_ref = model(z)
            logits = logits_from_z(z_ref, m, alpha, v)
            loss_task, bce, dice = balanced_loss(logits.reshape(-1), y.reshape(-1))
            cur = topk_set(logits, y)
            if variant.startswith("E2"):
                sel = random_set(y, len(cur), rng, device)
            else:
                sel = cur
            extra = F.softplus(logits.reshape(-1)[sel]).mean()
            loss = loss_task + extra
            opt.zero_grad()
            loss.backward()
            opt.step()
            b0_sel = topk_set(base_logits, y)
            inter = len(set(sel.detach().cpu().numpy().tolist()) & set(b0_sel.detach().cpu().numpy().tolist()))
            union = len(set(sel.detach().cpu().numpy().tolist()) | set(b0_sel.detach().cpu().numpy().tolist()))
            ovs.append(inter / max(union, 1))
            coss.append(float((z_ref.reshape(-1, 1024) * z.reshape(-1, 1024)).sum(dim=-1).mean().detach().cpu()))
            rnorms.append(float((z_ref - z).reshape(-1, 1024).norm(dim=-1).mean().detach().cpu()))
            vals.append((float(loss.detach().cpu()), float(bce.cpu()), float(dice.cpu()), float(extra.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        curves.append({"variant": variant, "epoch": epoch, "loss": arr[:, 0].mean(), "bce": arr[:, 1].mean(), "dice": arr[:, 2].mean(), "extra_hard_loss": arr[:, 3].mean()})
        overlap_rows.append({"variant": variant, "epoch": epoch, "mean_hard_set_jaccard_vs_B0": float(np.mean(ovs)), "steps": steps_per_epoch})
        feature_rows.append({"variant": variant, "epoch": epoch, "mean_cos_Zref_Z": float(np.mean(coss)), "mean_residual_norm": float(np.mean(rnorms))})
    torch.save({"model": model.state_dict(), "curves": curves, "overlap": overlap_rows, "feature_rows": feature_rows}, ckpt)
    return model.eval(), curves, overlap_rows, feature_rows, False


@torch.no_grad()
def score_variant(variant, patch, q_pc1, align_model, m_np, alpha, v_np, device, model=None, smoothing=False, smoke=False, batch_images=4):
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    m = torch.tensor(m_np, dtype=torch.float32, device=device)
    v = torch.tensor(v_np, dtype=torch.float32, device=device)
    for start in range(0, n, batch_images):
        zs = [aligned_grid(patch[i], q_pc1, align_model, device) for i in range(start, min(start + batch_images, n))]
        z = torch.stack(zs, dim=0)
        if model is not None:
            z = model(z)
        if smoothing:
            z = fixed_smooth(z)
        logits = logits_from_z(z, m, alpha, v)
        s = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        b = s.shape[0]
        maps[start : start + b] = s.astype(np.float16)
        image[start : start + b] = np.asarray([top1_image_score(mm) for mm in s], dtype=np.float32)
    return image, maps


def eval_variants(variants, q_pc1, align_model, m_np, alpha, v_np, device, smoke):
    raw, percat = [], []
    diag_rows, boundary_rows = [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, cfg in variants.items():
            print(f"[SRR] eval {variant} on {cell_id}", flush=True)
            image, maps = score_variant(variant, patch, q_pc1, align_model, m_np, alpha, v_np, device, **cfg, smoke=smoke)
            metric, per = evaluate_zsad_standard(cell_id, target, dataset, idx_use, kind, image, maps, SIDE)
            metric.update({"dataset": cell_id, "variant": variant, "source": "Full VisA"})
            raw.append(metric)
            for cat, vals in per.items():
                percat.append({"dataset": cell_id, "target": target, "category": cat, "variant": variant, **vals})
            drows, brows = localization_diagnostics(cell_id, target, dataset, idx_use, kind, maps, variant)
            diag_rows.extend(drows)
            boundary_rows.extend(brows)
    return raw, percat, diag_rows + boundary_rows


def localization_diagnostics(cell_id, target, dataset, idx_or_rows, kind, maps, variant):
    vals = defaultdict(list)
    bvals = defaultdict(list)
    for pos, ref in enumerate(idx_or_rows):
        item = dataset[int(ref)] if kind == "source_dataset" else dataset[pos]
        if "img_mask" not in item:
            continue
        m = resize_mask_flat(item, SIDE).reshape(SIDE, SIDE).astype(bool)
        s = np.asarray(maps[pos], dtype=np.float32).reshape(SIDE, SIDE)
        if not np.any(m):
            vals["normal_score"].extend(s.reshape(-1).tolist())
            bvals["far_normal"].extend(s.reshape(-1).tolist())
            continue
        vals["anomaly_score"].extend(s[m].tolist())
        vals["normal_score"].extend(s[~m].tolist())
        interior = ndimage.binary_erosion(m, structure=np.ones((3, 3)), border_value=0)
        boundary = m & ~interior
        near = ndimage.binary_dilation(m, structure=np.ones((3, 3)), iterations=1) & ~m
        far = ~(m | near)
        for name, mask in (("anomaly_interior", interior), ("anomaly_boundary", boundary), ("normal_near_boundary", near), ("far_normal", far)):
            if np.any(mask):
                bvals[name].extend(s[mask].tolist())
    rows = []
    if vals["anomaly_score"] and vals["normal_score"]:
        a = np.asarray(vals["anomaly_score"], dtype=np.float32)
        n = np.asarray(vals["normal_score"], dtype=np.float32)
        rows.append({
            "dataset": cell_id,
            "target": target,
            "variant": variant,
            "record_type": "score_concentration",
            "anomaly_mean": float(a.mean()),
            "anomaly_p90": float(np.percentile(a, 90)),
            "normal_mean": float(n.mean()),
            "normal_p95": float(np.percentile(n, 95)),
            "mean_separation": float(a.mean() - n.mean()),
            "p90_minus_p95": float(np.percentile(a, 90) - np.percentile(n, 95)),
        })
    brows = []
    for name, xs in sorted(bvals.items()):
        if xs:
            arr = np.asarray(xs, dtype=np.float32)
            brows.append({"dataset": cell_id, "target": target, "variant": variant, "record_type": "boundary_region", "region": name, "mean_score": float(arr.mean()), "p90_score": float(np.percentile(arr, 90)), "count": int(arr.size)})
    return rows, brows


def load_b0_rows():
    raw = read_csv(STD_OUT / "STANDARD_METRICS.csv")
    per = read_csv(STD_OUT / "PER_CATEGORY_STANDARD_METRICS.csv")
    for r in raw:
        r["variant"] = "B0_HardNormal"
        r["source"] = "Full VisA"
    for r in per:
        r["variant"] = "B0_HardNormal"
    return raw, per


def compare_rows(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        for v in ("E1_SpatialResidualRefinement", "E2_PointwiseCapacityControl", "E3_FrozenSpatialSmoothing"):
            if (ds, v) not in by:
                continue
            r = by[(ds, v)]
            rows.append({
                "dataset": ds,
                "variant": v,
                "delta_P-AUROC_vs_B0": fnum(r, "P-AUROC") - fnum(b0, "P-AUROC"),
                "delta_P-AP_vs_B0": fnum(r, "P-AP") - fnum(b0, "P-AP"),
                "delta_P-F1max_vs_B0": fnum(r, "P-F1max") - fnum(b0, "P-F1max"),
                "delta_PRO_vs_B0": fnum(r, "PRO") - fnum(b0, "PRO"),
                "delta_G_vs_B0": fnum(r, "G_pixel") - fnum(b0, "G_pixel"),
            })
    return rows


def decide(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    targets = ("MVTec", "BTAD", "CVC-ClinicDB")
    e1_b0_g = np.asarray([fnum(by[(d, "E1_SpatialResidualRefinement")], "G_pixel") - fnum(by[(d, "B0_HardNormal")], "G_pixel") for d in targets], dtype=np.float32)
    e1_b0_ap = {d: fnum(by[(d, "E1_SpatialResidualRefinement")], "P-AP") - fnum(by[(d, "B0_HardNormal")], "P-AP") for d in targets}
    e1_e2_ap_mean_mc = np.mean([
        fnum(by[(d, "E1_SpatialResidualRefinement")], "P-AP") - fnum(by[(d, "E2_PointwiseCapacityControl")], "P-AP")
        for d in ("MVTec", "CVC-ClinicDB")
    ])
    e1_e3_g = np.asarray([fnum(by[(d, "E1_SpatialResidualRefinement")], "G_pixel") - fnum(by[(d, "E3_FrozenSpatialSmoothing")], "G_pixel") for d in targets], dtype=np.float32)
    ap_mean_mc = np.mean([e1_b0_ap["MVTec"], e1_b0_ap["CVC-ClinicDB"]])
    btad_ap_drop = e1_b0_ap["BTAD"]
    btad_g_drop = e1_b0_g[1]
    f1_or_pro = (
        fnum(by[("MVTec", "E1_SpatialResidualRefinement")], "P-F1max") - fnum(by[("MVTec", "B0_HardNormal")], "P-F1max") >= 1.5
        or fnum(by[("CVC-ClinicDB", "E1_SpatialResidualRefinement")], "P-F1max") - fnum(by[("CVC-ClinicDB", "B0_HardNormal")], "P-F1max") >= 1.5
        or fnum(by[("MVTec", "E1_SpatialResidualRefinement")], "PRO") - fnum(by[("MVTec", "B0_HardNormal")], "PRO") >= 1.5
        or fnum(by[("CVC-ClinicDB", "E1_SpatialResidualRefinement")], "PRO") - fnum(by[("CVC-ClinicDB", "B0_HardNormal")], "PRO") >= 1.5
    )
    success = (
        (e1_b0_ap["MVTec"] >= 3.0 or e1_b0_ap["CVC-ClinicDB"] >= 2.0)
        and ap_mean_mc >= 2.0
        and e1_e2_ap_mean_mc >= 1.0
        and int(np.sum(e1_b0_g > 0)) >= 2
        and float(e1_b0_g.mean()) >= 0.7
        and btad_ap_drop >= -1.5
        and btad_g_drop >= -1.0
        and f1_or_pro
    )
    stats = {
        "MVTec_P_AP_gain": e1_b0_ap["MVTec"],
        "ClinicDB_P_AP_gain": e1_b0_ap["CVC-ClinicDB"],
        "MVTec_ClinicDB_mean_P_AP_gain": float(ap_mean_mc),
        "E1_minus_E2_MVTec_ClinicDB_mean_P_AP": float(e1_e2_ap_mean_mc),
        "targets_with_G_gain": int(np.sum(e1_b0_g > 0)),
        "mean_G_gain": float(e1_b0_g.mean()),
        "BTAD_P_AP_gain": btad_ap_drop,
        "BTAD_G_gain": float(btad_g_drop),
        "mean_E1_minus_E3_G": float(e1_e3_g.mean()),
    }
    if success:
        return "SPATIAL_RESIDUAL_REFINEMENT_IMPROVES_DENSE_RANKING", stats
    if abs(stats["E1_minus_E2_MVTec_ClinicDB_mean_P_AP"]) < 1.0:
        return "EXTRA_MAPPING_CAPACITY_NOT_SPATIAL_INTERACTION", stats
    if abs(stats["mean_E1_minus_E3_G"]) < 0.7:
        return "SIMPLE_SPATIAL_SMOOTHING_SUFFICIENT", stats
    auroc_gain = np.asarray([fnum(by[(d, "E1_SpatialResidualRefinement")], "P-AUROC") - fnum(by[(d, "B0_HardNormal")], "P-AUROC") for d in targets], dtype=np.float32)
    if auroc_gain.mean() > 0 and stats["MVTec_ClinicDB_mean_P_AP_gain"] <= 0 and not f1_or_pro:
        return "SPATIAL_REFINEMENT_IMPROVES_SEPARATION_NOT_LOCALIZATION_PRECISION", stats
    if e1_b0_g[1] > 0 and e1_b0_g[0] <= 0 and e1_b0_g[2] <= 0:
        return "SPATIAL_REFINEMENT_DOMAIN_DEPENDENT", stats
    return "SPATIAL_RESIDUAL_REFINEMENT_NO_SIGNAL", stats


def write_report(raw, comp, outcome, stats, sota_gap, runtime):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Spatial Residual Refinement - Standard Metric Formal Gate",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Strict source-only: target training/statistics/normal reference/adaptation were not used.",
        "",
        "## Metrics",
        "",
        "| Variant | Target | P-AUROC | P-AP | P-F1max | PRO | G |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for v in ("B0_HardNormal", "E1_SpatialResidualRefinement", "E2_PointwiseCapacityControl", "E3_FrozenSpatialSmoothing"):
        for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
            r = by[(ds, v)]
            lines.append(f"| {v} | {ds} | {fnum(r,'P-AUROC'):.4f} | {fnum(r,'P-AP'):.4f} | {fnum(r,'P-F1max'):.4f} | {fnum(r,'PRO'):.4f} | {fnum(r,'G_pixel'):.4f} |")
    lines.extend([
        "",
        "## Gate Stats",
        "",
        f"- MVTec P-AP gain: {stats['MVTec_P_AP_gain']:+.4f}",
        f"- ClinicDB P-AP gain: {stats['ClinicDB_P_AP_gain']:+.4f}",
        f"- MVTec+ClinicDB mean P-AP gain: {stats['MVTec_ClinicDB_mean_P_AP_gain']:+.4f}",
        f"- E1-E2 MVTec+ClinicDB mean P-AP: {stats['E1_minus_E2_MVTec_ClinicDB_mean_P_AP']:+.4f}",
        f"- Mean G gain: {stats['mean_G_gain']:+.4f}",
        f"- BTAD P-AP/G gain: {stats['BTAD_P_AP_gain']:+.4f} / {stats['BTAD_G_gain']:+.4f}",
        "",
        "## External SOTA Position",
        "",
        "| Reference | Target | Variant | Delta P-AUROC | Delta P-AP | Delta G |",
        "|---|---|---|---:|---:|---:|",
    ])
    for r in sota_gap:
        if r.get("comparison_type") == "DIRECT_COMPARABLE" and r.get("variant") == "E1_SpatialResidualRefinement":
            lines.append(f"| {r['reference']} | {r['target']} | {r['variant']} | {fnum(r,'delta_P_AUROC'):+.4f} | {fnum(r,'delta_P_AP'):+.4f} | {fnum(r,'delta_G'):+.4f} |")
    lines.extend(["", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SRR] device={device}", flush=True)
    q_pc1, align_model, m_np, alpha, v_np = load_base(device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    split_slots = make_split_slots(slots, labels_by_pos)

    e1, curves1, overlap1, feat1, reused1 = train_refiner("E1_SpatialResidualRefinement", True, split_slots, labels_by_pos, patch, q_pc1, align_model, m_np, alpha, v_np, device, args.smoke)
    e2, curves2, overlap2, feat2, reused2 = train_refiner("E2_PointwiseCapacityControl", False, split_slots, labels_by_pos, patch, q_pc1, align_model, m_np, alpha, v_np, device, args.smoke)
    eval_raw, eval_per, diag = eval_variants({
        "E1_SpatialResidualRefinement": {"model": e1, "smoothing": False},
        "E2_PointwiseCapacityControl": {"model": e2, "smoothing": False},
        "E3_FrozenSpatialSmoothing": {"model": None, "smoothing": True},
    }, q_pc1, align_model, m_np, alpha, v_np, device, args.smoke)
    b0_raw, b0_per = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_per = b0_per[:3]
    raw = b0_raw + eval_raw
    percat = b0_per + eval_per
    comp = compare_rows(raw)
    outcome, stats = decide(raw)
    refs = build_sota_reference()
    sota_gap = []
    for variant in ("B0_HardNormal", "E1_SpatialResidualRefinement"):
        rows_copy = [dict(r, variant=variant if r["variant"] == variant else r["variant"]) for r in raw if r["variant"] == variant]
        sota_gap.extend(build_sota_gap(rows_copy, refs, variant=variant))
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "LOCALIZATION_DIAGNOSTICS.csv", diag + comp)
    write_csv(OUT / "FEATURE_REFINEMENT.csv", feat1 + feat2 + curves1 + curves2)
    write_csv(OUT / "HARDNORMAL_OVERLAP.csv", overlap1 + overlap2)
    write_csv(OUT / "SOTA_GAP.csv", sota_gap)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_statistics": False,
        "target_training": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "checkpoints_reused": {"E1": bool(reused1), "E2": bool(reused2)},
    })
    write_report(raw, comp, outcome, stats, sota_gap, runtime)
    print(f"[SRR] outcome={outcome}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
