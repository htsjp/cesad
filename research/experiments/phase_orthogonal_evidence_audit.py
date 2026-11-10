import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_cross_source_mvtec import load_full_mvtec_source
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import evaluate_maps, fnum, write_json
from experiments.phase_private_residual_contrast_memory import contrast_logits
from experiments.phase_spatial_residual_refinement_standard_metric import aligned_grid
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np, patch_labels


OUT = Path("results/phase_orthogonal_evidence_audit/formal")
PAIRS = {
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
}
VARIANTS = ("B0_StateDirection", "E1_OrthogonalProbeOnly", "E2_EqualComplement", "E3_UncertaintyGatedComplement")
PROBE_EPOCHS = 5
PROBE_LR = 1e-4
SOURCE_CAP_PER_CLASS_CAT = 512


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


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def state_geometry(source_info):
    m = np.asarray(source_info["m"], dtype=np.float32)
    v = unit(source_info["v1"])
    alpha = float(source_info["alpha"])
    a = unit(m + alpha * v)
    n = unit(m - alpha * v)
    d = unit(a - n)
    return a, n, d, float(np.linalg.norm(a - n))


@torch.no_grad()
def feature_grid(source_info, patch_grid, device):
    if source_info["use_align"]:
        return aligned_grid(patch_grid, source_info["q_pc1"], source_info["align_model"], device).detach().cpu().numpy().reshape(SIDE * SIDE, -1).astype(np.float32)
    return norm_np(np.asarray(patch_grid, dtype=np.float32).reshape(SIDE * SIDE, -1)).astype(np.float32)


def source_dataset(source_name, smoke):
    if source_name == "Full VisA":
        _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    else:
        _name, dataset, indices, patch = load_full_mvtec_source(False)
    if smoke:
        by_cat = defaultdict(lambda: {0: [], 1: []})
        for pos, idx in enumerate(indices):
            item = dataset.data_all[int(idx)]
            by_cat[str(item["cls_name"])][int(item["anomaly"])].append((pos, idx))
        chosen = []
        for cat in sorted(by_cat)[:4]:
            chosen.extend(by_cat[cat][0][:3])
            chosen.extend(by_cat[cat][1][:3])
        patch = np.stack([np.asarray(patch[p], dtype=np.float32) for p, _idx in chosen], axis=0)
        indices = [idx for _p, idx in chosen]
    return dataset, list(indices), patch


def build_probe_source(source_info, d_np, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source_info['source'].replace(' ', '_')}_ORTH_SOURCE_{suffix}.npz"
    if cache.exists() and not smoke:
        data = np.load(cache)
        return data["x"].astype(np.float32), data["y"].astype(np.float32), data["cat"].astype(str)
    dataset, indices, patch = source_dataset(source_info["source"], smoke)
    rng = np.random.default_rng(SEED)
    by_key = defaultdict(list)
    cats = []
    d = d_np.astype(np.float32)
    for pos, idx in enumerate(indices):
        item = dataset[int(idx)]
        y_grid = patch_labels(item).astype(np.int64)
        z = feature_grid(source_info, patch[pos], device)
        z_perp = z - (z @ d)[:, None] * d[None, :]
        for cls in (0, 1):
            ids = np.flatnonzero(y_grid == cls)
            if ids.size == 0:
                continue
            key = (str(item["cls_name"]), cls)
            room = SOURCE_CAP_PER_CLASS_CAT - len(by_key[key])
            if room <= 0:
                continue
            if ids.size > room:
                ids = rng.choice(ids, size=room, replace=False)
            by_key[key].extend(z_perp[ids].tolist())
        if pos % 250 == 0:
            print(f"[SOER-Audit] source pool {source_info['source']} image {pos}/{len(indices)}", flush=True)
    xs, ys, cat_out = [], [], []
    for (cat, cls), vals in sorted(by_key.items()):
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float32)
        xs.append(arr)
        ys.append(np.full(arr.shape[0], cls, dtype=np.float32))
        cat_out.extend([cat] * arr.shape[0])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    cat = np.asarray(cat_out, dtype=object)
    np.savez_compressed(cache, x=x.astype(np.float16), y=y, cat=cat)
    return x, y, cat.astype(str)


def train_probe(source_info, d_np, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / "checkpoints" / f"{source_info['source'].replace(' ', '_')}_linear_probe_{suffix}.pt"
    if ckpt.exists() and not smoke:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["w"].astype(np.float32), float(payload["b"]), payload["log"]
    x_np, y_np, _cat = build_probe_source(source_info, d_np, device, smoke)
    w = torch.nn.Parameter(torch.zeros((x_np.shape[1],), dtype=torch.float32, device=device))
    b = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    opt = torch.optim.Adam([w, b], lr=PROBE_LR)
    rng = np.random.default_rng(SEED)
    epochs = 2 if smoke else PROBE_EPOCHS
    log = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(y_np))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_np[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_np[idx], dtype=torch.float32, device=device)
            logits = x @ w + b
            loss, bce, dice = balanced_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"source": source_info["source"], "epoch": epoch, "loss": float(np.mean(losses)), "lr": PROBE_LR}
        log.append(row)
        print(f"[SOER-Audit] train probe {source_info['source']} epoch={epoch}/{epochs} loss={row['loss']:.4f}", flush=True)
    payload = {"w": w.detach().cpu().numpy().astype(np.float32), "b": float(b.detach().cpu()), "log": log}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, ckpt)
    return payload["w"], payload["b"], log


def target_item(dataset, idx_or_rows, kind, pos):
    return dataset[int(idx_or_rows[pos])] if kind == "source_dataset" else dataset[pos]


def target_meta(dataset, idx_or_rows, kind, target, pos):
    item = target_item(dataset, idx_or_rows, kind, pos)
    if kind == "source_dataset":
        cat = str(item.get("cls_name", target))
        label = int(item.get("anomaly", 0))
    else:
        row = dict(idx_or_rows[pos])
        cat = str(row.get("category", item.get("cls_name", target)))
        label = int(row.get("label", item.get("anomaly", 0)))
    m = item["img_mask"].detach().cpu().numpy()
    return {"category": cat, "label": label, "mask": (np.squeeze(m) > 0.5).astype(np.uint8)}


def score_variants(source_info, d_np, norm_an, w_probe, b_probe, patch_grid, device):
    z = feature_grid(source_info, patch_grid, device)
    d = d_np.astype(np.float32)
    q0 = (z @ d) * norm_an
    z_perp = z - (z @ d)[:, None] * d[None, :]
    q_perp = z_perp @ w_probe.astype(np.float32) + float(b_probe)
    p0 = 1.0 / (1.0 + np.exp(-q0))
    u = 4.0 * p0 * (1.0 - p0)
    return {
        "B0_StateDirection": q0.reshape(SIDE, SIDE).astype(np.float32),
        "E1_OrthogonalProbeOnly": q_perp.reshape(SIDE, SIDE).astype(np.float32),
        "E2_EqualComplement": (q0 + q_perp).reshape(SIDE, SIDE).astype(np.float32),
        "E3_UncertaintyGatedComplement": (q0 + u * q_perp).reshape(SIDE, SIDE).astype(np.float32),
    }, z, z_perp, q0, q_perp


def selected_eval(source, target, maps, metas):
    old = tuple(lzp.VARIANTS)
    try:
        lzp.VARIANTS = VARIANTS
        return evaluate_maps(source, target, maps, metas)
    finally:
        lzp.VARIANTS = old


def evaluate_source(source_info, d_np, norm_an, w_probe, b_probe, device, smoke):
    suffix = "smoke" if smoke else "formal"
    raw, percat, corr_rows, uncertain_rows, energy_rows = [], [], [], [], []
    cells, _audits = protocol_cells(source_info["source"], device, smoke)
    for _cid, target, dataset, idx_or_rows, patch, kind in cells:
        if (source_info["source"], target) not in PAIRS:
            continue
        n = min(12, len(idx_or_rows)) if smoke else len(idx_or_rows)
        cache = OUT / f"{source_info['source'].replace(' ', '_')}_to_{target}_maps_{suffix}.npz"
        metas = [target_meta(dataset, idx_or_rows, kind, target, i) for i in range(n)]
        if cache.exists() and not smoke:
            data = np.load(cache, allow_pickle=True)
            maps = {v: [x.astype(np.float32) for x in data[v]] for v in VARIANTS}
            corr_rows.extend(data["corr"].tolist())
            uncertain_rows.extend(data["uncertain"].tolist())
            energy_rows.extend(data["energy"].tolist())
        else:
            maps = {v: [] for v in VARIANTS}
            corr, uncertain, energy = [], [], []
            for i in range(n):
                smaps, z, zperp, q0, qp = score_variants(source_info, d_np, norm_an, w_probe, b_probe, patch[i], device)
                for v in VARIANTS:
                    maps[v].append(smaps[v])
                meta = metas[i]
                gt37 = torch.tensor(meta["mask"].astype(np.float32)).view(1, 1, *meta["mask"].shape)
                gt37 = F.interpolate(gt37, size=(SIDE, SIDE), mode="nearest").view(-1).numpy() > 0.5
                for subset, mask in (("all", np.ones_like(gt37, dtype=bool)), ("normal", ~gt37), ("anomaly", gt37)):
                    if mask.sum() > 2:
                        corr.append({"source": source_info["source"], "dataset": target, "category": meta["category"], "image_index": i, "subset": subset, "pearson": float(stats.pearsonr(q0[mask], qp[mask]).statistic), "spearman": float(stats.spearmanr(q0[mask], qp[mask]).statistic)})
                        energy.append({"source": source_info["source"], "dataset": target, "category": meta["category"], "image_index": i, "subset": subset, "parallel_energy": float(np.mean((z[mask] @ d_np) ** 2)), "orthogonal_energy": float(np.mean(np.sum(zperp[mask] ** 2, axis=1))), "orthogonal_dot_mean_abs": float(np.mean(np.abs(zperp[mask] @ d_np))), "orthogonal_dot_max_abs": float(np.max(np.abs(zperp[mask] @ d_np)))})
                p0 = 1.0 / (1.0 + np.exp(-q0))
                um = (p0 >= 0.4) & (p0 <= 0.6)
                for subset, mask in (("uncertain_normal", um & ~gt37), ("uncertain_anomaly", um & gt37)):
                    vals = qp[mask]
                    uncertain.append({"source": source_info["source"], "dataset": target, "category": meta["category"], "image_index": i, "subset": subset, "count": int(vals.size), "mean_q_perp": float(vals.mean()) if vals.size else float("nan"), "median_q_perp": float(np.median(vals)) if vals.size else float("nan")})
                if i % 25 == 0:
                    print(f"[SOER-Audit] eval maps {source_info['source']} -> {target} image {i}/{n}", flush=True)
            corr_rows.extend(corr)
            uncertain_rows.extend(uncertain)
            energy_rows.extend(energy)
            np.savez_compressed(cache, **{v: np.stack(maps[v]).astype(np.float16) for v in VARIANTS}, corr=np.asarray(corr, dtype=object), uncertain=np.asarray(uncertain, dtype=object), energy=np.asarray(energy, dtype=object))
        rr, cc = selected_eval(source_info["source"], target, maps, metas)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "FULL_RES_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", percat)
    return raw, percat, corr_rows, uncertain_rows, energy_rows


def state_audit_row(source_info, d_np, norm_an, device, smoke):
    dataset, indices, patch = source_dataset(source_info["source"], True)
    item = dataset[int(indices[0])]
    z = feature_grid(source_info, patch[0], device)
    m = torch.tensor(source_info["m"], dtype=torch.float32, device=device)
    v = torch.tensor(source_info["v1"], dtype=torch.float32, device=device)
    q_two = contrast_logits(torch.tensor(z, dtype=torch.float32, device=device), m, source_info["alpha"], v).detach().cpu().numpy()
    q_dir = (z @ d_np) * norm_an
    a, n, _d, _norm = state_geometry(source_info)
    return {
        "source": source_info["source"],
        "norm_a_minus_n": norm_an,
        "cos_a_n": float(np.dot(a, n)),
        "tau_recorded": 1.0,
        "effective_contrast_norm": norm_an,
        "max_abs_logit_reconstruction_error": float(np.max(np.abs(q_two - q_dir))),
        "mean_abs_logit_reconstruction_error": float(np.mean(np.abs(q_two - q_dir))),
    }


def reproduction_rows(raw):
    rows = []
    for r in raw:
        if r["variant"] == "B0_StateDirection":
            rows.append({"source": r["source"], "dataset": r["dataset"], "variant": r["variant"], "M_FR_P_AUROC": r.get("M_FR_P_AUROC"), "M_FR_P_AP": r.get("M_FR_P_AP"), "M_FR_P_F1max": r.get("M_FR_P_F1max"), "M_FR_PRO": r.get("M_FR_PRO")})
    return rows


def morph_delta(percat):
    morph_path = Path("results/phase_final_bottleneck_attribution/formal/CATEGORY_MORPHOLOGY.csv")
    morph = read_csv(morph_path) if morph_path.exists() else []
    morph_by = {(r["source"], r["dataset"], r["category"]): r for r in morph}
    by = {(r["source"], r["dataset"], r["category"], r["variant"], r["protocol"]): r for r in percat}
    rows = []
    for key, b0 in by.items():
        source, dataset, cat, variant, protocol = key
        if variant != "B0_StateDirection" or protocol != "M":
            continue
        mo = morph_by.get((source, dataset, cat), {})
        for var in ("E2_EqualComplement", "E3_UncertaintyGatedComplement"):
            cur = by.get((source, dataset, cat, var, "M"))
            if cur:
                rows.append({
                    "source": source,
                    "dataset": dataset,
                    "category": cat,
                    "variant": var,
                    "B0_P_AP": fnum(b0, "M_FR_P_AP"),
                    "variant_P_AP": fnum(cur, "M_FR_P_AP"),
                    "delta_P_AP": fnum(cur, "M_FR_P_AP") - fnum(b0, "M_FR_P_AP"),
                    "anomaly_patch_ratio": fnum(mo, "anomaly_patch_ratio_mean"),
                    "thinness": fnum(mo, "thinness_mean"),
                })
    return rows


def decide(raw, corr_rows, morph_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    vals = []
    for source, dataset in sorted(PAIRS):
        b0 = by.get((source, dataset, "B0_StateDirection"))
        e3 = by.get((source, dataset, "E3_UncertaintyGatedComplement"))
        if b0 and e3:
            vals.append((source, dataset, fnum(e3, "M_FR_P_AP") - fnum(b0, "M_FR_P_AP")))
    arr = np.asarray([v[2] for v in vals], dtype=np.float64)
    corr = np.asarray([abs(fnum(r, "spearman")) for r in corr_rows if r.get("subset") == "all"], dtype=np.float64)
    mean_corr = float(np.nanmean(corr)) if corr.size else float("nan")
    ok = (
        dict(((s, d), x) for s, d, x in vals).get(("Full MVTec", "VisA"), -1e9) >= 1.5
        and dict(((s, d), x) for s, d, x in vals).get(("Full VisA", "MVTec"), -1e9) >= 0
        and all(dict(((s, d), x) for s, d, x in vals).get((s, "BTAD"), -1e9) >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and all(dict(((s, d), x) for s, d, x in vals).get((s, "MPDD"), -1e9) >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and all(dict(((s, d), x) for s, d, x in vals).get((s, "DTD-Synthetic"), -1e9) >= -1.0 for s in ("Full VisA", "Full MVTec"))
        and int(np.sum(arr >= -0.3)) >= 6
        and float(np.nanmean(arr)) >= 0.7
        and mean_corr < 0.9
    )
    if ok:
        outcome = "ORTHOGONAL_FEATURES_CONTAIN_TRANSFERABLE_ANOMALY_EVIDENCE"
    elif vals and float(np.nanmax(arr)) < 0.3:
        outcome = "ORTHOGONAL_FEATURES_HAVE_NO_TRANSFER_SIGNAL"
    elif vals and dict(((s, d), x) for s, d, x in vals).get(("Full MVTec", "VisA"), 0) > 0 and np.nanmean(arr) < 0:
        outcome = "ORTHOGONAL_EVIDENCE_IS_DATASET_SPECIFIC"
    else:
        outcome = "SOURCE_ORTHOGONAL_SIGNAL_DOES_NOT_TRANSFER"
    return outcome, {"E3_mean_delta_P_AP": float(np.nanmean(arr)) if arr.size else float("nan"), "E3_non_drop": int(np.sum(arr >= -0.3)) if arr.size else 0, "E3_deltas": {f"{s}->{d}": x for s, d, x in vals}, "mean_abs_q0_qperp_spearman": mean_corr}


def write_report(outcome, stats, raw, runtime):
    lines = [
        "# State-Orthogonal Evidence Recovery Phase 1",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "This is the orthogonal evidence audit. Existing Canonical Two-State + NTR checkpoints are frozen; only a source-only linear probe on z_perp is trained. No target statistics or tuning are used.",
        "",
        "## E3 Delta vs B0",
        "",
        "| Source | Target | Delta P-AP |",
        "|---|---|---:|",
    ]
    for k, v in stats.get("E3_deltas", {}).items():
        s, d = k.split("->")
        lines.append(f"| {s} | {d} | {v:+.4f} |")
    lines.extend(["", "## Decision Stats", "", "```json", json.dumps(stats, indent=2, default=json_default), "```", "", "Phase 2 was not run by this script.", "", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    if not args.smoke and not args.full:
        raise SystemExit("Refusing full audit by default. Use --smoke for run-through or --full for the formal run.")
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_raw, all_per, all_corr, all_uncertain, all_energy, all_probe, state_rows = [], [], [], [], [], [], []
    for source_info in (load_visa_source(device), load_mvtec_source()):
        _a, _n, d_np, norm_an = state_geometry(source_info)
        state_rows.append(state_audit_row(source_info, d_np, norm_an, device, args.smoke))
        w, b, log = train_probe(source_info, d_np, device, args.smoke)
        all_probe.extend(log)
        raw, per, corr, uncertain, energy = evaluate_source(source_info, d_np, norm_an, w, b, device, args.smoke)
        all_raw.extend(raw)
        all_per.extend(per)
        all_corr.extend(corr)
        all_uncertain.extend(uncertain)
        all_energy.extend(energy)
    morph = morph_delta(all_per)
    outcome, stats = decide(all_raw, all_corr, morph)
    runtime = time.time() - start
    write_csv(OUT / "B0_REPRODUCTION.csv", reproduction_rows(all_raw))
    write_csv(OUT / "STATE_DIRECTION_AUDIT.csv", state_rows)
    write_csv(OUT / "ORTHOGONAL_ENERGY.csv", all_energy)
    write_csv(OUT / "PROBE_TRAINING_LOG.csv", all_probe)
    write_csv(OUT / "FULL_RES_METRICS.csv", all_raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", all_per)
    write_csv(OUT / "SCORE_CORRELATION.csv", all_corr)
    write_csv(OUT / "UNCERTAIN_PATCH_AUDIT.csv", all_uncertain)
    write_csv(OUT / "MORPHOLOGY_DELTA_ANALYSIS.csv", morph)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime, "device": device, "smoke": bool(args.smoke), "full": bool(args.full)})
    write_report(outcome, stats, all_raw, runtime)
    print(f"[SOER-Audit] OUTCOME={outcome}", flush=True)
    print(f"[SOER-Audit] wrote {OUT}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
