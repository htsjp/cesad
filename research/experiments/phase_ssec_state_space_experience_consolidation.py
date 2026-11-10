import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source, protocol_cells
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import item_at, item_meta, mask_np, torch_resize
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_scee_lite_source_composition_experience_expansion import (
    PATCH_SAMPLE_SYN,
    SYN_OUT as SCEE_SYN_OUT,
    OUT as SCEE_OUT,
    crop_z_grids,
    fnum,
    load_runtime,
    scorer_z_grid,
)
from experiments.phase_sgpr import evaluate_maps, stitch_quadrant74
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs


OUT = Path("results/phase_ssec_state_space_experience_consolidation/formal")
VARIANTS = ("E0_Baseline", "E1_DirectSCEE", "E2_ProxyMainConsolidation")
HARD_PAIRS = (("Full MVTec", "VisA"), ("Full VisA", "MPDD"))
STABILITY_PAIR = ("Full MVTec", "BTAD")
BETA = 0.05
EPOCHS = 10
PROXY_STEPS = 4
ANCHOR_STEPS = 1
SYN_VARIANT = "E2_CoverageBiasedTransplant"


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def finite_mean(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def finite_median(vals):
    arr = np.asarray(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else float("nan")


def unpack_groups(result):
    return result[0] if isinstance(result, tuple) else result


def model_states(model_info):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = unit_np(np.asarray(model_info["v1"], dtype=np.float32))
    return (m - alpha * v).astype(np.float32), (m + alpha * v).astype(np.float32)


def states_from_v(model_info, v):
    m = np.asarray(model_info["m"], dtype=np.float32)
    alpha = float(model_info["alpha"])
    v = unit_np(np.asarray(v, dtype=np.float32))
    return (m - alpha * v).astype(np.float32), (m + alpha * v).astype(np.float32)


def contrast_direction(s_n, s_a):
    d = F.normalize(s_a, dim=-1) - F.normalize(s_n, dim=-1)
    return F.normalize(d, dim=-1)


def two_state_logits(z, s_n, s_a):
    z = F.normalize(z, dim=-1)
    s_n = F.normalize(s_n, dim=-1)
    s_a = F.normalize(s_a, dim=-1)
    return z @ s_a - z @ s_n


def two_state_loss(z, y, s_n, s_a):
    logits = two_state_logits(z, s_n, s_a)
    task, bce, dice = balanced_loss(logits, y)
    neg = torch.nonzero(y < 0.5, as_tuple=False).flatten()
    if neg.numel() == 0:
        hard = torch.zeros((), dtype=logits.dtype, device=logits.device)
    else:
        k = max(1, int(math.ceil(float(neg.numel()) * TOP_NEG_FRAC)))
        sel = neg[torch.topk(logits.detach()[neg], k=k, largest=True).indices]
        hard = F.softplus(logits[sel]).mean()
    return task + hard, task.detach(), hard.detach(), bce.detach(), dice.detach()


def balanced_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys, origins = [], [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(data["y"] == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(data["x"][choice])
            ys.append(data["y"][choice])
            origins.extend(data.get("origin", np.asarray(["real"] * len(data["y"]), dtype=object))[choice].tolist())
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order], np.asarray(origins, dtype=object)[order]


def encode_saved_synthetic(source, model_info, clip_model, preprocess, device, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{source.replace(' ', '_')}_coverage_biased_synth_features_{suffix}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        return [(str(c), data["x"][i], data["y"][i], "biased") for i, c in enumerate(data["cat"])]

    rows = [r for r in read_csv(SCEE_OUT / "SYNTHETIC_SAMPLE_AUDIT.csv") if r.get("source") == source and r.get("variant") == SYN_VARIANT]
    if smoke:
        rows = rows[: min(12, len(rows))]
    xs, ys, cats = [], [], []
    for i, row in enumerate(rows):
        if not row.get("preview_image") or not row.get("preview_mask"):
            continue
        img_path = Path(row["preview_image"])
        mask_path = Path(row["preview_mask"])
        if not img_path.is_absolute():
            img_path = ROOT / img_path
        if not mask_path.is_absolute():
            mask_path = ROOT / mask_path
        if not img_path.exists() or not mask_path.exists() or not img_path.is_file() or not mask_path.is_file():
            continue
        img = Image.open(img_path).convert("RGB")
        batch = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = visual_outputs(clip_model, batch, adapter=None, grad=False)
            patch = normalized_patch_feature(out).float().cpu().numpy().astype(np.float32)[0]
            z = scorer_z_grid(model_info, patch, device).detach().cpu().numpy().reshape(SIDE * SIDE, -1).astype(np.float32)
        mask = (np.asarray(Image.open(mask_path).convert("L")) > 0).astype(np.float32)
        t = torch.tensor(mask).view(1, 1, *mask.shape)
        y = (F.interpolate(t, size=(SIDE, SIDE), mode="nearest").view(-1).numpy() > 0.5).astype(np.float32)
        xs.append(z)
        ys.append(y)
        cats.append(row.get("receiver_category", row.get("donor_category", source)))
        img.close()
        if i % 50 == 0:
            print(f"[SSEC] encode synthetic {source} {i}/{len(rows)}", flush=True)
    if not xs:
        raise RuntimeError(f"No saved SCEE synthetic samples found for {source}")
    x = np.stack(xs, axis=0)
    y = np.stack(ys, axis=0)
    cat = np.asarray(cats, dtype=object)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=x, y=y, cat=cat)
    return [(str(cat[i]), x[i], y[i], "biased") for i in range(len(cat))]


def synthetic_groups_from_pack(base_groups, pack):
    groups = {c: {"x": np.empty((0, g["x"].shape[1]), dtype=np.float32), "y": np.empty((0,), dtype=np.float32), "origin": np.empty((0,), dtype=object)} for c, g in base_groups.items()}
    rng = np.random.default_rng(SEED)
    for cat, z, y, mode in pack:
        if cat not in groups:
            cat = sorted(groups)[int(rng.integers(0, len(groups)))]
        pos = np.flatnonzero(y > 0.5)
        neg = np.flatnonzero(y < 0.5)
        if pos.size == 0 or neg.size == 0:
            continue
        p_take = min(PATCH_SAMPLE_SYN // 2, pos.size)
        n_take = min(PATCH_SAMPLE_SYN // 2, neg.size)
        ids = np.concatenate([
            rng.choice(pos, p_take, replace=pos.size < p_take),
            rng.choice(neg, n_take, replace=neg.size < n_take),
        ])
        groups[cat]["x"] = np.concatenate([groups[cat]["x"], z[ids].astype(np.float32)], axis=0)
        groups[cat]["y"] = np.concatenate([groups[cat]["y"], y[ids].astype(np.float32)], axis=0)
        groups[cat]["origin"] = np.concatenate([groups[cat]["origin"], np.asarray([f"synthetic_{mode}"] * len(ids), dtype=object)], axis=0)
    return {c: g for c, g in groups.items() if len(g["y"]) > 0}


def clip_parameter_checksum(model):
    h = hashlib.sha256()
    with torch.no_grad():
        for k, v in model.named_parameters():
            h.update(k.encode("utf-8"))
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def train_ssec(source, model_info, real_groups, synth_groups, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{source.replace(' ', '_')}_E2_ProxyMainConsolidation_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["s_n"].astype(np.float32), payload["s_a"].astype(np.float32), payload["trajectory"], True

    print(f"[SSEC] train {source} E2 beta={BETA}", flush=True)
    s_n0, s_a0 = model_states(model_info)
    s_n_base = torch.tensor(s_n0, dtype=torch.float32, device=device)
    s_a_base = torch.tensor(s_a0, dtype=torch.float32, device=device)
    main_n = torch.nn.Parameter(s_n_base.clone())
    main_a = torch.nn.Parameter(s_a_base.clone())
    opt = torch.optim.AdamW([main_n, main_a], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    real_total = sum(len(g["y"]) for g in real_groups.values())
    real_steps = max(1, int(math.ceil(real_total / BATCH_SIZE)))
    if smoke:
        real_steps = min(real_steps, 2)
    proxy_steps = 1 if smoke else PROXY_STEPS
    trajectory = []
    prev_n = main_n.detach().clone()
    prev_a = main_a.detach().clone()
    for epoch in range(1, EPOCHS + 1):
        real_vals = []
        for _ in range(real_steps):
            xb, yb, _orig = balanced_batch(real_groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            loss, task, hard, bce, dice = two_state_loss(z, y, main_n, main_a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            real_vals.append((float(loss.detach().cpu()), float(task.cpu()), float(hard.cpu()), float(bce.cpu()), float(dice.cpu())))

        proxy_n = torch.nn.Parameter(main_n.detach().clone())
        proxy_a = torch.nn.Parameter(main_a.detach().clone())
        popt = torch.optim.AdamW([proxy_n, proxy_a], lr=LR, weight_decay=0.0)
        synth_vals = []
        for _ in range(proxy_steps):
            xb, yb, _orig = balanced_batch(synth_groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            loss, task, hard, bce, dice = two_state_loss(z, y, proxy_n, proxy_a)
            popt.zero_grad()
            loss.backward()
            popt.step()
            synth_vals.append((float(loss.detach().cpu()), float(task.cpu()), float(hard.cpu()), float(bce.cpu()), float(dice.cpu())))

        with torch.no_grad():
            before_n = main_n.detach().clone()
            before_a = main_a.detach().clone()
            main_n.copy_(F.normalize((1.0 - BETA) * main_n + BETA * proxy_n, dim=-1))
            main_a.copy_(F.normalize((1.0 - BETA) * main_a + BETA * proxy_a, dim=-1))
            displacement = 0.5 * (torch.norm(main_n - before_n) + torch.norm(main_a - before_a)).item()

        anchor_vals = []
        for _ in range(ANCHOR_STEPS):
            xb, yb, _orig = balanced_batch(real_groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            loss, task, hard, bce, dice = two_state_loss(z, y, main_n, main_a)
            opt.zero_grad()
            loss.backward()
            opt.step()
            anchor_vals.append((float(loss.detach().cpu()), float(task.cpu()), float(hard.cpu()), float(bce.cpu()), float(dice.cpu())))

        d_base = contrast_direction(s_n_base, s_a_base)
        d_main = contrast_direction(main_n.detach(), main_a.detach())
        d_proxy = contrast_direction(proxy_n.detach(), proxy_a.detach())
        cos_main = float(torch.sum(d_main * d_base).detach().cpu())
        cos_proxy = float(torch.sum(d_proxy * d_main).detach().cpu())
        disp_epoch = 0.5 * (torch.norm(main_n.detach() - prev_n) + torch.norm(main_a.detach() - prev_a)).item()
        prev_n = main_n.detach().clone()
        prev_a = main_a.detach().clone()
        row = {
            "source": source,
            "variant": "E2_ProxyMainConsolidation",
            "epoch": epoch,
            "beta": BETA,
            "real_source_loss": finite_mean([v[0] for v in real_vals]),
            "real_state_loss": finite_mean([v[1] for v in real_vals]),
            "real_ntr_loss": finite_mean([v[2] for v in real_vals]),
            "synthetic_loss": finite_mean([v[0] for v in synth_vals]),
            "synthetic_state_loss": finite_mean([v[1] for v in synth_vals]),
            "synthetic_ntr_loss": finite_mean([v[2] for v in synth_vals]),
            "anchor_loss": finite_mean([v[0] for v in anchor_vals]),
            "cos_d_main_t_d_E0": cos_main,
            "state_direction_angle_vs_E0": float(math.degrees(math.acos(max(-1.0, min(1.0, cos_main))))),
            "cos_d_proxy_d_main": cos_proxy,
            "main_sN_norm": float(torch.norm(main_n.detach()).cpu()),
            "main_sA_norm": float(torch.norm(main_a.detach()).cpu()),
            "proxy_sN_norm": float(torch.norm(proxy_n.detach()).cpu()),
            "proxy_sA_norm": float(torch.norm(proxy_a.detach()).cpu()),
            "consolidation_state_displacement": displacement,
            "epoch_state_displacement": disp_epoch,
        }
        trajectory.append(row)
        print(f"[SSEC] {source} epoch {epoch}/{EPOCHS} real={row['real_source_loss']:.4f} synth={row['synthetic_loss']:.4f} cosE0={cos_main:.4f}", flush=True)
    payload = {"s_n": main_n.detach().cpu().numpy().astype(np.float32), "s_a": main_a.detach().cpu().numpy().astype(np.float32), "trajectory": trajectory}
    torch.save(payload, ckpt)
    return payload["s_n"], payload["s_a"], trajectory, False


@torch.no_grad()
def score_z(model_info, z_grid, states, device):
    s_n, s_a = states
    s_n_t = torch.tensor(s_n, dtype=torch.float32, device=device)
    s_a_t = torch.tensor(s_a, dtype=torch.float32, device=device)
    logits = two_state_logits(z_grid.reshape(-1, z_grid.shape[-1]), s_n_t, s_a_t)
    return logits.detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)


def evaluate_e2(model_infos, states_by_source, clip_model, preprocess, pairs, device, smoke):
    raw, cats = [], []
    for source, target in pairs:
        info = model_infos[source]
        cells, _ = protocol_cells(source, device, smoke)
        match = [c for c in cells if c[1] == target][0]
        _cid, _target, dataset, refs, patch, kind = match
        if smoke:
            refs = refs[: min(24, len(refs))]
            patch = patch[: min(24, len(patch))]
        maps = {"E2_ProxyMainConsolidation": []}
        metas = []
        states = states_by_source[source]
        for pos, ref in enumerate(refs):
            if pos % 25 == 0:
                print(f"[SSEC] eval {source}->{target} image {pos}/{len(refs)}", flush=True)
            item = item_at(dataset, kind, ref, pos)
            cat, label, path = item_meta(item, kind, ref, target)
            img = Image.open(path).convert("RGB")
            global_z = scorer_z_grid(info, patch[pos], device)
            local_z = crop_z_grids(info, clip_model, preprocess, img, device)
            g37 = score_z(info, global_z, states, device)
            l74 = stitch_quadrant74([score_z(info, z, states, device) for z in local_z])
            g74 = torch_resize(g37, (SIDE * 2, SIDE * 2), align_corners=True)
            maps["E2_ProxyMainConsolidation"].append((0.5 * g74 + 0.5 * l74).astype(np.float32))
            metas.append({"category": cat, "label": int(label), "mask": mask_np(item), "img_path": path})
            img.close()
        rr, cc = evaluate_maps(source, target, refs, dataset, kind, maps, metas, variants=("E2_ProxyMainConsolidation",))
        raw.extend(rr)
        cats.extend(cc)
        write_csv(OUT / "MICRO_TRAIN_METRICS.partial.csv", raw)
        write_csv(OUT / "PER_CATEGORY_FULL_RES.partial.csv", cats)
    return raw, cats


def scee_reference_rows(pairs):
    raw = []
    by_variant = {"E0_OriginalSource": "E0_Baseline", "E2_CoverageBiasedTransplant": "E1_DirectSCEE"}
    rows = read_csv(SCEE_OUT / "MICRO_TRAIN_METRICS.csv")
    for r in rows:
        if (r.get("source"), r.get("dataset")) not in pairs or r.get("variant") not in by_variant:
            continue
        out = dict(r)
        out["variant"] = by_variant[r["variant"]]
        raw.append(out)
    return raw


def delta_rows(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in sorted({(r["source"], r["dataset"]) for r in raw}):
        base = by.get((source, target, "E0_Baseline"))
        if not base:
            continue
        for variant in ("E1_DirectSCEE", "E2_ProxyMainConsolidation"):
            cur = by.get((source, target, variant))
            if not cur:
                continue
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(cur, "M_FR_P_AUROC") - fnum(base, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": fnum(cur, "M_FR_P_AP") - fnum(base, "M_FR_P_AP"),
                "Delta_M_FR_P_F1max": fnum(cur, "M_FR_P_F1max") - fnum(base, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(cur, "M_FR_PRO") - fnum(base, "M_FR_PRO"),
            })
    for variant in ("E1_DirectSCEE", "E2_ProxyMainConsolidation"):
        vals = [fnum(r, "Delta_M_FR_P_AP") for r in rows if r["variant"] == variant]
        rows.append({"source": "SUMMARY", "dataset": "PILOT", "variant": variant, "mean_delta_M_FR_P_AP": finite_mean(vals), "median_delta_M_FR_P_AP": finite_median(vals), "improved": int(sum(v > 0 for v in vals if np.isfinite(v)))})
    return rows


def state_geometry(model_infos, states_by_source):
    rows = []
    scee_geom = {(r["source"], r["variant"]): r for r in read_csv(SCEE_OUT / "STATE_GEOMETRY.csv")}
    for source, info in model_infos.items():
        s0n, s0a = model_states(info)
        d0 = unit_np(F.normalize(torch.tensor(s0a), dim=-1).numpy() - F.normalize(torch.tensor(s0n), dim=-1).numpy())
        for variant in ("E0_Baseline", "E1_DirectSCEE", "E2_ProxyMainConsolidation"):
            if variant == "E0_Baseline":
                s_n, s_a = s0n, s0a
            elif variant == "E1_DirectSCEE":
                old = scee_geom.get((source, "E2_CoverageBiasedTransplant"), {})
                rows.append({
                    "source": source,
                    "variant": variant,
                    "cos_d_baseline": fnum(old, "cos_d_baseline"),
                    "angle_deg": fnum(old, "angle_deg"),
                    "state_direction_norm": fnum(old, "state_direction_norm"),
                    "state_alpha": float(info["alpha"]),
                    "source": source,
                })
                continue
            else:
                s_n, s_a = states_by_source[source]
            dn = F.normalize(torch.tensor(s_n), dim=-1).numpy()
            da = F.normalize(torch.tensor(s_a), dim=-1).numpy()
            d = unit_np(da - dn)
            cos = float(np.dot(d0, d))
            rows.append({
                "source": source,
                "variant": variant,
                "cos_d_baseline": cos,
                "angle_deg": float(math.degrees(math.acos(max(-1.0, min(1.0, cos))))),
                "sN_norm": float(np.linalg.norm(s_n)),
                "sA_norm": float(np.linalg.norm(s_a)),
                "state_alpha": float(info["alpha"]),
            })
    return rows


def decide(deltas, geom):
    e2 = [r for r in deltas if r.get("variant") == "E2_ProxyMainConsolidation" and r.get("source") != "SUMMARY"]
    if not e2:
        return "STATE_SPACE_EXPERIENCE_CONSOLIDATION_HAS_NO_COMPETITIVE_VALUE", {"reason": "missing E2 metrics"}
    vals = [fnum(r, "Delta_M_FR_P_AP") for r in e2]
    mean_delta = finite_mean(vals)
    one_big = any(v >= 1.0 for v in vals if np.isfinite(v))
    other_ok = all(v >= -0.2 for v in vals if np.isfinite(v))
    improved = sum(v > 0 for v in vals if np.isfinite(v))
    geo = {(r["source"], r["variant"]): r for r in geom}
    drift_ok = True
    for source in ("Full MVTec", "Full VisA"):
        e1 = fnum(geo.get((source, "E1_DirectSCEE"), {}), "angle_deg")
        e2a = fnum(geo.get((source, "E2_ProxyMainConsolidation"), {}), "angle_deg")
        if np.isfinite(e1) and np.isfinite(e2a) and not (e2a < e1):
            drift_ok = False
    stats = {"mean_delta": mean_delta, "improved": improved, "one_transfer_ge_1": one_big, "all_transfers_ge_neg_0_2": other_ok, "drift_reduced_vs_E1": drift_ok}
    if mean_delta >= 1.0 and one_big and other_ok and drift_ok:
        return "EXPERIENCE_CONSOLIDATION_REDUCES_STATE_DRIFT_AND_IMPROVES_TRANSFER", stats
    if drift_ok and mean_delta < 1.0:
        return "CONSOLIDATION_REDUCES_DRIFT_BUT_NOT_TRANSFER", stats
    if not drift_ok and mean_delta < 1.0:
        return "DIRECT_SYNTHETIC_TRAINING_IS_NOT_THE_MAIN_CAUSE_OF_NEGATIVE_TRANSFER", stats
    return "STATE_SPACE_EXPERIENCE_CONSOLIDATION_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(outcome, decision, raw, deltas, geom, trajectory, checks, runtime):
    lines = [
        "# Phase SSEC: State-Space Experience Consolidation",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + Fixed LRP. E1 is read from SCEE-Lite Coverage-Biased direct joint training; E2 trains a proxy state on saved SCEE synthetic features and slowly consolidates into the main state with beta=0.05.",
        "",
        "## Checksums",
        "",
        f"- CLIP parameter checksum unchanged: `{checks.get('clip_parameter_checksum_unchanged')}`",
        f"- Runtime buffer note: `{checks.get('runtime_buffer_note', '')}`",
        f"- Feature cache checksum: `{checks.get('feature_cache_checksum', '')[:16]}`",
        "",
        "## Pilot Metrics",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in raw:
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
    lines += ["", "## Delta P-AP vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
    for r in deltas:
        if r.get("source") == "SUMMARY":
            continue
        lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
    lines += ["", "## State Geometry", "", "| Source | Variant | cos(d,E0) | angle deg | sN norm | sA norm |", "|---|---|---:|---:|---:|---:|"]
    for r in geom:
        lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_baseline'):.4f} | {fnum(r,'angle_deg'):.4f} | {fnum(r,'sN_norm'):.4f} | {fnum(r,'sA_norm'):.4f} |")
    lines += [
        "",
        "## Decision",
        "",
        f"- E2 mean Delta P-AP: `{decision.get('mean_delta', float('nan')):+.4f}`",
        f"- E2 improved transfers: `{decision.get('improved', 0)}/2`",
        f"- At least one transfer >= +1.0: `{decision.get('one_transfer_ge_1')}`",
        f"- Other transfer >= -0.2: `{decision.get('all_transfers_ge_neg_0_2')}`",
        f"- E2 drift smaller than E1: `{decision.get('drift_reduced_vs_E1')}`",
        "",
        f"Runtime: {runtime:.1f}s.",
    ]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SSEC] device={device}", flush=True)
    preprocess, clip_model = load_runtime(device)
    before_hash = clip_parameter_checksum(clip_model)
    infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    real_groups = {
        "Full VisA": unpack_groups(load_visa_groups(infos["Full VisA"], device, args.smoke)),
        "Full MVTec": unpack_groups(load_mvtec_groups(args.smoke)),
    }
    packs = {s: encode_saved_synthetic(s, infos[s], clip_model, preprocess, device, args.smoke) for s in infos}
    synth_groups = {s: synthetic_groups_from_pack(real_groups[s], packs[s]) for s in infos}
    cache_hash = hashlib.sha256()
    for s in sorted(packs):
        for cat, z, y, _mode in packs[s]:
            cache_hash.update(s.encode("utf-8"))
            cache_hash.update(cat.encode("utf-8"))
            cache_hash.update(np.asarray(z, dtype=np.float32).tobytes())
            cache_hash.update(np.asarray(y, dtype=np.float32).tobytes())

    states, traj = {}, []
    for source in infos:
        s_n, s_a, rows, _cached = train_ssec(source, infos[source], real_groups[source], synth_groups[source], device, args.smoke)
        states[source] = (s_n, s_a)
        traj.extend(rows)
    write_csv(OUT / "STATE_TRAJECTORY.csv", traj)

    raw = scee_reference_rows(set(HARD_PAIRS))
    e2_raw, e2_cat = evaluate_e2(infos, states, clip_model, preprocess, HARD_PAIRS, device, args.smoke)
    raw.extend(e2_raw)
    write_csv(OUT / "MICRO_TRAIN_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", e2_cat)
    deltas = delta_rows(raw)
    geom = state_geometry(infos, states)
    write_csv(OUT / "MICRO_DELTA_SUMMARY.csv", deltas)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom)
    after_hash = clip_parameter_checksum(clip_model)
    checks = {
        "clip_parameter_checksum_before": before_hash,
        "clip_parameter_checksum_after": after_hash,
        "clip_parameter_checksum_unchanged": before_hash == after_hash,
        "runtime_buffer_note": "visual.positional_embedding may be resized by the existing 518px forward path; CLIP trainable parameters are compared separately.",
        "feature_cache_checksum": cache_hash.hexdigest(),
    }
    write_json(OUT / "CHECKSUMS.json", checks)
    outcome, decision = decide(deltas, geom)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision": decision, "checksums": checks, "runtime_sec": runtime, "beta": BETA})
    write_report(outcome, decision, raw, deltas, geom, traj, checks, runtime)
    print(f"[SSEC] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
