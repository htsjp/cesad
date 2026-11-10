import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_hard_normal_contrast import (
    TOP_NEG_FRAC,
    balanced_category_batch,
    collect_source_patch_pool,
    final_negative_distribution,
)
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_source_partition_contrast_invariance import contrast_geometry
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, PATCH_SAMPLE_PER_CLASS_CATEGORY, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR, SIDE, norm_np


OUT = Path("results/phase_context_decoupled_anomaly_replay/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
CACHE_VERSION = "patch_grid_replay_v1"

MOECLIP = {
    "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
    "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
    "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
}

VISUALAD = {
    "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
    "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
    "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
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
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def load_b0_rows():
    raw, percat = [], []
    for name, dest in (("RAW_METRICS.csv", raw), ("PER_CATEGORY.csv", percat)):
        for row in read_csv(HN_OUT / name):
            if row["variant"] != "E1_HardNormalContrast":
                continue
            out = dict(row)
            out["variant"] = "B0_HardNormal"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                if key in out:
                    try:
                        out[key] = float(out[key])
                    except ValueError:
                        out[key] = float("nan")
            if "mask_images" in out:
                out["mask_images"] = int(float(out["mask_images"]))
            dest.append(out)
    return raw, percat


def load_context(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    init_v = unit_np(d_t)
    b0_v = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32))
    align_model = make_model(states_c, corrupt_payload, device).eval()
    return q_pc1, corrupt_payload, align_model, m_c.astype(np.float32), alpha, init_v.astype(np.float32), b0_v


def transformed_image(pos, patch, align_model, q_pc1, device, cache):
    pos = int(pos)
    if pos in cache:
        return cache[pos]
    z0 = remove_pc1(np.asarray(patch[pos], dtype=np.float32), q_pc1)
    z0 = norm_np(z0.reshape(SIDE * SIDE, 1024))
    with torch.no_grad():
        z = align_model.adapt(torch.tensor(z0, dtype=torch.float32, device=device)).detach().cpu().numpy().astype(np.float32)
    cache[pos] = z
    return z


def split_source_records(dataset, indices, labels_by_pos, slots, smoke):
    by_cat = {}
    zero = np.zeros((SIDE * SIDE,), dtype=np.int64)
    max_per = 12 if smoke else None
    allowed = {int(p) for positions in slots.values() for p in positions}
    for pos, idx in enumerate(indices):
        if int(pos) not in allowed:
            continue
        cat = str(dataset.data_all[int(idx)]["cls_name"])
        y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.int64)
        bucket = by_cat.setdefault(cat, {"normal": [], "anomaly": []})
        key = "anomaly" if y.max() > 0 else "normal"
        if max_per is None or len(bucket[key]) < max_per:
            bucket[key].append({"pos": int(pos), "category": cat, "label": y.astype(np.float32)})
    return {c: v for c, v in by_cat.items() if v["normal"] and v["anomaly"]}


def append_samples(group, x, y, origin):
    if group["x"].size:
        group["x"] = np.concatenate([group["x"], x.astype(np.float32)], axis=0)
        group["y"] = np.concatenate([group["y"], y.astype(np.float32)], axis=0)
        group["origin"] = np.concatenate([group["origin"], np.asarray(origin, dtype=object)], axis=0)
    else:
        group["x"] = x.astype(np.float32)
        group["y"] = y.astype(np.float32)
        group["origin"] = np.asarray(origin, dtype=object)


def build_replay_groups(mode, base_groups, records, patch, align_model, q_pc1, device, smoke):
    rng = np.random.default_rng(SEED)
    cache = {}
    groups = {
        cat: {"x": data["x"].copy(), "y": data["y"].copy(), "origin": data["origin"].copy()}
        for cat, data in base_groups.items()
    }
    matrix = {}
    area_rows = []
    sample_rows = []
    cats = sorted(records)
    replay_count = 0
    for donor_cat in cats:
        for donor in records[donor_cat]["anomaly"]:
            if mode == "cross":
                receiver_cats = [c for c in cats if c != donor_cat and records[c]["normal"]]
            elif mode == "same":
                receiver_cats = [donor_cat]
            else:
                raise ValueError(mode)
            if not receiver_cats:
                continue
            receiver_cat = str(rng.choice(receiver_cats))
            receiver = rng.choice(records[receiver_cat]["normal"])
            donor_z = transformed_image(donor["pos"], patch, align_model, q_pc1, device, cache)
            recv_z = transformed_image(receiver["pos"], patch, align_model, q_pc1, device, cache)
            mask = donor["label"].astype(bool)
            if not np.any(mask):
                continue
            replay = recv_z.copy()
            replay[mask] = donor_z[mask]
            pos_ids = np.flatnonzero(mask)
            neg_ids = np.flatnonzero(~mask)
            pos_take = min(PATCH_SAMPLE_PER_CLASS_CATEGORY, pos_ids.size)
            neg_take = min(PATCH_SAMPLE_PER_CLASS_CATEGORY, neg_ids.size)
            pos_pick = rng.choice(pos_ids, size=pos_take, replace=pos_ids.size < pos_take)
            neg_pick = rng.choice(neg_ids, size=neg_take, replace=neg_ids.size < neg_take)
            ids = np.concatenate([pos_pick, neg_pick])
            yy = np.concatenate([np.ones(pos_take, dtype=np.float32), np.zeros(neg_take, dtype=np.float32)])
            xx = replay[ids]
            origin = [f"replay_{mode}_anomaly"] * pos_take + [f"replay_{mode}_background"] * neg_take
            append_samples(groups[receiver_cat], xx, yy, origin)
            replay_count += 1
            matrix[(donor_cat, receiver_cat)] = matrix.get((donor_cat, receiver_cat), 0) + 1
            area = float(mask.mean())
            area_rows.append({
                "variant": f"{mode}_replay",
                "donor_category": donor_cat,
                "receiver_category": receiver_cat,
                "source_area_ratio": area,
                "replay_area_ratio": area,
                "area_ratio_delta": 0.0,
            })
            sample_rows.append({
                "variant": f"{mode}_replay",
                "donor_category": donor_cat,
                "receiver_category": receiver_cat,
                "donor_pos": donor["pos"],
                "receiver_pos": receiver["pos"],
                "positive_patch_count": int(pos_ids.size),
                "area_ratio": area,
            })
    matrix_rows = [
        {"variant": f"{mode}_replay", "donor_category": d, "receiver_category": r, "count": int(v)}
        for (d, r), v in sorted(matrix.items())
    ]
    audit = [{"variant": f"{mode}_replay", "record_type": "summary", "replay_samples": replay_count, "implementation": "source frozen patch-grid anomaly content transplanted into receiver normal patch grid"}]
    for cat, data in sorted(groups.items()):
        audit.append({
            "variant": f"{mode}_replay",
            "record_type": "group_after_replay",
            "category": cat,
            "samples": int(data["x"].shape[0]),
            "positive_samples": int(np.sum(data["y"] > 0.5)),
            "negative_samples": int(np.sum(data["y"] < 0.5)),
            "replay_positive_samples": int(np.sum(np.char.startswith(data["origin"].astype(str), f"replay_{mode}_anomaly"))),
            "replay_negative_samples": int(np.sum(np.char.startswith(data["origin"].astype(str), f"replay_{mode}_background"))),
        })
    return groups, audit, matrix_rows, area_rows, sample_rows


def train_variant(variant, groups, m_c, alpha, init_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}_{CACHE_VERSION}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["stats"], True
    print(f"[Replay] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m_c, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    curves, stats = [], []
    for epoch in range(1, EPOCHS + 1):
        vals = []
        for step in range(steps_per_epoch):
            xb, yb, origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, alpha, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            neg_scores = logits.detach()[neg_idx]
            selected = neg_idx[torch.topk(neg_scores, k=k, largest=True).indices]
            hard_loss = F.softplus(logits[selected]).mean()
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sel_np = selected.detach().cpu().numpy()
            sel_origin = origin[sel_np].astype(str)
            stats.append({
                "variant": variant,
                "record_type": "train_selection",
                "epoch": epoch,
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "task_loss": float(task_loss.detach().cpu()),
                "hard_loss": float(hard_loss.detach().cpu()),
                "bce": float(bce.detach().cpu()),
                "dice": float(dice.detach().cpu()),
                "selected_count": int(len(sel_np)),
                "selected_score_mean": float(logits.detach()[selected].cpu().numpy().mean()),
                "normal_image_ratio": float(np.mean(sel_origin == "normal_image")),
                "anomaly_mask_out_ratio": float(np.mean(sel_origin == "anomaly_mask_out")),
                "replay_background_ratio": float(np.mean(np.char.find(sel_origin, "replay_") >= 0)),
            })
            vals.append([float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())])
        arr = np.asarray(vals, dtype=np.float32)
        row = {"variant": variant, "epoch": epoch, "loss": float(arr[:, 0].mean()), "task_loss": float(arr[:, 1].mean()), "hard_loss": float(arr[:, 2].mean()), "bce": float(arr[:, 3].mean()), "dice": float(arr[:, 4].mean())}
        curves.append(row)
        print(f"[Replay] {variant} epoch {epoch}/{EPOCHS} loss={row['loss']:.4f} hard={row['hard_loss']:.4f}", flush=True)
    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "stats": stats, "curves": curves}, ckpt)
    write_csv(OUT / f"{variant}_TRAINING_CURVES.csv", curves)
    return v, stats, False


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_CrossCategoryAnomalyReplay")]
        e2 = by[(ds, "E2_SameCategoryReplayControl")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary], dtype=np.float32)
    e2_b0 = np.asarray([r["E2_G"] - r["B0_G"] for r in summary], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary], dtype=np.float32)
    mt_or_clinic = max([r["E1_minus_B0_G"] for r in summary if r["dataset"] in ("MVTec", "CVC-ClinicDB")])
    btad = [r["E1_minus_B0_G"] for r in summary if r["dataset"] == "BTAD"][0]
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and mt_or_clinic >= 1.5 and btad >= -1.0:
        outcome = "CONTEXT_DECOUPLED_ANOMALY_REPLAY_SUPPORTED"
    elif int(np.sum(e1_b0 > 0)) >= 2 and int(np.sum(e2_b0 > 0)) >= 2 and abs(float(e1_e2.mean())) < 0.5:
        outcome = "GENERIC_ANOMALY_AUGMENTATION_SUPPORTED"
    elif btad > 1.0 and int(np.sum(e1_b0 > 0)) == 1:
        outcome = "ANOMALY_REPLAY_DOMAIN_DEPENDENT"
    else:
        outcome = "CONTEXT_DECOUPLED_ANOMALY_REPLAY_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "max_MVTec_or_ClinicDB_gain": float(mt_or_clinic),
        "BTAD_gain": float(btad),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_CrossCategoryAnomalyReplay"}
    for ds, row in current.items():
        for ref_name, table, status in (
            ("MoECLIP OpenCLIP ViT-L/14-336 518x518", MOECLIP, "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS"),
            ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD, "REFERENCE_ONLY_OR_PROTOCOL_CAUTION"),
        ):
            ref = table[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_CrossCategoryAnomalyReplay",
                "current_P-AUROC": float(row["P-AUROC"]),
                "current_P-AP": float(row["P-AP"]),
                "current_G": float(row["G_pixel"]),
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
                "comparison_status": status,
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary, outcome, stats, geom, sota, runtime):
    low_cos_note = ""
    if float(geom[0]["cos_v_B0_v_E1"]) < 0.5:
        low_cos_note = " E1 improves/changes with very low B0 cosine; treat it as possible source-boundary relearning."
    lines = [
        "# Context-Decoupled Anomaly Replay",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB, strict source-only. This implementation uses the existing frozen patch-feature training path: donor anomaly patch-grid content inside the source mask is transplanted into a receiver normal patch grid. No target data, prompt, Memory, adapter, diffusion, GAN, or loss sweep is used.",
        "",
        "| Dataset | B0 HardNormal G | E1 Cross Replay G | E2 Same Replay G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"cos(d_E1,d_B0): {float(geom[0]['cos_v_B0_v_E1']):.6f}.{low_cos_note}",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary reference when source/target/backbone/resolution match closely; VisualAD(CLIP) is retained as secondary reference only.",
        "",
        "| Dataset | Current G | Reference | Reference G | Delta G | Status |",
        "|---|---:|---|---:|---:|---|",
    ])
    for r in sota:
        lines.append(f"| {r['dataset']} | {float(r['current_G']):.4f} | {r['reference']} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop: no paste-count sweep, blend sweep, adapter, Memory, prompt, or target tuning is started.", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Replay] device={device}", flush=True)
    q_pc1, corrupt_payload, align_model, m_c, alpha, init_v, b0_v = load_context(device)
    _name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, args.smoke)
    base_groups, base_audit = collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)
    records = split_source_records(dataset, indices, labels_by_pos, slots, args.smoke)
    cross_groups, cross_audit, cross_matrix, cross_area, cross_samples = build_replay_groups("cross", base_groups, records, patch, align_model, q_pc1, device, args.smoke)
    same_groups, same_audit, same_matrix, same_area, same_samples = build_replay_groups("same", base_groups, records, patch, align_model, q_pc1, device, args.smoke)
    v_e1, stats_e1, reused_e1 = train_variant("E1_CrossCategoryAnomalyReplay", cross_groups, m_c, alpha, init_v, device, args.smoke)
    v_e2, stats_e2, reused_e2 = train_variant("E2_SameCategoryReplayControl", same_groups, m_c, alpha, init_v, device, args.smoke)
    models = {
        "E1_CrossCategoryAnomalyReplay": make_model(hybrid_states(m_c, alpha * v_e1), corrupt_payload, device),
        "E2_SameCategoryReplayControl": make_model(hybrid_states(m_c, alpha * v_e2), corrupt_payload, device),
    }
    raw_new, percat_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, percat_b0 = load_b0_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        percat_b0 = percat_b0[:3]
    raw = raw_b0 + raw_new
    percat = percat_b0 + percat_new
    summary = summarize(raw)
    outcome, decision_stats = decide(summary)
    geom = contrast_geometry(b0_v, v_e1, v_e2)
    sota = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary)
    write_csv(OUT / "REPLAY_AUDIT.csv", base_audit + cross_audit + same_audit + cross_samples + same_samples)
    write_csv(OUT / "DONOR_RECEIVER_MATRIX.csv", cross_matrix + same_matrix)
    write_csv(OUT / "ANOMALY_AREA_DISTRIBUTION.csv", cross_area + same_area)
    write_csv(OUT / "TRAINING_STATS.csv", stats_e1 + stats_e2)
    write_csv(OUT / "SOURCE_VALIDATION_ANOMALY_SCORE.csv", final_negative_distribution("E1_CrossCategoryAnomalyReplay", cross_groups, v_e1, m_c, alpha) + final_negative_distribution("E2_SameCategoryReplayControl", same_groups, v_e2, m_c, alpha))
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    np.savez_compressed(OUT / "contrast_vectors.npz", B0=b0_v.astype(np.float32), E1=v_e1.astype(np.float32), E2=v_e2.astype(np.float32))
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "source": "Full VisA",
        "targets": ["MVTec", "BTAD", "CVC-ClinicDB"],
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "implementation_note": "Replay is performed in the existing frozen patch-feature training space by transplanting donor anomaly mask patch features into receiver normal patch grids; no image generator or target data is used.",
        "checkpoints_reused": {"B0": True, "E1": bool(reused_e1), "E2": bool(reused_e2)},
    })
    write_report(summary, outcome, decision_stats, geom, sota, runtime)
    print(f"[Replay] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
