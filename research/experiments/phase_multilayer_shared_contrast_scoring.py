import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_cross_layer_persistent_hard_normal import MOECLIP, VISUALAD
from experiments.phase_hard_normal_contrast import PC1_DEBIAS_OUT, json_default, load_payload
from experiments.phase_liv_memory_support_modes import external_cell
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np
from models.mlp import average_neighbor
from phase_xa.official_task_forward import normalized_patch_feature, visual_outputs


OUT = Path("results/phase_multilayer_shared_contrast_scoring/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
LAYERS = [6, 12, 18, 24]
DEEP_LAYERS = [18, 24]
IMAGE_BATCH = 1


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def coerce_row(row, variant):
    out = dict(row)
    out["variant"] = variant
    for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
        try:
            out[key] = float(out[key])
        except Exception:
            out[key] = float("nan")
    if "mask_images" in out:
        out["mask_images"] = int(float(out["mask_images"]))
    return out


def load_b0_rows():
    raw, percat = [], []
    for row in read_csv(HN_OUT / "RAW_METRICS.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            raw.append(coerce_row(row, "B0_FinalLayerHardNormal"))
    for row in read_csv(HN_OUT / "PER_CATEGORY.csv"):
        if row.get("variant") == "E1_HardNormalContrast":
            percat.append(coerce_row(row, "B0_FinalLayerHardNormal"))
    return raw, percat


def load_hardnormal_model(device):
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    vecs = np.load(HN_OUT / "contrast_vectors.npz")
    v_hn = np.asarray(vecs["E1"], dtype=np.float32)
    v_hn = v_hn / max(float(np.linalg.norm(v_hn)), 1e-12)
    model = make_model(hybrid_states(m_c, alpha * v_hn), corrupt_payload, device).eval()
    return model, q_pc1, alpha, v_hn


def snapshot_patch(outputs_pack, layer):
    outputs = outputs_pack["outputs"]
    if layer == 24:
        return normalized_patch_feature(outputs).float()
    snap = outputs_pack["snapshots"][f"true_block{layer}"]
    branch = snap[0] if isinstance(snap, list) else snap
    patch = branch.permute(1, 0, 2)[:, 1:, :].float()
    patch = average_neighbor(patch)
    return F.normalize(patch, dim=-1)


@torch.no_grad()
def score_layer_patch(patch, scorer, q_pc1, device):
    x = patch.detach().float().reshape(-1, patch.shape[-1])
    q = torch.tensor(q_pc1, dtype=torch.float32, device=device)
    x = x - (x @ q).unsqueeze(-1) * q
    x = F.normalize(x, dim=-1)
    logits, _ = scorer.logits(x)
    return torch.sigmoid(logits).detach().cpu().numpy().reshape(-1, SIDE, SIDE).astype(np.float32)


def score_image_layers(official_model, scorer, q_pc1, item, device):
    image = item["img"].unsqueeze(0).to(device)
    pack = visual_outputs(official_model, image, adapter=None, grad=False, snapshot_mode="true_clone")
    out = {}
    for layer in LAYERS:
        out[layer] = score_layer_patch(snapshot_patch(pack, layer), scorer, q_pc1, device)[0]
    return out


def source_calibration(smoke, device, official_model, scorer, q_pc1):
    suffix = "smoke" if smoke else "formal"
    path = OUT / f"SOURCE_LAYER_CALIBRATION_{suffix}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    _cell_id, _target, dataset, idx_or_rows, _patch, kind = target_cells(smoke)[0]
    if kind != "source_dataset":
        raise RuntimeError("MVTec target cell contract changed.")
    # For Full VisA source-only calibration, reuse the source data order from HardNormal's VisA-source side.
    from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy

    _name, src_dataset, src_indices, _cls, _patch_src, _y_img, _cat_img = load_full_visa_source_lazy(False)
    indices = list(src_indices)
    if smoke:
        indices = indices[:48]
    vals = {layer: [] for layer in LAYERS}
    start = time.time()
    for n, idx in enumerate(indices, start=1):
        item = src_dataset[int(idx)]
        maps = score_image_layers(official_model, scorer, q_pc1, item, device)
        for layer in LAYERS:
            vals[layer].append(maps[layer].reshape(-1))
        if n % 50 == 0:
            print(f"[MLScore] source calibration {n}/{len(indices)}", flush=True)
    rows = []
    data = {"seconds": time.time() - start, "target_information_used": False, "layers": LAYERS, "rows": rows}
    for layer in LAYERS:
        arr = np.concatenate(vals[layer]).astype(np.float64)
        rows.append({
            "layer": layer,
            "mean": float(arr.mean()),
            "std": float(arr.std() + 1e-8),
            "patch_count": int(arr.size),
        })
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")
    write_csv(OUT / "LAYER_SOURCE_CALIBRATION.csv", rows)
    return data


def layer_norm_params(calibration):
    return {int(r["layer"]): (float(r["mean"]), float(r["std"])) for r in calibration["rows"]}


def target_cell_image_refs(smoke):
    cells = target_cells(smoke)
    return [c for c in cells if c[0] in ("MVTec", "BTAD", "CVC-ClinicDB")]


def item_for_cell(dataset, idx_or_rows, kind, pos):
    if kind == "source_dataset":
        return dataset[int(idx_or_rows[pos])]
    return dataset[pos]


def score_target_cell(cell_id, target, dataset, idx_or_rows, kind, official_model, scorer, q_pc1, calibration, smoke):
    suffix = "smoke" if smoke else "formal"
    cache = OUT / f"{cell_id}_LAYER_SCORE_MAPS_{suffix}.npz"
    n = min(24, len(idx_or_rows)) if smoke else len(idx_or_rows)
    if cache.exists():
        data = np.load(cache)
        maps_by_layer = {layer: data[f"layer{layer}"].astype(np.float32) for layer in LAYERS}
    else:
        maps_by_layer = {layer: np.zeros((n, SIDE, SIDE), dtype=np.float16) for layer in LAYERS}
        for pos in range(n):
            item = item_for_cell(dataset, idx_or_rows, kind, pos)
            maps = score_image_layers(official_model, scorer, q_pc1, item, device=next(scorer.parameters()).device)
            for layer in LAYERS:
                maps_by_layer[layer][pos] = maps[layer].astype(np.float16)
            if (pos + 1) % 50 == 0:
                print(f"[MLScore] target {cell_id} images {pos+1}/{n}", flush=True)
        np.savez_compressed(cache, **{f"layer{layer}": maps_by_layer[layer] for layer in LAYERS})
        maps_by_layer = {layer: maps_by_layer[layer].astype(np.float32) for layer in LAYERS}
    params = layer_norm_params(calibration)
    z = {}
    for layer in LAYERS:
        mu, sd = params[layer]
        z[layer] = (maps_by_layer[layer] - mu) / sd
    fused = {
        "E1_MultiLayerSharedContrast": np.mean([z[layer] for layer in LAYERS], axis=0).astype(np.float32),
        "E2_DeepLayerControl": np.mean([z[layer] for layer in DEEP_LAYERS], axis=0).astype(np.float32),
    }
    for layer in LAYERS:
        fused[f"D_layer{layer}_single"] = z[layer].astype(np.float32)
    idx_use = idx_or_rows[:n]
    raw, percat = [], []
    corr_rows = layer_correlations(cell_id, z)
    for variant, maps in fused.items():
        image = np.asarray([top1_image_score(m) for m in maps], dtype=np.float32)
        metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
        metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
        raw.append(metric)
        for cat, vals in per.items():
            row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
            row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
            percat.append(row)
    return raw, percat, corr_rows


def layer_correlations(cell_id, z):
    rows = []
    flat = {layer: z[layer].reshape(-1).astype(np.float32) for layer in LAYERS}
    for i, a in enumerate(LAYERS):
        for b in LAYERS[i + 1 :]:
            pa = flat[a]
            pb = flat[b]
            if pa.size > 200000:
                ids = np.linspace(0, pa.size - 1, 200000).astype(np.int64)
                pa = pa[ids]
                pb = pb[ids]
            rows.append({
                "dataset": cell_id,
                "layer_a": a,
                "layer_b": b,
                "pearson": float(pearsonr(pa, pb).statistic),
                "spearman": float(spearmanr(pa, pb).statistic),
            })
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_FinalLayerHardNormal")]
        e1 = by[(ds, "E1_MultiLayerSharedContrast")]
        e2 = by[(ds, "E2_DeepLayerControl")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    mean_gain = float(e1_b0.mean())
    improved = int(np.sum(e1_b0 > 0))
    mt_or_cvc = bool(e1_b0[0] >= 1.5 or e1_b0[2] >= 1.5)
    btad_ok = bool(e1_b0[1] >= -1.0)
    if improved >= 2 and mean_gain >= 1.0 and mt_or_cvc and btad_ok:
        outcome = "MULTILAYER_SHARED_CONTRAST_SUPPORTED"
    elif e1_b0[1] > 0 and e1_b0[0] <= 0 and e1_b0[2] <= 0:
        outcome = "MULTILAYER_SCORING_DOMAIN_DEPENDENT"
    else:
        outcome = "MULTILAYER_SCORE_FUSION_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": mean_gain,
        "E1_targets_improved_over_B0": improved,
        "MVTec_or_ClinicDB_gain_ge_1p5": mt_or_cvc,
        "BTAD_drop_not_below_1pp": btad_ok,
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_MultiLayerSharedContrast"}
    for ref_name, ref_map in (("MoECLIP pixel", MOECLIP), ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD)):
        for ds, ref in ref_map.items():
            row = current[ds]
            p_auroc = float(row["P-AUROC"])
            p_ap = float(row["P-AP"])
            g = float(row["G_pixel"])
            rows.append({
                "dataset": ds,
                "current_method": "E1_MultiLayerSharedContrast",
                "current_P-AUROC": p_auroc,
                "current_P-AP": p_ap,
                "current_G": g,
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": p_auroc - ref["P-AUROC"],
                "Delta_P_AP": p_ap - ref["P-AP"],
                "Delta_G": g - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS",
            })
    write_csv(OUT / "SOTA_GAP.csv", rows)
    return rows


def write_report(summary_rows, outcome, stats, sota_rows, runtime):
    lines = [
        "# Multi-Layer Shared-Contrast Scoring Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. HardNormal is not retrained. Layers 6/12/18/24 use raw 1024-d frozen CLIP patch snapshots, the same PC1 removal, the same rank4 alignment/scorer, and source-only score mean/std calibration.",
        "",
        "| Dataset | B0 Final-Layer HN G | E1 Multi-Layer G | E2 18/24 G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"E1 improved targets: {stats['E1_targets_improved_over_B0']}/3.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is treated as the primary direct reference requested for this gate; VisualAD(CLIP) is retained as the secondary direct reference. Comparability is limited to VisA-source pixel P-AUROC/P-AP/G; exact implementation and preprocessing should still be reproduced before leaderboard claims.",
        "",
        "| Reference | Dataset | Current G | Reference G | Delta G | Status |",
        "|---|---|---:|---:|---:|---|",
    ])
    for r in sota_rows:
        lines.append(f"| {r['reference']} | {r['dataset']} | {float(r['current_G']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "Stop-loss: if this round does not improve mean G by >= +1pp over Final-Layer HardNormal and does not shrink the MVTec/ClinicDB SOTA gap, do not add learned layer weights or attention fusion.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[MLScore] device={device}", flush=True)
    scorer, q_pc1, _alpha, _v = load_hardnormal_model(device)
    official_model, *_ = load_components("checkpoints/test_on_mvtec.pth", device)
    calibration = source_calibration(args.smoke, device, official_model, scorer, q_pc1)
    raw, percat, corr = [], [], []
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw.extend(b0_raw)
    percat.extend(b0_percat)
    for cell_id, target, dataset, idx_or_rows, _patch, kind in target_cell_image_refs(args.smoke):
        print(f"[MLScore] score target {cell_id}", flush=True)
        r, p, c = score_target_cell(cell_id, target, dataset, idx_or_rows, kind, official_model, scorer, q_pc1, calibration, args.smoke)
        raw.extend(r)
        percat.extend(p)
        corr.extend(c)
    summary_rows = summarize(raw)
    outcome, stats = decide(summary_rows)
    sota_rows = write_sota_gap(raw)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "LAYER_SCORE_CORRELATION.csv", corr)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": time.time() - start,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "hardnormal_retrained": False,
        "layers": LAYERS,
        "deep_control_layers": DEEP_LAYERS,
    })
    write_report(summary_rows, outcome, stats, sota_rows, time.time() - start)
    print(f"[MLScore] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
