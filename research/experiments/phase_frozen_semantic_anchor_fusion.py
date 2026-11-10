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

import AnomalyCLIP_lib
from experiments.mrad_audit_utils import tokenize_texts
from experiments.phase_private_residual_contrast_memory import unit_np
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np


OUT = Path("results/phase_frozen_semantic_anchor_fusion/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPS = 1e-6
CHUNK = 128

NORMAL_PROMPTS = [
    "normal",
    "flawless",
    "intact",
    "defect-free",
    "undamaged",
    "a normal object",
    "an intact object",
]
ANOMALY_PROMPTS = [
    "anomalous",
    "defective",
    "damaged",
    "abnormal",
    "corrupted",
    "a defective object",
    "an anomalous object",
]

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


def load_hardnormal_model(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    v_hn = unit_np(np.load(HN_OUT / "contrast_vectors.npz")["E1"].astype(np.float32))
    model = make_model(hybrid_states(m_c, alpha * v_hn), corrupt_payload, device).eval()
    return q_pc1, model


@torch.no_grad()
def encode_text_anchors_and_visual_projection(device):
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=None, download_root=str(ROOT / ".cache" / "clip"))
    model.eval()
    prompts = NORMAL_PROMPTS + ANOMALY_PROMPTS
    tokens = tokenize_texts(prompts).to(device)
    text = F.normalize(model.encode_text(tokens).float(), dim=-1)
    normal = F.normalize(text[: len(NORMAL_PROMPTS)].mean(dim=0), dim=-1)
    anomaly = F.normalize(text[len(NORMAL_PROMPTS) :].mean(dim=0), dim=-1)
    anchors = torch.stack([normal, anomaly], dim=0).detach().cpu().numpy().astype(np.float32)
    visual_proj = model.visual.proj.detach().float().cpu().numpy().astype(np.float32)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return anchors, visual_proj


def random_anchors(dim):
    rng = np.random.default_rng(0)
    r = rng.standard_normal((2, dim)).astype(np.float32)
    return unit_np(r)


@torch.no_grad()
def score_arrays(raw_z, state_model, anchors, visual_proj, q_pc1, device):
    raw_orig = norm_np(np.asarray(raw_z, dtype=np.float32).reshape(-1, raw_z.shape[-1]))
    raw_state = remove_pc1(np.asarray(raw_z, dtype=np.float32), q_pc1)
    raw_state = norm_np(raw_state.reshape(-1, raw_state.shape[-1]))
    x = torch.tensor(raw_state, dtype=torch.float32, device=device)
    logits, _ = state_model.logits(x)
    s_state = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
    a = torch.tensor(anchors, dtype=torch.float32, device=device)
    vp = torch.tensor(visual_proj, dtype=torch.float32, device=device)
    xt = F.normalize(torch.tensor(raw_orig, dtype=torch.float32, device=device) @ vp, dim=-1)
    l = xt @ F.normalize(a, dim=-1).T
    ex = torch.softmax(l, dim=-1)[:, 1].detach().cpu().numpy().astype(np.float32)
    return s_state, ex


def compute_source_stats(state_model, anchors_text, anchors_random, visual_proj, q_pc1, device, smoke):
    cache = OUT / ("source_calibration_smoke.npz" if smoke else "source_calibration.npz")
    if cache.exists():
        data = np.load(cache)
        return {k: float(data[k]) for k in data.files}
    _name, _dataset, _indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    n = min(48, patch.shape[0]) if smoke else patch.shape[0]
    vals = {"state": [], "text": [], "random": []}
    for i in range(n):
        ss, st = score_arrays(patch[i], state_model, anchors_text, visual_proj, q_pc1, device)
        _ss, sr = score_arrays(patch[i], state_model, anchors_random, visual_proj, q_pc1, device)
        vals["state"].append(ss)
        vals["text"].append(st)
        vals["random"].append(sr)
    out = {}
    for key, parts in vals.items():
        arr = np.concatenate(parts).astype(np.float32)
        out[f"{key}_mean"] = float(arr.mean())
        out[f"{key}_std"] = float(max(arr.std(), EPS))
    np.savez_compressed(cache, **out)
    return out


def zscore(x, mean, std):
    return ((np.asarray(x, dtype=np.float32) - float(mean)) / max(float(std), EPS)).astype(np.float32)


def corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size < 2 or np.std(a) < EPS or np.std(b) < EPS:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def resize_binary_mask(item):
    from experiments.phase_xxxviii_visual_state_representation import patch_labels

    return patch_labels(item).astype(np.uint8)


def target_item(dataset, ref, kind, pos):
    return dataset[int(ref)] if kind == "source_dataset" else dataset[pos]


@torch.no_grad()
def evaluate_fusion(state_model, anchors_text, anchors_random, visual_proj, q_pc1, stats, device, smoke):
    raw, percat, diag_rows, dist_rows = [], [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind in target_cells(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        maps = {
            "E1_FrozenSemanticAnchorFusion": np.zeros((n, SIDE, SIDE), dtype=np.float16),
            "E2_RandomSemanticDirectionControl": np.zeros((n, SIDE, SIDE), dtype=np.float16),
            "D1_TextOnly": np.zeros((n, SIDE, SIDE), dtype=np.float16),
            "D2_RandomOnly": np.zeros((n, SIDE, SIDE), dtype=np.float16),
        }
        images = {k: np.zeros((n,), dtype=np.float32) for k in maps}
        all_state, all_text, all_random, all_gt = [], [], [], []
        for start in range(0, n, CHUNK):
            end = min(n, start + CHUNK)
            ss_list, st_list, sr_list = [], [], []
            for i in range(start, end):
                ss, st = score_arrays(patch[i], state_model, anchors_text, visual_proj, q_pc1, device)
                _ss, sr = score_arrays(patch[i], state_model, anchors_random, visual_proj, q_pc1, device)
                ss_list.append(ss)
                st_list.append(st)
                sr_list.append(sr)
            s_state = np.stack(ss_list, axis=0).reshape(end - start, SIDE, SIDE)
            s_text = np.stack(st_list, axis=0).reshape(end - start, SIDE, SIDE)
            s_random = np.stack(sr_list, axis=0).reshape(end - start, SIDE, SIDE)
            z_state = zscore(s_state, stats["state_mean"], stats["state_std"])
            z_text = zscore(s_text, stats["text_mean"], stats["text_std"])
            z_random = zscore(s_random, stats["random_mean"], stats["random_std"])
            fused = 0.5 * z_state + 0.5 * z_text
            fused_random = 0.5 * z_state + 0.5 * z_random
            block = {
                "E1_FrozenSemanticAnchorFusion": fused,
                "E2_RandomSemanticDirectionControl": fused_random,
                "D1_TextOnly": z_text,
                "D2_RandomOnly": z_random,
            }
            for name, arr in block.items():
                maps[name][start:end] = arr.astype(np.float16)
                images[name][start:end] = np.asarray([top1_image_score(x) for x in arr], dtype=np.float32)
            all_state.append(s_state.reshape(-1))
            all_text.append(s_text.reshape(-1))
            all_random.append(s_random.reshape(-1))
            gt_parts = []
            for local, ref in enumerate(idx_use[start:end], start=start):
                item = target_item(dataset, ref, kind, local)
                gt_parts.append(resize_binary_mask(item))
            all_gt.append(np.stack(gt_parts, axis=0).reshape(-1))
        state_flat = np.concatenate(all_state)
        text_flat = np.concatenate(all_text)
        random_flat = np.concatenate(all_random)
        gt_flat = np.concatenate(all_gt).astype(bool)
        diag_rows.extend([
            {"dataset": cell_id, "stat": "corr_state_text", "value": corr(state_flat, text_flat)},
            {"dataset": cell_id, "stat": "corr_state_random", "value": corr(state_flat, random_flat)},
        ])
        for name, vals in (("S_text", text_flat), ("S_random", random_flat), ("S_state", state_flat)):
            dist_rows.append({
                "dataset": cell_id,
                "score": name,
                "normal_mean": float(vals[~gt_flat].mean()),
                "normal_std": float(vals[~gt_flat].std()),
                "anomaly_mean": float(vals[gt_flat].mean()) if np.any(gt_flat) else float("nan"),
                "anomaly_std": float(vals[gt_flat].std()) if np.any(gt_flat) else float("nan"),
            })
        for variant in maps:
            print(f"[SemanticAnchor] eval {variant} on {cell_id}", flush=True)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, images[variant], maps[variant])
            metric.update({"dataset": cell_id, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            for cat, vals in per.items():
                row = {"dataset": cell_id, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat, diag_rows, dist_rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormal")]
        e1 = by[(ds, "E1_FrozenSemanticAnchorFusion")]
        e2 = by[(ds, "E2_RandomSemanticDirectionControl")]
        text = by[(ds, "D1_TextOnly")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "TextOnly_G": float(text["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
            "state_only_P-AP": float(b0["P-AP"]),
            "fused_P-AP": float(e1["P-AP"]),
        })
    return rows


def decide(summary):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary], dtype=np.float32)
    mt_or_clinic = max([r["E1_minus_B0_G"] for r in summary if r["dataset"] in ("MVTec", "CVC-ClinicDB")])
    btad = [r["E1_minus_B0_G"] for r in summary if r["dataset"] == "BTAD"][0]
    if int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 0.5 and mt_or_clinic >= 1.5 and btad >= -1.0:
        outcome = "FROZEN_SEMANTIC_ANCHOR_SUPPORTED"
    elif float(e1_e2.mean()) > 0.5 and float(e1_b0.mean()) < 1.0:
        outcome = "TEXT_SEMANTIC_PRIOR_HAS_WEAK_SIGNAL"
    else:
        outcome = "FROZEN_SEMANTIC_ANCHOR_NO_SIGNAL"
    return outcome, {
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "max_MVTec_or_ClinicDB_gain": float(mt_or_clinic),
        "BTAD_gain": float(btad),
    }


def write_sota_gap(raw):
    rows = []
    current = {r["dataset"]: r for r in raw if r["variant"] == "E1_FrozenSemanticAnchorFusion"}
    for ds, row in current.items():
        for ref_name, table, status in (
            ("MoECLIP OpenCLIP ViT-L/14-336 518x518", MOECLIP, "DIRECT_COMPARABLE_WITH_CAUTION_VISASOURCE_PIXEL_METRICS"),
            ("CVPR 2026 VisualAD(CLIP) pixel", VISUALAD, "REFERENCE_ONLY_OR_PROTOCOL_CAUTION"),
        ):
            ref = table[ds]
            rows.append({
                "dataset": ds,
                "current_method": "E1_FrozenSemanticAnchorFusion",
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


def write_report(summary, outcome, stats, sota, runtime):
    lines = [
        "# Frozen Semantic Anchor Fusion Pilot",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB. HardNormal is reused unchanged; frozen category-agnostic normal/anomaly text anchors are fused 1:1 after source-only z-score calibration.",
        "",
        "| Dataset | B0 HardNormal G | E1 Text Fusion G | E2 Random Fusion G | TextOnly G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['TextOnly_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        f"E1 improved targets: {stats['E1_targets_improved_over_B0']}/3.",
        "",
        "## External SOTA Position",
        "",
        "MoECLIP is the primary reference when source/target/backbone/resolution match closely; VisualAD(CLIP) is retained as secondary reference only. No target statistics or prompt/category tuning were used.",
        "",
        "| Dataset | Current G | Reference | Reference G | Delta G | Status |",
        "|---|---:|---|---:|---:|---|",
    ])
    for r in sota:
        lines.append(f"| {r['dataset']} | {float(r['current_G']):.4f} | {r['reference']} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend(["", "Stop: no prompt learning, weight sweep, temperature sweep, category prompts, Memory, or target calibration is started.", f"Runtime: {runtime:.1f}s."])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SemanticAnchor] device={device}", flush=True)
    q_pc1, state_model = load_hardnormal_model(device)
    anchors_text, visual_proj = encode_text_anchors_and_visual_projection(device)
    anchors_random = random_anchors(anchors_text.shape[-1])
    stats = compute_source_stats(state_model, anchors_text, anchors_random, visual_proj, q_pc1, device, args.smoke)
    raw_new, percat_new, corr_rows, dist_rows = evaluate_fusion(state_model, anchors_text, anchors_random, visual_proj, q_pc1, stats, device, args.smoke)
    raw_b0, percat_b0 = load_b0_rows()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        percat_b0 = percat_b0[:3]
    raw = raw_b0 + raw_new
    percat = percat_b0 + percat_new
    summary = summarize(raw)
    outcome, decision_stats = decide(summary)
    sota = write_sota_gap(raw)
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary)
    write_csv(OUT / "TEXT_CORRELATION.csv", corr_rows)
    write_csv(OUT / "TEXT_SCORE_DISTRIBUTIONS.csv", dist_rows)
    write_csv(OUT / "SOURCE_CALIBRATION.csv", [{"stat": k, "value": v} for k, v in stats.items()])
    np.savez_compressed(OUT / "semantic_anchors.npz", text=anchors_text, random=anchors_random)
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
        "fusion_weight": [0.5, 0.5],
        "normal_prompts": NORMAL_PROMPTS,
        "anomaly_prompts": ANOMALY_PROMPTS,
        "text_anchor_space": "frozen CLIP 768d joint embedding; 1024d patch hidden is projected by frozen visual.proj",
    })
    write_report(summary, outcome, decision_stats, sota, runtime)
    print(f"[SemanticAnchor] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
