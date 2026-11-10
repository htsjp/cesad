import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_keyless_contrast_memory import load_slot_vectors
from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, safe_ap, safe_auc
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed, remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np

OUT = Path("results/phase_transferability_filtered_contrast_memory/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
TCM_OUT = Path("results/phase_transferable_contrast_memory_pilot/formal_corrected")


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


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def source_true_masks():
    from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps

    true_by_pos, _corrupt, _audit = source_label_maps(False)
    return true_by_pos


@torch.no_grad()
def source_slot_pixel_scores(vectors, align_model, m_shared, alpha, q_pc1, dataset, indices, patch, labels_by_pos, smoke, batch_images=8):
    names = list(vectors)
    states = np.stack([hybrid_states(m_shared, alpha * unit(vectors[n])) for n in names], axis=0).astype(np.float32)
    states_t = F.normalize(torch.tensor(states, dtype=torch.float32, device=next(align_model.parameters()).device), dim=-1)
    positions = list(range(len(indices)))
    if smoke:
        positions = positions[:96]
    by_cat = {cat: {name: {"y": [], "s": []} for name in names} for cat in sorted({str(dataset.data_all[int(indices[p])]["cls_name"]) for p in positions})}
    zero = np.zeros((SIDE * SIDE,), dtype=np.uint8)
    for start in range(0, len(positions), batch_images):
        batch_pos = positions[start : start + batch_images]
        arr = remove_pc1(np.asarray([patch[p] for p in batch_pos], dtype=np.float32), q_pc1)
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=states_t.device)
        z = align_model.adapt(x)
        logits = z @ states_t[:, 1, :].T - z @ states_t[:, 0, :].T
        scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32).reshape(len(batch_pos), SIDE * SIDE, len(names))
        for bi, pos in enumerate(batch_pos):
            cat = str(dataset.data_all[int(indices[pos])]["cls_name"])
            y = np.asarray(labels_by_pos.get(int(pos), zero), dtype=np.uint8)
            for j, name in enumerate(names):
                by_cat[cat][name]["y"].append(y)
                by_cat[cat][name]["s"].append(scores[bi, :, j])
        if start and start % 400 == 0:
            print(f"[TFContrast] source scoring {start}/{len(positions)}", flush=True)
    rows = []
    for slot in names:
        vals = []
        self_vals = []
        for cat, slots in by_cat.items():
            y = np.concatenate(slots[slot]["y"])
            s = np.concatenate(slots[slot]["s"])
            p_auc = safe_auc(y, s)
            p_ap = safe_ap(y, s)
            g = 0.5 * (p_auc + p_ap) if np.isfinite(p_auc) and np.isfinite(p_ap) else float("nan")
            same = cat == slot or slot == f"random_{cat}"
            rows.append({"memory": "source_slot_eval", "slot": slot, "source_category": cat, "self_category": bool(same), "P-AUROC": p_auc, "P-AP": p_ap, "G_pixel": g})
            if np.isfinite(g):
                if same:
                    self_vals.append(g)
                else:
                    vals.append(g)
        rows.append({"memory": "transferability", "slot": slot, "source_category": "__cross_mean__", "self_category": False, "P-AUROC": float("nan"), "P-AP": float("nan"), "G_pixel": float(np.mean(vals)) if vals else float("nan")})
        rows.append({"memory": "transferability", "slot": slot, "source_category": "__self_mean__", "self_category": True, "P-AUROC": float("nan"), "P-AP": float("nan"), "G_pixel": float(np.mean(self_vals)) if self_vals else float("nan")})
    return rows


def transferability_weights(eval_rows, vectors, label):
    t = {}
    self_g = {}
    for r in eval_rows:
        if r.get("memory") != "transferability":
            continue
        if r["slot"] not in vectors:
            continue
        if r["source_category"] == "__cross_mean__":
            val = float(r["G_pixel"])
            t[r["slot"]] = val if np.isfinite(val) else 0.0
        if r["source_category"] == "__self_mean__":
            val = float(r["G_pixel"])
            self_g[r["slot"]] = val if np.isfinite(val) else float("nan")
    pos = {k: max(float(t.get(k, 0.0)), 0.0) for k in vectors}
    den = sum(pos.values())
    if den <= 0:
        w = {k: 1.0 / max(len(vectors), 1) for k in vectors}
    else:
        w = {k: pos.get(k, 0.0) / den for k in vectors}
    q = unit(sum(w[k] * unit(vectors[k]) for k in vectors))
    rows = []
    for k in vectors:
        rows.append({"memory": label, "slot": k, "T_c_cross_category_G": t.get(k, float("nan")), "self_category_G": self_g.get(k, float("nan")), "weight": w[k]})
    return q, rows


def uniform_consensus(vectors):
    return unit(np.mean(np.stack([unit(v) for v in vectors.values()], axis=0), axis=0))


def vector_cosine_rows(vectors, label):
    names = list(vectors)
    mat = norm_np(np.stack([vectors[n] for n in names], axis=0))
    sim = mat @ mat.T
    rows = []
    off = sim[~np.eye(sim.shape[0], dtype=bool)]
    rows.append({"memory": label, "slot_a": "__summary__", "slot_b": "__summary__", "pairwise_cosine": float(off.mean()), "pairwise_cosine_std": float(off.std())})
    for i, a in enumerate(names):
        for j, b in enumerate(names[i + 1 :], start=i + 1):
            rows.append({"memory": label, "slot_a": a, "slot_b": b, "pairwise_cosine": float(sim[i, j]), "pairwise_cosine_std": float("nan")})
    return rows


def load_b0_rows():
    raw, percat = [], []
    with (TCM_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "B0_GlobalContrast_ReconstructedAlpha":
                continue
            out = dict(row)
            out["variant"] = "B0_GlobalContrast"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    with (TCM_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] != "B0_GlobalContrast_ReconstructedAlpha":
                continue
            out = dict(row)
            out["variant"] = "B0_GlobalContrast"
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            percat.append(out)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        e1 = by[(ds, "E1_UniformMemoryConsensus")]
        e2 = by[(ds, "E2_TransferabilityFilteredConsensus")]
        e3 = by[(ds, "E3_RandomTransferabilityConsensus")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_E1_G": float(e2["G_pixel"]) - float(e1["G_pixel"]),
            "E2_minus_E3_G": float(e2["G_pixel"]) - float(e3["G_pixel"]),
            "E2_minus_B0_P-AUROC": float(e2["P-AUROC"]) - float(b0["P-AUROC"]),
            "E2_minus_B0_P-AP": float(e2["P-AP"]) - float(b0["P-AP"]),
            "E2_minus_E3_P-AUROC": float(e2["P-AUROC"]) - float(e3["P-AUROC"]),
            "E2_minus_E3_P-AP": float(e2["P-AP"]) - float(e3["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e2_b0 = np.asarray([r["E2_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e2_e1 = np.asarray([r["E2_minus_E1_G"] for r in summary_rows], dtype=np.float32)
    e2_e3 = np.asarray([r["E2_minus_E3_G"] for r in summary_rows], dtype=np.float32)
    stable_b0 = int(np.sum(e2_b0 > 0)) >= 2 and float(e2_b0.mean()) >= 1.0
    if stable_b0 and float(e2_e3.mean()) >= 1.0:
        outcome = "TRANSFERABILITY_FILTERED_CONTRAST_MEMORY_SUPPORTED"
    elif float(e2_e1.mean()) > 0 and float(e2_e3.mean()) > 0 and float(e2_b0.mean()) <= 0:
        outcome = "MEMORY_FILTERING_SIGNAL_BUT_GLOBAL_CONTRAST_STILL_BEST"
    elif abs(float(e2_e1.mean())) < 1.0:
        outcome = "CROSS_SOURCE_TRANSFERABILITY_WEIGHTING_NO_SIGNAL"
    else:
        outcome = "CONTRAST_MEMORY_METHOD_CLOSED"
    return outcome, {
        "mean_E2_minus_B0_G": float(e2_b0.mean()),
        "mean_E2_minus_E1_G": float(e2_e1.mean()),
        "mean_E2_minus_E3_G": float(e2_e3.mean()),
        "E2_targets_improved_over_B0": int(np.sum(e2_b0 > 0)),
    }


def write_report(summary_rows, outcome, stats, runtime):
    lines = [
        "# Transferability-Filtered Contrast Memory",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only. Existing true/random contrast slots are reused without retraining. Cross-category transferability weights are computed only on held source categories and the final target model keeps one consensus contrast direction.",
        "",
        "| Dataset | B0 Global G | E1 Uniform G | E2 TF G | E3 Random TF G | E2-B0 | E2-E1 | E2-E3 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E2_minus_E1_G']:+.4f} | {r['E2_minus_E3_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f} pp.",
        f"Mean E2-E1 G: {stats['mean_E2_minus_E1_G']:+.4f} pp.",
        f"Mean E2-E3 G: {stats['mean_E2_minus_E3_G']:+.4f} pp.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TFContrast] device={device}", flush=True)
    true_vectors = load_slot_vectors("E1_")
    random_vectors = load_slot_vectors("E2_")
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_c, _d_c = decompose(states_c)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    _m_t, d_t = decompose(np.asarray(true_payload["patch_states"], dtype=np.float32))
    alpha = float(np.linalg.norm(d_t))
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y, _cats = load_full_visa_source_lazy(False)
    labels_by_pos = source_true_masks()

    true_eval = source_slot_pixel_scores(true_vectors, align_model, m_c, alpha, q_pc1, dataset, indices, patch, labels_by_pos, args.smoke)
    random_eval = source_slot_pixel_scores(random_vectors, align_model, m_c, alpha, q_pc1, dataset, indices, patch, labels_by_pos, args.smoke)
    q_uniform = uniform_consensus(true_vectors)
    q_tf, true_weight_rows = transferability_weights(true_eval, true_vectors, "E2_TransferabilityFilteredConsensus")
    q_rand_tf, random_weight_rows = transferability_weights(random_eval, random_vectors, "E3_RandomTransferabilityConsensus")

    models = {
        "E1_UniformMemoryConsensus": make_model(hybrid_states(m_c, alpha * q_uniform), corrupt_payload, device),
        "E2_TransferabilityFilteredConsensus": make_model(hybrid_states(m_c, alpha * q_tf), corrupt_payload, device),
        "E3_RandomTransferabilityConsensus": make_model(hybrid_states(m_c, alpha * q_rand_tf), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    outcome, stats = decide(summary_rows)

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "SOURCE_TRANSFERABILITY.csv", true_eval + random_eval)
    write_csv(OUT / "SLOT_WEIGHTS.csv", true_weight_rows + random_weight_rows)
    write_csv(OUT / "PAIRWISE_CONTRAST_COSINE.csv", vector_cosine_rows(true_vectors, "E1_true_slots") + vector_cosine_rows(random_vectors, "E3_random_slots"))
    np.savez_compressed(OUT / "consensus_vectors.npz", uniform=q_uniform.astype(np.float32), transferability_filtered=q_tf.astype(np.float32), random_transferability_filtered=q_rand_tf.astype(np.float32))
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "slot_retraining": False,
        "alpha": alpha,
        "source": str(TCM_OUT),
    })
    write_report(summary_rows, outcome, stats, runtime)
    print(f"[TFContrast] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
