import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_keyless_contrast_memory import load_slot_vectors
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import norm_np

OUT = Path("results/phase_source_specific_contrast_subspace_removal/formal")
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


def residual_pc1(vectors, v_global, label):
    names = list(vectors)
    vg = unit(v_global)
    residuals = []
    for name in names:
        v = unit(vectors[name])
        residuals.append(v - float(v @ vg) * vg)
    r = np.stack(residuals, axis=0).astype(np.float64)
    _u, s, vt = np.linalg.svd(r, full_matrices=False)
    u1 = unit(vt[0].astype(np.float32))
    total = float(np.sum(s * s))
    rows = []
    for i, sv in enumerate(s):
        rows.append({
            "memory": label,
            "rank": i + 1,
            "singular_value": float(sv),
            "explained_variance_ratio": float((sv * sv) / max(total, 1e-12)),
            "cumulative_explained_variance_ratio": float(np.sum(s[: i + 1] * s[: i + 1]) / max(total, 1e-12)),
        })
    stats = {
        "memory": label,
        "slot_count": int(len(names)),
        "pc1_explained_variance": float((s[0] * s[0]) / max(total, 1e-12)),
        "residual_frobenius_norm": float(np.sqrt(total)),
        "cos_u1_v_global": float(u1 @ vg),
    }
    return u1, rows, stats


def remove_direction_from_contrast(d_t, u):
    proj = float(np.asarray(d_t, dtype=np.float32) @ unit(u))
    clean = np.asarray(d_t, dtype=np.float32) - proj * unit(u)
    return unit(clean), proj, float(np.linalg.norm(clean))


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
        e1 = by[(ds, "E1_ResidualRemovedGlobal")]
        e2 = by[(ds, "E2_RandomResidualControl")]
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


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    supported = int(np.sum(e1_b0 > 0)) >= 2 and float(e1_b0.mean()) >= 1.0 and float(e1_e2.mean()) >= 1.0
    return (
        "SOURCE_SPECIFIC_CONTRAST_FORGETTING_SUPPORTED" if supported else "SOURCE_SPECIFIC_CONTRAST_FORGETTING_NO_SIGNAL",
        {
            "mean_E1_minus_B0_G": float(e1_b0.mean()),
            "mean_E1_minus_E2_G": float(e1_e2.mean()),
            "E1_targets_improved_over_B0": int(np.sum(e1_b0 > 0)),
        },
    )


def write_report(summary_rows, stats_rows, outcome, decision, runtime):
    stats = {r["memory"]: r for r in stats_rows}
    lines = [
        "# Source-Specific Contrast Subspace Removal",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: reuses trained true/random contrast slots and global contrast. Source-specific residual PC1 is computed source-only, then removed from raw d_T before reconstructing the fixed-alpha StateAligned scorer.",
        "",
        "| Dataset | B0 Global G | E1 Residual-Removed G | E2 RandomResidual G | E1-B0 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {decision['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-E2 G: {decision['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        f"True residual PC1 EVR: {stats['true_slots']['pc1_explained_variance']:.6f}; projection removed from d_T: {stats['true_slots']['projection_removed_from_d_T']:.8f}",
        f"Random residual PC1 EVR: {stats['random_slots']['pc1_explained_variance']:.6f}; projection removed from d_T: {stats['random_slots']['projection_removed_from_d_T']:.8f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SpecForget] device={device}", flush=True)
    true_vectors = load_slot_vectors("E1_")
    random_vectors = load_slot_vectors("E2_")
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    v_global = unit(d_t)
    u_true, spec_true, stats_true = residual_pc1(true_vectors, v_global, "true_slots")
    u_random, spec_random, stats_random = residual_pc1(random_vectors, v_global, "random_slots")
    v_clean, proj_true, clean_norm_true = remove_direction_from_contrast(d_t, u_true)
    v_random_clean, proj_random, clean_norm_random = remove_direction_from_contrast(d_t, u_random)
    stats_true.update({"projection_removed_from_d_T": proj_true, "clean_contrast_norm_before_renorm": clean_norm_true})
    stats_random.update({"projection_removed_from_d_T": proj_random, "clean_contrast_norm_before_renorm": clean_norm_random})
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    models = {
        "E1_ResidualRemovedGlobal": make_model(hybrid_states(m_c, alpha * v_clean), corrupt_payload, device),
        "E2_RandomResidualControl": make_model(hybrid_states(m_c, alpha * v_random_clean), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    b0_raw, b0_percat = load_b0_rows()
    if args.smoke:
        b0_raw = b0_raw[:3]
        b0_percat = b0_percat[:3]
    raw = b0_raw + raw
    percat = b0_percat + percat
    summary_rows = summarize(raw)
    outcome, decision = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "RESIDUAL_SPECTRUM.csv", spec_true + spec_random)
    write_csv(OUT / "RESIDUAL_SUBSPACE_STATS.csv", [stats_true, stats_random])
    np.savez_compressed(OUT / "cleaned_contrast_vectors.npz", v_global=v_global.astype(np.float32), v_clean=v_clean.astype(np.float32), v_random_clean=v_random_clean.astype(np.float32), u_spec_true=u_true.astype(np.float32), u_spec_random=u_random.astype(np.float32))
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "slot_retraining": False,
        "alpha": alpha,
    })
    write_report(summary_rows, [stats_true, stats_random], outcome, decision, runtime)
    print(f"[SpecForget] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
