import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import norm_np

OUT = Path("results/phase_state_contrast_direction_magnitude/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")


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


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def geometry_rows(d_t, d_c):
    nt = float(np.linalg.norm(d_t))
    nc = float(np.linalg.norm(d_c))
    return [
        {"metric": "norm_d_T", "value": nt},
        {"metric": "norm_d_C", "value": nc},
        {"metric": "norm_d_T_over_d_C", "value": nt / max(nc, 1e-12)},
        {"metric": "cos_d_T_d_C", "value": float(unit(d_t) @ unit(d_c))},
    ]


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_TrueContrast")]
        b1 = by[(ds, "B1_CorruptContrast")]
        e1 = by[(ds, "E1_TrueDirection-CorruptMagnitude")]
        e2 = by[(ds, "E2_CorruptDirection-TrueMagnitude")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "B1_G": float(b1["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E2_minus_B0_G": float(e2["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_B1_G": float(e1["G_pixel"]) - float(b1["G_pixel"]),
            "E2_minus_B1_G": float(e2["G_pixel"]) - float(b1["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e2_b0 = np.asarray([r["E2_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_b1 = np.asarray([r["E1_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    e2_b1 = np.asarray([r["E2_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    mean_e1_b0 = float(e1_b0.mean())
    mean_e2_b0 = float(e2_b0.mean())
    mean_e1_b1 = float(e1_b1.mean())
    mean_e2_b1 = float(e2_b1.mean())
    e1_close_b0 = abs(mean_e1_b0) <= 3.0
    e2_close_b0 = abs(mean_e2_b0) <= 3.0
    e1_close_b1 = abs(mean_e1_b1) <= 3.0
    e2_close_b1 = abs(mean_e2_b1) <= 3.0
    vals_b0 = np.asarray([r["B0_G"] for r in summary_rows])
    vals_b1 = np.asarray([r["B1_G"] for r in summary_rows])
    vals_e1 = np.asarray([r["E1_G"] for r in summary_rows])
    vals_e2 = np.asarray([r["E2_G"] for r in summary_rows])
    lo = np.minimum(vals_b0, vals_b1) + 3.0
    hi = np.maximum(vals_b0, vals_b1) - 3.0
    both_between = bool(np.all((vals_e1 > lo) & (vals_e1 < hi) & (vals_e2 > lo) & (vals_e2 < hi)))
    if e1_close_b0 and e2_close_b1:
        outcome = "STATE_CONTRAST_DIRECTION_DOMINANT"
    elif e2_close_b0 and e1_close_b1:
        outcome = "STATE_CONTRAST_MAGNITUDE_DOMINANT"
    elif both_between:
        outcome = "STATE_CONTRAST_DIRECTION_AND_MAGNITUDE_BOTH_REQUIRED"
    else:
        outcome = "STATE_CONTRAST_FACTOR_INCONCLUSIVE"
    return outcome, {
        "mean_E1_minus_B0_G": mean_e1_b0,
        "mean_E2_minus_B0_G": mean_e2_b0,
        "mean_E1_minus_B1_G": mean_e1_b1,
        "mean_E2_minus_B1_G": mean_e2_b1,
        "close_threshold_pp": 3.0,
    }


def write_report(summary_rows, geom, outcome, stats, runtime):
    g = {r["metric"]: r["value"] for r in geom}
    lines = [
        "# State Contrast Direction-vs-Magnitude Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: fixed corrupted midpoint m_C and corrupted alignment A_C. Only contrast direction and magnitude are swapped; no retraining or target statistics.",
        "",
        "| Dataset | B0 TrueContrast G | B1 CorruptContrast G | E1 TrueDir-CorruptMag G | E2 CorruptDir-TrueMag G | E1-B0 | E2-B0 | E1-B1 | E2-B1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E2_minus_B0_G']:+.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E2_minus_B1_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E2-B0 G: {stats['mean_E2_minus_B0_G']:+.4f} pp.",
        f"Mean E1-B1 G: {stats['mean_E1_minus_B1_G']:+.4f} pp.",
        f"Mean E2-B1 G: {stats['mean_E2_minus_B1_G']:+.4f} pp.",
        "",
        f"||d_T||: {g['norm_d_T']:.6f}",
        f"||d_C||: {g['norm_d_C']:.6f}",
        f"||d_T||/||d_C||: {g['norm_d_T_over_d_C']:.6f}",
        f"cos(d_T,d_C): {g['cos_d_T_d_C']:.6f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ContrastDirMag] device={device}", flush=True)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, d_c = decompose(states_c)
    d_e1 = float(np.linalg.norm(d_c)) * unit(d_t)
    d_e2 = float(np.linalg.norm(d_t)) * unit(d_c)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    models = {
        "B0_TrueContrast": make_model(hybrid_states(m_c, d_t), corrupt_payload, device),
        "B1_CorruptContrast": make_model(hybrid_states(m_c, d_c), corrupt_payload, device),
        "E1_TrueDirection-CorruptMagnitude": make_model(hybrid_states(m_c, d_e1), corrupt_payload, device),
        "E2_CorruptDirection-TrueMagnitude": make_model(hybrid_states(m_c, d_e2), corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    summary_rows = summarize(raw)
    geom = geometry_rows(d_t, d_c)
    outcome, stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "CONTRAST_GEOMETRY.csv", geom)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "memory_used": False,
        "midpoint_fixed": "m_C",
        "alignment_fixed": "A_C",
    })
    write_report(summary_rows, geom, outcome, stats, runtime)
    print(f"[ContrastDirMag] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
