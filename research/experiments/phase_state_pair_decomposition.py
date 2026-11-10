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

from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xxxviii_visual_state_representation import norm_np

OUT = Path("results/phase_state_pair_decomposition/formal")
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


def decompose(states):
    s = np.asarray(states, dtype=np.float32)
    m = (s[1] + s[0]) / 2.0
    d = (s[1] - s[0]) / 2.0
    return m, d


def hybrid_states(midpoint, contrast):
    s_n = midpoint - contrast
    s_a = midpoint + contrast
    return np.stack([s_n, s_a]).astype(np.float32)


def geometry_rows(states_t, states_c):
    m_t, d_t = decompose(states_t)
    m_c, d_c = decompose(states_c)
    return [
        {"metric": "cos_m_T_m_C", "value": float(norm_np(m_t) @ norm_np(m_c))},
        {"metric": "cos_d_T_d_C", "value": float(norm_np(d_t) @ norm_np(d_c))},
        {"metric": "norm_m_T", "value": float(np.linalg.norm(m_t))},
        {"metric": "norm_m_C", "value": float(np.linalg.norm(m_c))},
        {"metric": "norm_d_T", "value": float(np.linalg.norm(d_t))},
        {"metric": "norm_d_C", "value": float(np.linalg.norm(d_c))},
        {"metric": "norm_m_T_over_m_C", "value": float(np.linalg.norm(m_t) / max(float(np.linalg.norm(m_c)), 1e-12))},
        {"metric": "norm_d_T_over_d_C", "value": float(np.linalg.norm(d_t) / max(float(np.linalg.norm(d_c)), 1e-12))},
    ]


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_TrueStates_plus_A_C")]
        b1 = by[(ds, "B1_CorruptStates_plus_A_C")]
        e1 = by[(ds, "E1_TrueContrast")]
        e2 = by[(ds, "E2_TrueMidpoint")]
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
    b0_b1_gap = np.asarray([r["B0_G"] - r["B1_G"] for r in summary_rows], dtype=np.float32)
    between_e1 = np.all((np.asarray([r["E1_G"] for r in summary_rows]) > np.asarray([r["B1_G"] for r in summary_rows]) + 3.0) & (np.asarray([r["E1_G"] for r in summary_rows]) < np.asarray([r["B0_G"] for r in summary_rows]) - 3.0))
    between_e2 = np.all((np.asarray([r["E2_G"] for r in summary_rows]) > np.asarray([r["B1_G"] for r in summary_rows]) + 3.0) & (np.asarray([r["E2_G"] for r in summary_rows]) < np.asarray([r["B0_G"] for r in summary_rows]) - 3.0))
    if e1_close_b0 and e2_close_b1:
        outcome = "STATE_CONTRAST_DOMINANT"
    elif e2_close_b0 and e1_close_b1:
        outcome = "STATE_COMMON_COMPONENT_DOMINANT"
    elif bool(np.all(b0_b1_gap > 6.0)) and between_e1 and between_e2:
        outcome = "STATE_COMMON_AND_CONTRAST_BOTH_REQUIRED"
    else:
        outcome = "STATE_PAIR_DECOMPOSITION_INCONCLUSIVE"
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
        "# State Pair Decomposition Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: fixed corrupted alignment A_C. Only state midpoint/contrast components are recombined; no retraining or target statistics.",
        "",
        "| Dataset | B0 TrueStates+A_C G | B1 CorruptStates+A_C G | E1 TrueContrast G | E2 TrueMidpoint G | E1-B0 | E2-B0 | E1-B1 | E2-B1 |",
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
        f"cos(m_T,m_C): {g['cos_m_T_m_C']:.6f}",
        f"cos(d_T,d_C): {g['cos_d_T_d_C']:.6f}",
        f"||m_T||/||m_C||: {g['norm_m_T_over_m_C']:.6f}",
        f"||d_T||/||d_C||: {g['norm_d_T_over_d_C']:.6f}",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StatePair] device={device}", flush=True)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_t, d_t = decompose(states_t)
    m_c, d_c = decompose(states_c)
    states_e1 = hybrid_states(m_c, d_t)
    states_e2 = hybrid_states(m_t, d_c)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    models = {
        "B0_TrueStates_plus_A_C": make_model(states_t, corrupt_payload, device),
        "B1_CorruptStates_plus_A_C": make_model(states_c, corrupt_payload, device),
        "E1_TrueContrast": make_model(states_e1, corrupt_payload, device),
        "E2_TrueMidpoint": make_model(states_e2, corrupt_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    summary_rows = summarize(raw)
    geom = geometry_rows(states_t, states_c)
    outcome, stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom)
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
        "alignment_fixed": "A_C",
        "true_checkpoint": str(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt"),
        "corrupt_checkpoint": str(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt"),
    })
    write_report(summary_rows, geom, outcome, stats, runtime)
    print(f"[StatePair] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
