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
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xxxviii_visual_state_representation import norm_np

OUT = Path("results/phase_statealigned_state_alignment_swap/formal")
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


def make_model(states, align_payload, device):
    model = AdapterToState(1024, states).to(device).eval()
    state = model.state_dict()
    src = align_payload["model"]
    state["a.weight"] = src["a.weight"].detach().clone()
    state["b.weight"] = src["b.weight"].detach().clone()
    state["states"] = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
    model.load_state_dict(state, strict=True)
    return model.eval()


def state_geometry(states_t, states_c):
    st = norm_np(states_t)
    sc = norm_np(states_c)
    dt = norm_np(st[1] - st[0])
    dc = norm_np(sc[1] - sc[0])
    rows = [
        {"metric": "cos_sA_T_sA_C", "value": float(st[1] @ sc[1])},
        {"metric": "cos_sN_T_sN_C", "value": float(st[0] @ sc[0])},
        {"metric": "cos_delta_T_delta_C", "value": float(dt @ dc)},
        {"metric": "norm_sA_T_raw", "value": float(np.linalg.norm(states_t[1]))},
        {"metric": "norm_sN_T_raw", "value": float(np.linalg.norm(states_t[0]))},
        {"metric": "norm_sA_C_raw", "value": float(np.linalg.norm(states_c[1]))},
        {"metric": "norm_sN_C_raw", "value": float(np.linalg.norm(states_c[0]))},
        {"metric": "cos_sA_sN_T", "value": float(st[1] @ st[0])},
        {"metric": "cos_sA_sN_C", "value": float(sc[1] @ sc[0])},
    ]
    return rows


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        tt = by[(ds, "B0_TT_S_T_plus_A_T")]
        cc = by[(ds, "B1_CC_S_C_plus_A_C")]
        tc = by[(ds, "E1_TC_S_T_plus_A_C")]
        ct = by[(ds, "E2_CT_S_C_plus_A_T")]
        rows.append({
            "dataset": ds,
            "TT_G": float(tt["G_pixel"]),
            "CC_G": float(cc["G_pixel"]),
            "TC_G": float(tc["G_pixel"]),
            "CT_G": float(ct["G_pixel"]),
            "TT_minus_CC_G": float(tt["G_pixel"]) - float(cc["G_pixel"]),
            "TC_minus_CC_G": float(tc["G_pixel"]) - float(cc["G_pixel"]),
            "CT_minus_CC_G": float(ct["G_pixel"]) - float(cc["G_pixel"]),
            "TT_minus_TC_G": float(tt["G_pixel"]) - float(tc["G_pixel"]),
            "TT_minus_CT_G": float(tt["G_pixel"]) - float(ct["G_pixel"]),
            "TT_minus_CC_P-AUROC": float(tt["P-AUROC"]) - float(cc["P-AUROC"]),
            "TT_minus_CC_P-AP": float(tt["P-AP"]) - float(cc["P-AP"]),
            "TC_minus_CC_P-AUROC": float(tc["P-AUROC"]) - float(cc["P-AUROC"]),
            "TC_minus_CC_P-AP": float(tc["P-AP"]) - float(cc["P-AP"]),
            "CT_minus_CC_P-AUROC": float(ct["P-AUROC"]) - float(cc["P-AUROC"]),
            "CT_minus_CC_P-AP": float(ct["P-AP"]) - float(cc["P-AP"]),
        })
    return rows


def decide(summary_rows):
    tt_tc = np.asarray([r["TT_minus_TC_G"] for r in summary_rows], dtype=np.float32)
    tt_ct = np.asarray([r["TT_minus_CT_G"] for r in summary_rows], dtype=np.float32)
    tc_cc = np.asarray([r["TC_minus_CC_G"] for r in summary_rows], dtype=np.float32)
    ct_cc = np.asarray([r["CT_minus_CC_G"] for r in summary_rows], dtype=np.float32)
    mean_tt_tc = float(tt_tc.mean())
    mean_tt_ct = float(tt_ct.mean())
    mean_tc_cc = float(tc_cc.mean())
    mean_ct_cc = float(ct_cc.mean())
    tc_close_tt = abs(mean_tt_tc) <= 3.0
    ct_close_tt = abs(mean_tt_ct) <= 3.0
    tc_close_cc = abs(mean_tc_cc) <= 3.0
    ct_close_cc = abs(mean_ct_cc) <= 3.0
    if tc_close_tt and ct_close_cc:
        outcome = "TRANSFERABLE_KNOWLEDGE_STATE_DOMINANT"
    elif ct_close_tt and tc_close_cc:
        outcome = "TRANSFERABLE_KNOWLEDGE_ALIGNMENT_DOMINANT"
    elif mean_tt_tc >= 3.0 and mean_tt_ct >= 3.0 and mean_tc_cc >= -3.0 and mean_ct_cc >= -3.0:
        outcome = "STATE_ALIGNMENT_COADAPTATION_REQUIRED"
    elif tc_close_tt and ct_close_tt:
        outcome = "STATE_ALIGNMENT_INFORMATION_REDUNDANT"
    else:
        outcome = "STATE_ALIGNMENT_CAUSALITY_INCONCLUSIVE"
    return outcome, {
        "mean_TT_minus_CC_G": float(np.mean([r["TT_minus_CC_G"] for r in summary_rows])),
        "mean_TC_minus_CC_G": mean_tc_cc,
        "mean_CT_minus_CC_G": mean_ct_cc,
        "mean_TT_minus_TC_G": mean_tt_tc,
        "mean_TT_minus_CT_G": mean_tt_ct,
        "close_threshold_pp": 3.0,
    }


def write_report(summary_rows, geom_rows, outcome, stats, runtime_sec):
    g = {r["metric"]: r["value"] for r in geom_rows}
    lines = [
        "# StateAligned State-Alignment Causal Swap Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA source-only PC1Removed StateAligned checkpoints. Only normal/anomaly states and rank4 alignment weights are swapped; no retraining or target statistics are used.",
        "",
        "| Dataset | TT G | CC G | TC G | CT G | TT-CC | TC-CC | CT-CC | TT-TC | TT-CT |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['TT_G']:.4f} | {r['CC_G']:.4f} | {r['TC_G']:.4f} | {r['CT_G']:.4f} | {r['TT_minus_CC_G']:+.4f} | {r['TC_minus_CC_G']:+.4f} | {r['CT_minus_CC_G']:+.4f} | {r['TT_minus_TC_G']:+.4f} | {r['TT_minus_CT_G']:+.4f} |"
        )
    lines.extend([
        "",
        f"Mean TT-CC G: {stats['mean_TT_minus_CC_G']:+.4f} pp.",
        f"Mean TC-CC G: {stats['mean_TC_minus_CC_G']:+.4f} pp.",
        f"Mean CT-CC G: {stats['mean_CT_minus_CC_G']:+.4f} pp.",
        f"Mean TT-TC G: {stats['mean_TT_minus_TC_G']:+.4f} pp.",
        f"Mean TT-CT G: {stats['mean_TT_minus_CT_G']:+.4f} pp.",
        "",
        f"cos(sA_T,sA_C): {g['cos_sA_T_sA_C']:.6f}",
        f"cos(sN_T,sN_C): {g['cos_sN_T_sN_C']:.6f}",
        f"cos(delta_T,delta_C): {g['cos_delta_T_delta_C']:.6f}",
        "",
        f"Runtime: {runtime_sec:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[StateSwap] device={device}", flush=True)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)

    models = {
        "B0_TT_S_T_plus_A_T": make_model(states_t, true_payload, device),
        "B1_CC_S_C_plus_A_C": make_model(states_c, corrupt_payload, device),
        "E1_TC_S_T_plus_A_C": make_model(states_t, corrupt_payload, device),
        "E2_CT_S_C_plus_A_T": make_model(states_c, true_payload, device),
    }
    raw, percat = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    summary_rows = summarize(raw)
    geom_rows = state_geometry(states_t, states_c)
    outcome, decision_stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "STATE_GEOMETRY.csv", geom_rows)
    runtime_sec = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime_sec,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "memory_used": False,
        "variants": list(models.keys()),
        "true_checkpoint": str(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt"),
        "corrupt_checkpoint": str(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt"),
        "pc1_source": str(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy"),
    })
    write_report(summary_rows, geom_rows, outcome, decision_stats, runtime_sec)
    print(f"[StateSwap] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
