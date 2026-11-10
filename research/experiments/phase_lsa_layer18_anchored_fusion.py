import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_lsa_layerwise_state_alignment import BASELINE, PILOT_PAIRS
from experiments.phase_parameter_free_local_zoom_pyramid_fullres import fnum
from experiments.standard_zsad_metrics import safe_ap
import torch
import torch.nn.functional as F
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_lsa_layer18_anchored_fusion/formal")
LSA_OUT = Path("results/phase_layerwise_state_alignment/formal")
E1 = "E1_L18_SAF"
VARIANTS = ("A0_Core_NTR_FixedLRP", "A1_Base90_L18_10", "A2_Base75_L18_25")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_lsa_maps(source, target):
    cache = LSA_OUT / f"{source.replace(' ', '_')}_to_{target}_LSA_MAPS_formal.npz"
    if not cache.exists():
        raise FileNotFoundError(cache)
    data = np.load(cache, allow_pickle=True)
    base = np.asarray(data[BASELINE], dtype=np.float32)
    lsa_e1 = np.asarray(data[E1], dtype=np.float32)
    q18 = (2.0 * lsa_e1 - base).astype(np.float32)
    maps = {
        VARIANTS[0]: [x.astype(np.float32) for x in base],
        VARIANTS[1]: [x.astype(np.float32) for x in (0.9 * base + 0.1 * q18)],
        VARIANTS[2]: [x.astype(np.float32) for x in (0.75 * base + 0.25 * q18)],
    }
    metas = data["metas"].tolist()
    return maps, metas


def selected_eval(source, target, maps, metas):
    raw = []
    by_cat = {}
    for i, meta in enumerate(metas):
        by_cat.setdefault(meta["category"], []).append(i)
    for variant in VARIANTS:
        cat_ap = []
        for cat, idxs in sorted(by_cat.items()):
            ys, ss = [], []
            for i in idxs:
                m = np.asarray(metas[i]["mask"], dtype=np.uint8)
                s = np.asarray(maps[variant][i], dtype=np.float32)
                mt = torch.tensor(m.astype(np.float32)).view(1, 1, *m.shape)
                mg = F.interpolate(mt, size=s.shape, mode="nearest").view(*s.shape).numpy().astype(np.uint8)
                ys.append(mg.reshape(-1))
                ss.append(s.reshape(-1))
            ap = safe_ap(np.concatenate(ys), np.concatenate(ss))
            if np.isfinite(ap):
                cat_ap.append(ap)
        row = {
            "source": source,
            "dataset": target,
            "variant": variant,
            "protocol": "LRP_74",
            "aggregation": "CATEGORY_MACRO_FAST_P_AP_ONLY",
            "M_FR_P_AUROC": float("nan"),
            "M_FR_P_AP": float(np.mean(cat_ap)) if cat_ap else float("nan"),
            "M_FR_P_F1max": float("nan"),
            "M_FR_PRO": float("nan"),
        }
        raw.append(row)
    return raw, []


def delta_summary(raw):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for source, target in PILOT_PAIRS:
        b = by[(source, target, VARIANTS[0])]
        for variant in VARIANTS[1:]:
            e = by[(source, target, variant)]
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP"),
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
    return rows


def summarize(deltas):
    out = []
    for variant in VARIANTS[1:]:
        vals = np.asarray([float(r["Delta_M_FR_P_AP"]) for r in deltas if r["variant"] == variant], dtype=np.float64)
        out.append({
            "variant": variant,
            "mean_delta_p_ap": float(vals.mean()),
            "median_delta_p_ap": float(np.median(vals)),
            "improved": int(np.sum(vals > 0)),
            "non_drop_ge_neg_0p2": int(np.sum(vals >= -0.2)),
            "total": int(vals.size),
            "mpdd_delta_p_ap": float(next(r["Delta_M_FR_P_AP"] for r in deltas if r["variant"] == variant and r["dataset"] == "MPDD")),
        })
    return out


def decide(summary):
    best = max(summary, key=lambda r: r["mean_delta_p_ap"])
    if best["mean_delta_p_ap"] > 0 and best["improved"] >= 3 and best["mpdd_delta_p_ap"] >= -0.2:
        return "LAYER18_SAF_ANCHORED_FUSION_HAS_WEAK_SIGNAL"
    return "LAYER18_SAF_ANCHORED_FUSION_NO_USEFUL_GAIN"


def write_report(raw, deltas, summary, outcome, runtime):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    lines = [
        "# Layer18 SAF Anchored Fusion",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        "This is a no-retrain close-out. It reuses LSA E1 layer18 maps and reconstructs `q18 = 2*E1 - q_base` from the previous `0.5*q_base + 0.5*q18` cache.",
        "",
        "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO | Delta P-AP |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for source, target in PILOT_PAIRS:
        b = by[(source, target, VARIANTS[0])]
        for variant in VARIANTS:
            r = by[(source, target, variant)]
            delta = 0.0 if variant == VARIANTS[0] else fnum(r, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            lines.append(f"| {source} | {target} | {variant} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} | {delta:+.4f} |")
    lines += ["", "## Summary", "", "| Variant | Mean Delta P-AP | Median | Improved | Non-drop >= -0.2 | MPDD Delta |", "|---|---:|---:|---:|---:|---:|"]
    for s in summary:
        lines.append(f"| {s['variant']} | {s['mean_delta_p_ap']:+.4f} | {s['median_delta_p_ap']:+.4f} | {s['improved']}/{s['total']} | {s['non_drop_ge_neg_0p2']}/{s['total']} | {s['mpdd_delta_p_ap']:+.4f} |")
    lines += ["", "Conclusion: anchored layer18 fusion does not justify continuing SAF/LSA if MPDD remains negative or mean gain is not positive under this fixed two-weight close-out."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw, percat = [], []
    for source, target in PILOT_PAIRS:
        maps, metas = load_lsa_maps(source, target)
        rr, cc = selected_eval(source, target, maps, metas)
        raw.extend(rr)
        percat.extend(cc)
        write_csv(OUT / "RAW_METRICS.partial.csv", raw)
    deltas = delta_summary(raw)
    summary = summarize(deltas)
    outcome = decide(summary)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_STANDARD_METRICS.csv", percat)
    write_csv(OUT / "PILOT_DELTAS.csv", deltas)
    write_json(OUT / "OUTCOME.json", {"outcome": outcome, "summary": summary})
    write_report(raw, deltas, summary, outcome, time.time() - t0)
    print(f"[LSAAnchored] OUTCOME={outcome}", flush=True)


if __name__ == "__main__":
    main()
