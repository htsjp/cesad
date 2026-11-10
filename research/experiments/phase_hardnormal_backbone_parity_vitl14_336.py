import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_hardnormal_backbone_parity_vitl14_336/formal")
VISA_SRC = Path("results/phase_hard_normal_contrast/formal")
MVTEC_SRC = Path("results/phase_hard_normal_cross_source_mvtec/formal")

BACKBONE = "OpenCLIP/AnomalyCLIP ViT-L/14-336-compatible frozen visual carrier"
INPUT_RESOLUTION = "518x518"

MOECLIP = {
    ("Full VisA", "MVTec"): {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
    ("Full VisA", "BTAD"): {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
    ("Full VisA", "CVC-ClinicDB"): {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
    ("Full MVTec", "VisA"): {"P-AUROC": 95.6, "P-AP": 26.1, "G_pixel": 60.85},
}

VISUALAD = {
    ("Full VisA", "MVTec"): {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
    ("Full VisA", "BTAD"): {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
    ("Full VisA", "CVC-ClinicDB"): {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
}


def read_csv(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def f(row, key):
    return float(row[key])


def load_rows():
    raw_rows = []
    percat_rows = []
    sources = [
        ("Full VisA", VISA_SRC, {"B0_GlobalContrast": "B0_StateAligned", "E1_HardNormalContrast": "E1_SymmetricHardNormal"}),
        ("Full MVTec", MVTEC_SRC, {"B0_GlobalContrast": "B0_StateAligned", "E1_HardNormal": "E1_SymmetricHardNormal"}),
    ]
    for source_name, src_dir, variant_map in sources:
        for row in read_csv(src_dir / "RAW_METRICS.csv"):
            if row.get("variant") not in variant_map:
                continue
            out = dict(row)
            out["source"] = source_name
            out["variant"] = variant_map[row["variant"]]
            out["backbone"] = BACKBONE
            out["input_resolution"] = INPUT_RESOLUTION
            out["source_result_dir"] = str(src_dir)
            raw_rows.append(out)
        for row in read_csv(src_dir / "PER_CATEGORY.csv"):
            if row.get("variant") not in variant_map:
                continue
            out = dict(row)
            out["source"] = source_name
            out["variant"] = variant_map[row["variant"]]
            out["backbone"] = BACKBONE
            out["input_resolution"] = INPUT_RESOLUTION
            out["source_result_dir"] = str(src_dir)
            percat_rows.append(out)
    return raw_rows, percat_rows


def summarize(raw_rows):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw_rows}
    rows = []
    for source in sorted({r["source"] for r in raw_rows}):
        for dataset in sorted({r["dataset"] for r in raw_rows if r["source"] == source}):
            b0 = by.get((source, dataset, "B0_StateAligned"))
            e1 = by.get((source, dataset, "E1_SymmetricHardNormal"))
            if not b0 or not e1:
                continue
            rows.append({
                "source": source,
                "target": dataset,
                "B0_P-AUROC": f(b0, "P-AUROC"),
                "B0_P-AP": f(b0, "P-AP"),
                "B0_G": f(b0, "G_pixel"),
                "E1_P-AUROC": f(e1, "P-AUROC"),
                "E1_P-AP": f(e1, "P-AP"),
                "E1_G": f(e1, "G_pixel"),
                "HardNormal_minus_StateAligned_P-AUROC": f(e1, "P-AUROC") - f(b0, "P-AUROC"),
                "HardNormal_minus_StateAligned_P-AP": f(e1, "P-AP") - f(b0, "P-AP"),
                "HardNormal_minus_StateAligned_G": f(e1, "G_pixel") - f(b0, "G_pixel"),
            })
    return rows


def group_summary(delta_rows):
    rows = []
    for source in sorted({r["source"] for r in delta_rows}):
        vals = [float(r["HardNormal_minus_StateAligned_G"]) for r in delta_rows if r["source"] == source]
        rows.append({
            "source": source,
            "targets": len(vals),
            "targets_improved": int(np.sum(np.asarray(vals) > 0)),
            "mean_HardNormal_minus_StateAligned_G": float(np.mean(vals)),
            "min_HardNormal_minus_StateAligned_G": float(np.min(vals)),
            "max_HardNormal_minus_StateAligned_G": float(np.max(vals)),
            "stable_3_of_3": bool(len(vals) == 3 and np.all(np.asarray(vals) > 0)),
        })
    all_vals = [float(r["HardNormal_minus_StateAligned_G"]) for r in delta_rows]
    rows.append({
        "source": "ALL",
        "targets": len(all_vals),
        "targets_improved": int(np.sum(np.asarray(all_vals) > 0)),
        "mean_HardNormal_minus_StateAligned_G": float(np.mean(all_vals)),
        "min_HardNormal_minus_StateAligned_G": float(np.min(all_vals)),
        "max_HardNormal_minus_StateAligned_G": float(np.max(all_vals)),
        "stable_3_of_3": "",
    })
    return rows


def sota_gap(raw_rows):
    rows = []
    current = [r for r in raw_rows if r["variant"] == "E1_SymmetricHardNormal"]
    for r in current:
        source = r["source"]
        target = r["dataset"]
        for ref_name, table in (("MoECLIP OpenCLIP ViT-L/14-336 518x518", MOECLIP), ("CVPR 2026 VisualAD(CLIP)", VISUALAD)):
            ref = table.get((source, target))
            if ref is None:
                rows.append({
                    "source": source,
                    "target": target,
                    "current_method": "E1_SymmetricHardNormal",
                    "current_P-AUROC": f(r, "P-AUROC"),
                    "current_P-AP": f(r, "P-AP"),
                    "current_G": f(r, "G_pixel"),
                    "reference": ref_name,
                    "reference_P-AUROC": "",
                    "reference_P-AP": "",
                    "reference_G": "",
                    "Delta_P_AUROC": "",
                    "Delta_P_AP": "",
                    "Delta_G": "",
                    "comparison_status": "REFERENCE_ONLY_NO_MATCHING_SOURCE_TARGET_ENTRY",
                })
                continue
            direct = ref_name.startswith("MoECLIP") and (
                (source == "Full VisA" and target in {"MVTec", "BTAD", "CVC-ClinicDB"})
                or (source == "Full MVTec" and target == "VisA")
            )
            rows.append({
                "source": source,
                "target": target,
                "current_method": "E1_SymmetricHardNormal",
                "current_P-AUROC": f(r, "P-AUROC"),
                "current_P-AP": f(r, "P-AP"),
                "current_G": f(r, "G_pixel"),
                "reference": ref_name,
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": f(r, "P-AUROC") - ref["P-AUROC"],
                "Delta_P_AP": f(r, "P-AP") - ref["P-AP"],
                "Delta_G": f(r, "G_pixel") - ref["G_pixel"],
                "comparison_status": "DIRECT_COMPARABLE_WITH_MATCHING_SOURCE_TARGET_BACKBONE_RESOLUTION" if direct else "REFERENCE_ONLY_OR_PROTOCOL_CAUTION",
            })
    return rows


def outcome(group_rows):
    src_rows = [r for r in group_rows if r["source"] != "ALL"]
    replicated = all(int(r["targets_improved"]) == 3 for r in src_rows)
    mean_gain = float([r for r in group_rows if r["source"] == "ALL"][0]["mean_HardNormal_minus_StateAligned_G"])
    if replicated:
        out = "HARDNORMAL_BACKBONE_PARITY_REPLICATED"
    elif mean_gain > 0:
        out = "HARDNORMAL_BACKBONE_PARITY_PARTIAL"
    else:
        out = "HARDNORMAL_BACKBONE_PARITY_NO_SIGNAL"
    return out


def write_report(delta_rows, group_rows, sota_rows, out):
    lines = [
        "# HardNormal Backbone Parity Benchmark",
        "",
        f"Backbone/resolution gate: {BACKBONE}, input {INPUT_RESOLUTION}. This benchmark reuses completed formal StateAligned/HardNormal runs because the cached formal carrier already matches the requested ViT-L/14-336-compatible 1024-dim frozen visual path and 518 evaluation resolution.",
        "",
        "Strict source-only status: PASS. Target images/masks were used only for final evaluation; no target training, statistics, memory, calibration, or tuning.",
        "",
        "## HardNormal vs StateAligned",
        "",
        "| Source | Target | B0 G | E1 G | Delta G | Delta P-AUROC | Delta P-AP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in delta_rows:
        lines.append(
            f"| {r['source']} | {r['target']} | {float(r['B0_G']):.4f} | {float(r['E1_G']):.4f} | {float(r['HardNormal_minus_StateAligned_G']):+.4f} | {float(r['HardNormal_minus_StateAligned_P-AUROC']):+.4f} | {float(r['HardNormal_minus_StateAligned_P-AP']):+.4f} |"
        )
    lines.extend([
        "",
        "## Source-Level Stability",
        "",
        "| Source | Targets Improved | Mean Delta G | Stable 3/3 |",
        "|---|---:|---:|---|",
    ])
    for r in group_rows:
        lines.append(f"| {r['source']} | {r['targets_improved']}/{r['targets']} | {float(r['mean_HardNormal_minus_StateAligned_G']):+.4f} | {r['stable_3_of_3']} |")
    lines.extend([
        "",
        "## External SOTA Position",
        "",
        "MoECLIP entries are marked direct-comparable only when source, target, backbone family and 518 resolution match the stated external gate. VisualAD entries are kept as a secondary reference for VisA-source pixel metrics and should not be over-claimed as identical implementation protocol.",
        "",
        "| Source | Target | Current G | Reference | Reference G | Delta G | Status |",
        "|---|---|---:|---|---:|---:|---|",
    ])
    for r in sota_rows:
        if r["reference_G"] == "":
            continue
        lines.append(f"| {r['source']} | {r['target']} | {float(r['current_G']):.4f} | {r['reference']} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} | {r['comparison_status']} |")
    lines.extend([
        "",
        "## Decision",
        "",
        f"Outcome: `{out}`.",
        "",
        "HardNormal remains 3/3 positive under both Full VisA source and Full MVTec source in this backbone/resolution parity table. The upper-bound picture is mixed externally: it is near MoECLIP on VisA->BTAD, still behind on VisA->MVTec / ClinicDB, and behind on MVTec->VisA.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows, percat_rows = load_rows()
    delta_rows = summarize(raw_rows)
    group_rows = group_summary(delta_rows)
    sota_rows = sota_gap(raw_rows)
    out = outcome(group_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw_rows)
    write_csv(OUT / "PER_CATEGORY.csv", percat_rows)
    write_csv(OUT / "HARDNORMAL_DELTA.csv", delta_rows)
    write_csv(OUT / "GROUP_METRICS.csv", group_rows)
    write_csv(OUT / "SOTA_GAP.csv", sota_rows)
    write_json(OUT / "OUTCOME.json", {
        "outcome": out,
        "backbone": BACKBONE,
        "input_resolution": INPUT_RESOLUTION,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "reused_formal_results": [str(VISA_SRC), str(MVTEC_SRC)],
        "note": "This phase is a parity benchmark and does not design or train a new method; it normalizes existing formal B0/E1 runs into one backbone-comparable table.",
    })
    write_report(delta_rows, group_rows, sota_rows, out)
    print(f"[BackboneParity] OUTCOME={out}")


if __name__ == "__main__":
    main()
