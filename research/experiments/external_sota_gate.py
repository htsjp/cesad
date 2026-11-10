import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv


VISUALAD_CLIP_PIXEL = {
    "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
    "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
    "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
}


def _float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def read_raw(out_dir):
    path = Path(out_dir) / "RAW_METRICS.csv"
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def method_rows(raw, variant):
    rows = []
    for row in raw:
        dataset = row.get("dataset") or row.get("target") or row.get("cell")
        if row.get("variant") == variant and dataset in VISUALAD_CLIP_PIXEL:
            rows.append((dataset, row))
    return rows


def build_sota_gap(out_dir, method_variant, method_label=None):
    method_label = method_label or method_variant
    rows = []
    seen = set()
    for dataset, row in method_rows(read_raw(out_dir), method_variant):
        if dataset in seen:
            continue
        seen.add(dataset)
        ref = VISUALAD_CLIP_PIXEL[dataset]
        p_auroc = _float(row, "P-AUROC")
        p_ap = _float(row, "P-AP")
        g = _float(row, "G_pixel")
        rows.append({
            "dataset": dataset,
            "current_method": method_label,
            "current_P-AUROC": p_auroc,
            "current_P-AP": p_ap,
            "current_G": g,
            "reference": "CVPR 2026 VisualAD(CLIP) pixel, VisA-source cross-dataset",
            "reference_P-AUROC": ref["P-AUROC"],
            "reference_P-AP": ref["P-AP"],
            "reference_G": ref["G_pixel"],
            "Delta_P_AUROC": p_auroc - ref["P-AUROC"],
            "Delta_P_AP": p_ap - ref["P-AP"],
            "Delta_G": g - ref["G_pixel"],
            "comparison_status": "directly_comparable_if_protocol_matches_VisA_source_frozen_CLIP_pixel_PAUROC_PAP",
        })
    if set(seen) != set(VISUALAD_CLIP_PIXEL):
        missing = sorted(set(VISUALAD_CLIP_PIXEL) - set(seen))
        raise RuntimeError(f"Missing SOTA comparable datasets for {method_variant}: {missing}")
    write_csv(Path(out_dir) / "SOTA_GAP.csv", rows)
    return rows


def external_section(rows, stop_loss_note):
    mean_delta = sum(float(r["Delta_G"]) for r in rows) / len(rows)
    lines = [
        "## External SOTA Position",
        "",
        "Reference: CVPR 2026 VisualAD(CLIP) pixel results for VisA-source cross-dataset evaluation.",
        "",
        "| Dataset | Current P-AUROC | Current P-AP | Current G | VisualAD P-AUROC | VisualAD P-AP | VisualAD G | Delta G |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['dataset']} | {float(r['current_P-AUROC']):.4f} | {float(r['current_P-AP']):.4f} | {float(r['current_G']):.4f} | "
            f"{float(r['reference_P-AUROC']):.4f} | {float(r['reference_P-AP']):.4f} | {float(r['reference_G']):.4f} | {float(r['Delta_G']):+.4f} |"
        )
    lines.extend([
        "",
        "Directly comparable: pixel P-AUROC / P-AP / G on MVTec, BTAD, and CVC-ClinicDB under VisA-source cross-dataset frozen-CLIP-style evaluation.",
        "Reference-only caveat: implementation details such as exact feature extractor, preprocessing, checkpoint provenance, and evaluation code may differ; do not claim full leaderboard superiority without protocol-level reproduction.",
        f"Mean Delta_G versus VisualAD(CLIP): {mean_delta:+.4f} pp.",
        stop_loss_note,
    ])
    return "\n".join(lines) + "\n"


def update_report(out_dir, rows, stop_loss_note):
    report = Path(out_dir) / "REPORT.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    marker = "## External SOTA Position"
    if marker in text:
        text = text[: text.index(marker)].rstrip() + "\n\n"
    text = text.rstrip() + "\n\n" + external_section(rows, stop_loss_note)
    report.write_text(text, encoding="utf-8")


def apply_gate(out_dir, method_variant, method_label=None, stop_loss_note="Stop-loss: not triggered for this single report; evaluate over consecutive method experiments."):
    rows = build_sota_gap(out_dir, method_variant, method_label)
    update_report(out_dir, rows, stop_loss_note)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--stop-loss-note", default="Stop-loss: not triggered for this single report; evaluate over consecutive method experiments.")
    args = parser.parse_args()
    apply_gate(args.out_dir, args.variant, args.label, args.stop_loss_note)


if __name__ == "__main__":
    main()
