from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "paper_assets" / "metric_extension"
SRC = BASE / "UNIFIED_BASELINE_METRICS.csv"
OUT_MD = BASE / "BIG_MODEL_COMPARISON.md"
OUT_CSV = BASE / "BIG_MODEL_COMPARISON.csv"
OUT_MEAN = BASE / "BIG_MODEL_MEAN_SUMMARY.csv"

TRANSFERS = [
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
]

METHOD_ORDER = [
    "Ours",
    "AnomalyCLIP",
    "MRAD-TF",
    "MRAD-FT",
    "MRAD-CLIP",
    "VisualAD",
    "FB-CLIP",
]

METRICS = ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1MAX", "AUPRO"]
DISPLAY = {
    "I_AUROC": "I-AUROC",
    "I_AP": "I-AP",
    "P_AUROC": "P-AUROC",
    "P_AP": "P-AP",
    "P_F1MAX": "P-F1max",
    "AUPRO": "AUPRO",
}


def as_float(v) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x if np.isfinite(x) else float("nan")


def fmt(v: float, best: float | None = None) -> str:
    if not np.isfinite(v):
        return "NA"
    s = f"{v:.2f}"
    if best is not None and np.isfinite(best) and abs(v - best) <= 1e-9:
        return f"**{s}**"
    return s


def method_sort_key(method: str) -> tuple[int, str]:
    if method in METHOD_ORDER:
        return (METHOD_ORDER.index(method), method)
    return (999, method)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        alt = path.with_suffix(path.suffix + ".new")
        df.to_csv(alt, index=False)
        print(f"[big-comparison] {path.name} is locked; wrote {alt.name} instead")


def build() -> None:
    df = pd.read_csv(SRC)
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")

    rows = []
    lines = [
        "# Big Model Comparison",
        "",
        "Scope: same-protocol cached/evaluated models currently available in `UNIFIED_BASELINE_METRICS.csv`.",
        "",
        "Bold values mark the best available value for that metric within the same source→target cell. `NA` means the artifact did not contain that metric for that method/cell.",
        "",
        "Methods included: Ours, AnomalyCLIP, MRAD-TF, MRAD-FT, MRAD-CLIP, VisualAD, FB-CLIP.",
        "",
    ]

    for source, target in TRANSFERS:
        cell = df[(df["source"].eq(source)) & (df["target"].eq(target))].copy()
        if cell.empty:
            continue
        cell = cell.sort_values("method", key=lambda s: s.map(lambda x: method_sort_key(str(x))))
        best = {m: cell[m].max(skipna=True) for m in METRICS}
        lines += [
            f"## {source} -> {target}",
            "",
            "| Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in cell.iterrows():
            method = str(r["method"])
            line_vals = [fmt(as_float(r[m]), best[m]) for m in METRICS]
            lines.append(f"| {method} | " + " | ".join(line_vals) + " |")
            row = {"source": source, "target": target, "method": method}
            for m in METRICS:
                v = as_float(r[m])
                row[m] = v
                row[f"{m}_is_best"] = bool(np.isfinite(v) and np.isfinite(best[m]) and abs(v - best[m]) <= 1e-9)
            rows.append(row)
        lines.append("")

    big = pd.DataFrame(rows)
    safe_to_csv(big, OUT_CSV)

    summary_rows = []
    for method in sorted(df["method"].dropna().astype(str).unique(), key=method_sort_key):
        part = df[df["method"].eq(method)]
        row = {"method": method}
        for m in METRICS:
            vals = pd.to_numeric(part[m], errors="coerce")
            row[f"{m}_mean"] = float(vals.mean(skipna=True)) if vals.notna().any() else float("nan")
            row[f"{m}_n"] = int(vals.notna().sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    safe_to_csv(summary, OUT_MEAN)

    lines += [
        "## Mean Summary",
        "",
        "Means are computed over available same-protocol cells for each method. `n` reports available cells for that metric.",
        "",
        "| Method | I-AUROC mean/n | I-AP mean/n | P-AUROC mean/n | P-AP mean/n | P-F1max mean/n | AUPRO mean/n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    best_mean = {m: summary[f"{m}_mean"].max(skipna=True) for m in METRICS}
    for _, r in summary.iterrows():
        vals = []
        for m in METRICS:
            v = as_float(r[f"{m}_mean"])
            n = int(r[f"{m}_n"])
            vals.append(f"{fmt(v, best_mean[m])}/{n}" if np.isfinite(v) else f"NA/{n}")
        lines.append(f"| {r['method']} | " + " | ".join(vals) + " |")

    common_cells = [
        ("Full MVTec", "VisA"),
        ("Full MVTec", "BTAD"),
        ("Full MVTec", "MPDD"),
        ("Full VisA", "MVTec"),
        ("Full VisA", "BTAD"),
        ("Full VisA", "MPDD"),
    ]
    common_methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    common_rows = []
    for method in common_methods:
        part = df[(df["method"].eq(method)) & (df[["source", "target"]].apply(tuple, axis=1).isin(common_cells))]
        if len(part) != len(common_cells):
            continue
        row = {"method": method}
        for metric in METRICS:
            vals = pd.to_numeric(part[metric], errors="coerce")
            row[metric] = float(vals.mean(skipna=True)) if vals.notna().any() else float("nan")
        common_rows.append(row)
    common = pd.DataFrame(common_rows)
    if not common.empty:
        safe_to_csv(common, BASE / "BIG_MODEL_COMMON6_MEAN_SUMMARY.csv")
        common_best = {m: common[m].max(skipna=True) for m in METRICS}
        lines += [
            "",
            "## Common 6-Cell Mean Summary",
            "",
            "This mean excludes DTD-Synthetic so every listed method has the same six transfer cells.",
            "",
            "| Method | I-AUROC | I-AP | P-AUROC | P-AP | P-F1max | AUPRO |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for _, r in common.iterrows():
            vals = [fmt(as_float(r[m]), common_best[m]) for m in METRICS]
            lines.append(f"| {r['method']} | " + " | ".join(vals) + " |")

    lines += [
        "",
        "## Notes",
        "",
        "- DTD-Synthetic Ours rows now use the DTD-only full image+pixel completion artifact; several comparison-model DTD rows remain pixel-only where raw image metrics were not cached.",
        "- MRAD same-protocol local artifacts currently cover 6/8 industrial cells; DTD-Synthetic cells are not filled for MRAD.",
        "- Literature-reported numbers remain separate in `TABLE_REPORTED_REFERENCE.csv` and are not mixed into this same-protocol comparison.",
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[big-comparison] wrote {OUT_MD}")


if __name__ == "__main__":
    build()
