from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "results" / "paper_assets" / "final_audit" / "VISA_CATEGORY_FINAL.csv"
OUT_CSV = ROOT / "results" / "paper_assets" / "final_audit" / "fig5_visa_category_analysis.csv"
FIG_DIR = ROOT / "results" / "paper_assets" / "journal_figures"
OUT_PNG = FIG_DIR / "fig5_visa_category_analysis.png"
OUT_PDF = FIG_DIR / "fig5_visa_category_analysis.pdf"

EXPECTED_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]

EXPECTED_RHO = {
    "fn_ratio": -0.7902,
    "thinness": -0.4615,
    "feature_shift": -0.0140,
}


def load_fig5_data() -> pd.DataFrame:
    src = pd.read_csv(SOURCE_CSV)
    missing = sorted(set(EXPECTED_CATEGORIES) - set(src["category"].astype(str)))
    extra = sorted(set(src["category"].astype(str)) - set(EXPECTED_CATEGORIES))
    if missing or extra:
        raise RuntimeError(f"Unexpected VisA categories. missing={missing}, extra={extra}")

    df = pd.DataFrame(
        {
            "category": src["category"].astype(str),
            "p_ap": src["Core P-AP"].astype(float),
            "fn_ratio": src["Core FN ratio"].astype(float),
            "thinness": src["thinness"].astype(float),
            "feature_shift": src["CLIP feature shift"].astype(float),
        }
    )
    df = df.set_index("category").loc[EXPECTED_CATEGORIES].reset_index()
    if df["p_ap"].max() <= 1.0:
        df["p_ap"] *= 100.0
    return df


def compute_rhos(df: pd.DataFrame) -> dict[str, float]:
    return {
        key: float(spearmanr(df["p_ap"], df[key], nan_policy="omit").statistic)
        for key in EXPECTED_RHO
    }


def compute_pvalues(df: pd.DataFrame) -> dict[str, float]:
    return {
        key: float(spearmanr(df["p_ap"], df[key], nan_policy="omit").pvalue)
        for key in EXPECTED_RHO
    }


def audit_or_fail(df: pd.DataFrame, rhos: dict[str, float]) -> None:
    print("=== FIG 5 DATA AUDIT ===")
    print(f"Number of categories: {len(df)}")
    print("")
    print("Spearman:")
    print(f"P-AP vs FN ratio = {rhos['fn_ratio']:.12f}")
    print("Expected = -0.7902")
    print("")
    print(f"P-AP vs thinness = {rhos['thinness']:.12f}")
    print("Expected = -0.4615")
    print("")
    print(f"P-AP vs feature shift = {rhos['feature_shift']:.12f}")
    print("Expected = -0.0140")
    print("")
    print("Variant provenance:")
    print("P-AP = CESAD-Core / E0_Core_NTR_FixedLRP, Full MVTec -> VisA category-level full-resolution pixel AP")
    print("FN ratio = CESAD-Core / E0_Core_NTR_FixedLRP, pooled category pixels at category F1-optimal threshold")
    print("Thinness = dataset GT morphology statistic, not model output")
    print("Feature shift = post-hoc CLIP/MRAD-wrapper patch-feature shift to source all-patch centroid")
    print("")
    print("Source files:")
    print(SOURCE_CSV)
    print(ROOT / "results" / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "VISA_CATEGORY_AUDIT.csv")
    print(ROOT / "experiments" / "build_paper_p0_audit.py")
    print(ROOT / "experiments" / "phase_mva_mrad_same_protocol_visa_factorization.py")
    print("")

    failed = []
    for key, expected in EXPECTED_RHO.items():
        if abs(rhos[key] - expected) > 1e-4:
            failed.append((key, rhos[key], expected))
    if failed:
        print("FIG5 DATA VALIDATION FAILED")
        for key, got, expected in failed:
            print(f"{key}: computed={got:.12f}, expected={expected:.12f}")
        raise SystemExit(2)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.16, color="#9a9a9a")
    ax.tick_params(axis="both", labelsize=9.5, width=0.8, length=3.5)
    ax.set_axisbelow(True)


def add_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    x_label: str,
    panel: str,
    rho: float,
    pvalue: float,
) -> None:
    x = df[x_col].to_numpy(dtype=float)
    y = df["p_ap"].to_numpy(dtype=float)

    ax.scatter(
        x,
        y,
        s=54,
        color="#4C78A8",
        alpha=0.92,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )

    ax.set_xlabel(x_label, fontsize=10.5)
    ax.set_ylabel("CESAD-Core category P-AP (%)", fontsize=10.5)
    ax.set_ylim(0, 65)
    x_pad = (float(np.nanmax(x)) - float(np.nanmin(x))) * 0.10
    ax.set_xlim(float(np.nanmin(x)) - x_pad, float(np.nanmax(x)) + x_pad)
    ax.text(0.02, 0.98, panel, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top", ha="left")
    ax.text(
        0.98,
        0.96,
        f"Spearman $\\rho$ = {rho:.3f}, $p$ = {pvalue:.3f}",
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        ha="right",
        color="#222222",
    )
    style_axes(ax)


def make_figure(df: pd.DataFrame, rhos: dict[str, float]) -> None:
    pvalues = compute_pvalues(df)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.7), sharey=True)
    add_scatter_panel(
        axes[0],
        df,
        "fn_ratio",
        "Category false-negative ratio",
        "(a)",
        rhos["fn_ratio"],
        pvalues["fn_ratio"],
    )
    add_scatter_panel(
        axes[1],
        df,
        "thinness",
        "Category anomaly thinness",
        "(b)",
        rhos["thinness"],
        pvalues["thinness"],
    )

    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print("")
    print("Saved:")
    print(OUT_CSV)
    print(OUT_PNG)
    print(OUT_PDF)
    print("")
    print("Suggested caption:")
    print(
        "Fig. 5. Category-level analysis of CESAD on MVTec→VisA. Each point represents one VisA category. "
        "Category P-AP is negatively correlated with (a) the false-negative ratio and (b) anomaly thinness. "
        "Spearman correlations are reported in each panel. The analysis is descriptive and does not imply causality."
    )


def main() -> None:
    df = load_fig5_data()
    rhos = compute_rhos(df)
    audit_or_fail(df, rhos)
    make_figure(df, rhos)


if __name__ == "__main__":
    main()
