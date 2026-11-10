from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "results" / "paper_assets" / "final_audit" / "VISA_CATEGORY_FINAL.csv"
OUT_DIR = ROOT / "results" / "paper_assets" / "journal_figures"
OUT_CSV = OUT_DIR / "fig5_main.csv"
OUT_PNG = OUT_DIR / "fig5_main.png"
OUT_PDF = OUT_DIR / "fig5_main.pdf"
SCRIPT_PATH = Path(__file__).resolve()

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
EXPECTED_RHO = -0.7902


def load_data() -> pd.DataFrame:
    raw = pd.read_csv(SOURCE_CSV)
    categories = set(raw["category"].astype(str))
    missing = sorted(set(EXPECTED_CATEGORIES) - categories)
    extra = sorted(categories - set(EXPECTED_CATEGORIES))
    if missing or extra:
        raise RuntimeError(f"Unexpected category set. missing={missing}, extra={extra}")

    df = pd.DataFrame(
        {
            "category": raw["category"].astype(str),
            "p_ap": raw["Core P-AP"].astype(float),
            "fn_ratio": raw["Core FN ratio"].astype(float),
            "thinness": raw["thinness"].astype(float),
        }
    )
    df = df.set_index("category").loc[EXPECTED_CATEGORIES].reset_index()
    if df["p_ap"].max() <= 1.0:
        df["p_ap"] *= 100.0
    return df


def validate(df: pd.DataFrame) -> tuple[float, float]:
    result = spearmanr(df["p_ap"], df["fn_ratio"], nan_policy="omit")
    rho = float(result.statistic)
    pvalue = float(result.pvalue)
    if abs(rho - EXPECTED_RHO) > 1e-4:
        print("FIG5 VALIDATION FAILED")
        print(f"Computed rho = {rho:.12f}")
        print(f"Expected rho = {EXPECTED_RHO:.12f}")
        raise SystemExit(2)
    return rho, pvalue


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.20, color="#9a9a9a")
    ax.tick_params(axis="both", labelsize=9.5, width=0.8, length=3.5)
    ax.set_axisbelow(True)


def plot(df: pd.DataFrame, rho: float, pvalue: float) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    color = "#4C78A8"
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # (a) Category-wise P-AP
    ax = axes[0]
    order = df.sort_values("p_ap", ascending=True)
    y = np.arange(len(order))
    ax.barh(y, order["p_ap"], color=color, alpha=0.92, edgecolor="white", linewidth=0.6, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(order["category"], fontsize=9.3)
    ax.set_xlabel("Category P-AP (%)", fontsize=10.5)
    ax.set_title("Category-wise P-AP", fontsize=10.8, pad=8)
    ax.set_xlim(0, 66)
    ax.text(0.01, 0.98, "(a)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
    for yi, val in zip(y, order["p_ap"]):
        ax.text(float(val) + 1.0, yi, f"{val:.1f}", va="center", ha="left", fontsize=8.5, color="#333333")
    style_axis(ax)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.20, color="#9a9a9a")
    ax.grid(False, axis="y")

    # (b) P-AP vs. False-Negative Ratio
    ax = axes[1]
    x = df["fn_ratio"].to_numpy(dtype=float)
    yv = df["p_ap"].to_numpy(dtype=float)
    ax.scatter(x, yv, s=56, color=color, alpha=0.92, edgecolor="white", linewidth=0.6, zorder=3)
    for cat, offset in {
        "macaroni2": (5, 7),
        "pcb3": (6, 4),
        "chewinggum": (6, 6),
    }.items():
        row = df[df["category"] == cat]
        ax.annotate(
            cat,
            xy=(float(row["fn_ratio"].iloc[0]), float(row["p_ap"].iloc[0])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8.5,
            color="#333333",
            ha="left",
            va="bottom",
        )
    ax.set_xlabel("False-Negative Ratio", fontsize=10.5)
    ax.set_ylabel("Category P-AP (%)", fontsize=10.5)
    ax.set_title("P-AP vs. False-Negative Ratio", fontsize=10.8, pad=8)
    ax.set_ylim(0, 66)
    xpad = (float(np.nanmax(x)) - float(np.nanmin(x))) * 0.10
    ax.set_xlim(float(np.nanmin(x)) - xpad, float(np.nanmax(x)) + xpad)
    ax.text(0.01, 0.98, "(b)", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")
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
    style_axis(ax)

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load_data()
    rho, pvalue = validate(df)
    plot(df, rho, pvalue)
    print("=== FIG 5 AUDIT ===")
    print(f"Data source: {SOURCE_CSV}")
    print("Variant: CESAD-Core")
    print(f"Number of categories: {len(df)}")
    print(f"Spearman(P-AP, FN ratio) = {rho:.12f}, p = {pvalue:.12f}")
    print("Expected ≈ -0.7902")
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved PNG: {OUT_PNG}")
    print(f"Saved PDF: {OUT_PDF}")
    print(f"Saved script: {SCRIPT_PATH}")
    print("")
    print("Suggested caption:")
    print(
        "Fig. 5. Category-level analysis of CESAD-Core on MVTec→VisA. "
        "(a) P-AP for the 12 VisA categories. "
        "(b) Relationship between category P-AP and false-negative ratio. "
        "Each point denotes one category."
    )


if __name__ == "__main__":
    main()
