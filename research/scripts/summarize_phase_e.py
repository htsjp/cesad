from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rows = list(csv.DictReader((ROOT / "results" / "phase_e" / "formal_d1_d2.csv").open(newline="", encoding="utf-8")))
    for r in rows:
        print(
            r["direction"],
            r["variant"],
            f"I={float(r['image_auroc']):.6f}",
            f"dE0={float(r['delta_vs_E0_image_auroc']) * 100:+.2f}pt",
            f"dE1={float(r['delta_vs_E1_image_auroc']) * 100:+.2f}pt",
            f"AP={float(r['image_ap']):.6f}",
            f"rho={float(r['score_spearman_vs_E0']):.4f}",
            f"MAE={float(r['score_mae_vs_E0']):.6f}",
        )


if __name__ == "__main__":
    main()
