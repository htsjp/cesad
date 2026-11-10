from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "results" / "phase_d" / "formal_d1_d2.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        print(
            row["direction"],
            row["variant"],
            f"I={float(row['image_auroc']):.6f}",
            f"dI={float(row['delta_image_auroc']) * 100:+.2f}pt",
            f"AP={float(row['image_ap']):.6f}",
            f"dAP={float(row['delta_image_ap']) * 100:+.2f}pt",
            f"rho={float(row['score_spearman_vs_full']):.4f}",
        )


if __name__ == "__main__":
    main()
