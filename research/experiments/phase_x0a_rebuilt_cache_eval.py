import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_x0a_mandatory_gates import OUT, evaluate_safe_identity, write_csv


def main():
    device = "cuda:0"
    cache_root = OUT / "safe_rebuilt_cache"
    rows, costs = [], []
    for cfg in [
        ("D1_rebuilt_cache", "visa", "data/VISA", "checkpoints/test_on_visa.pth", str(cache_root), "mvtec", [85.20, 88.00, 95.80, 88.00]),
        ("D2_rebuilt_cache", "mvtec", "data/mvtec", "checkpoints/test_on_mvtec.pth", str(cache_root), "visa", [93.90, 97.30, 92.90, 86.80]),
    ]:
        metrics, _, cost = evaluate_safe_identity(*cfg[:6], device)
        mean = metrics[-1]
        ref = cfg[6]
        mean["ref_image_auroc"] = ref[0]
        mean["ref_image_ap"] = ref[1]
        mean["ref_pixel_auroc"] = ref[2]
        mean["ref_pixel_aupro"] = ref[3]
        mean["delta_image_auroc"] = mean["image_auroc"] - ref[0]
        mean["delta_image_ap"] = mean["image_ap"] - ref[1]
        mean["delta_pixel_auroc"] = mean["pixel_auroc"] - ref[2]
        mean["delta_pixel_aupro"] = mean["pixel_aupro"] - ref[3]
        rows.append(mean)
        costs.append(cost)
        write_csv(OUT / "cache_parity_running.csv", rows)
    write_csv(OUT / "cache_parity.csv", rows)
    existing_costs = []
    path = OUT / "computational_costs.csv"
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            existing_costs = list(csv.DictReader(f))
    write_csv(path, existing_costs + costs)


if __name__ == "__main__":
    main()
