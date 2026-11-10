from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/graph_reconstruction_audit/all_results.csv")
    parser.add_argument("--out", default="results/graph_reconstruction_audit/group_comparison.json")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.results, newline="", encoding="utf-8")))
    by_direction = {}
    for row in rows:
        by_direction.setdefault(row["direction"], {})[row["group"]] = row
    comparisons = {}
    for direction, groups in by_direction.items():
        g5 = groups.get("G5", {})
        comparisons[direction] = {"has_required_groups": all(g in groups for g in ["G0", "G1", "G2", "G3", "G4", "G5"])}
        for baseline in ["G0", "G1", "G2", "G3", "G4"]:
            if baseline in groups and g5:
                comparisons[direction][f"G5_minus_{baseline}_image_auroc"] = float(g5["image_auroc"]) - float(groups[baseline]["image_auroc"])
                comparisons[direction][f"G5_minus_{baseline}_unique_selected"] = int(g5["unique_selected_prototypes"]) - int(groups[baseline]["unique_selected_prototypes"])
                comparisons[direction][f"G5_minus_{baseline}_gini"] = float(g5["prototype_usage_gini"]) - float(groups[baseline]["prototype_usage_gini"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(comparisons, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
