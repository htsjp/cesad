import json
from pathlib import Path

import yaml

from mrad_audit_utils import copy_log, load_config, parse_args, save_run_metadata


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_stats(root):
    root = Path(root)
    items = []
    for path in root.rglob("retrieval_stats.json"):
        try:
            items.append((path, load_json(path)))
        except Exception:
            continue
    return items


def render_report(cfg, stats_items):
    lines = [
        "# Retrieval Analysis Report",
        "",
        "This report aggregates saved retrieval statistics from the mechanism audit runs.",
        "",
        "## Inputs",
        "",
    ]
    for root in cfg.get("stat_roots", []):
        lines.append(f"- `{root}`")
    lines.extend(["", "## Summary", ""])
    if not stats_items:
        lines.append("No `retrieval_stats.json` files were found. Run the audit scripts first.")
        return "\n".join(lines) + "\n"

    for path, stats in stats_items:
        rel = path.parent.as_posix()
        lines.append(f"### {rel}")
        memory = stats.get("memory_size", {})
        lines.append(f"- Memory size: image={memory.get('image')}, patch={memory.get('patch')}")
        for prefix in ("image", "patch"):
            entropy = stats.get(f"{prefix}_entropy", {})
            margin = stats.get(f"{prefix}_margin", {})
            coverage = stats.get(f"{prefix}_coverage")
            label = stats.get(f"{prefix}_top1_label", {})
            lines.append(
                "- {prefix}: entropy_mean={ent}, margin_mean={margin}, coverage={coverage}, top1_label={label}".format(
                    prefix=prefix,
                    ent=entropy.get("mean") if isinstance(entropy, dict) else None,
                    margin=margin.get("mean") if isinstance(margin, dict) else None,
                    coverage=coverage,
                    label=label,
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Mechanism Reading Guide",
            "",
            "- Feature insufficiency is indicated when all memory variants have low margins and low AUROC, but layer changes recover performance.",
            "- Memory compression loss is indicated when token, KMeans, FPS, or statistics memory improves coverage/PRO over mean prototypes at comparable retrieval settings.",
            "- Retrieval strategy limitations are indicated when memory coverage is low, entropy is high, or a small set of entries dominates top-1 usage despite adequate layer/prototype performance.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args("Aggregate MRAD retrieval statistics")
    cfg = load_config(args.config, {"output_dir": args.output_dir, "seed": args.seed})
    output_dir = Path(cfg.get("output_dir", "analysis"))
    save_run_metadata(output_dir, cfg, int(cfg.get("seed", 0)))
    stats_items = []
    for root in cfg.get("stat_roots", []):
        stats_items.extend(collect_stats(root))
    report = render_report(cfg, stats_items)
    report_path = output_dir / "retrieval_report.md"
    report_path.write_text(report, encoding="utf-8")
    copy_log(output_dir, [f"Aggregated {len(stats_items)} retrieval stat files into {report_path}"])


if __name__ == "__main__":
    main()
