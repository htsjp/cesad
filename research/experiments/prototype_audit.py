from pathlib import Path

import torch
import math

from mrad_audit_utils import (
    build_prototype_memory,
    copy_log,
    default_source_dataset,
    load_config,
    load_model,
    make_loader,
    parse_args,
    run_visual_retrieval,
    save_anomaly_maps,
    save_metrics,
    save_run_metadata,
    set_seed,
    write_csv,
)


def memory_gpu_mb(memory):
    total = 0
    for key in ("image_keys", "image_values", "patch_keys", "patch_values"):
        t = memory[key]
        total += t.numel() * t.element_size()
    return total / (1024 * 1024)


def estimate_token_memory(loader):
    side_tokens = 37 * 37
    entries = len(loader.dataset) * side_tokens
    key_gb = entries * 768 * 4 / (1024 ** 3)
    return entries, key_gb


def main():
    args = parse_args("MRAD region prototype compression audit")
    cfg = load_config(
        args.config,
        {
            "dataset": args.dataset,
            "source_dataset": args.source_dataset,
            "device": args.device,
            "output_dir": args.output_dir,
            "seed": args.seed,
            "limit": args.limit,
        },
    )
    cfg["source_dataset"] = cfg.get("source_dataset") or default_source_dataset(cfg["dataset"])
    set_seed(int(cfg.get("seed", 0)))

    output_dir = Path(cfg.get("output_dir", "results/prototype_audit"))
    save_run_metadata(output_dir, cfg, int(cfg.get("seed", 0)))
    logs = []

    model = load_model(cfg, cfg.get("device", "cuda:0"))
    _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)

    rows = []
    all_metrics = {}
    all_retrieval = {}
    for spec in cfg["memory_types"]:
        memory_type = spec["type"]
        run_cfg = dict(cfg)
        if "k" in spec:
            run_cfg["k"] = int(spec["k"])
        label = spec.get("name") or (f"{memory_type}_k{run_cfg.get('k')}" if memory_type in ("kmeans", "fps") else memory_type)
        run_dir = output_dir / label
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        retrieval_path = run_dir / "retrieval_stats.json"
        if cfg.get("resume_existing", True) and metrics_path.exists():
            import json
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            retrieval = {}
            if retrieval_path.exists():
                with open(retrieval_path, "r", encoding="utf-8") as f:
                    retrieval = json.load(f)
            mean = metrics.get("mean", {})
            memory_size = retrieval.get("memory_size", {})
            row = {
                "Memory Type": label,
                "Entries": int(memory_size.get("image", 0) + memory_size.get("patch", 0)),
                "I-AUROC": mean.get("I-AUROC", float("nan")),
                "P-AUROC": mean.get("P-AUROC", float("nan")),
                "PRO": mean.get("PRO", float("nan")),
                "GPU Memory": float("nan"),
            }
            rows.append(row)
            all_metrics[label] = metrics
            all_retrieval[label] = retrieval
            logs.append(f"{label}: skipped existing result at {metrics_path}")
            continue
        if memory_type == "token":
            entries, key_gb = estimate_token_memory(source_loader)
            max_entries = int(cfg.get("max_exact_token_entries", 200000))
            if entries > max_entries:
                row = {
                    "Memory Type": label,
                    "Entries": entries,
                    "I-AUROC": float("nan"),
                    "P-AUROC": float("nan"),
                    "PRO": float("nan"),
                    "GPU Memory": key_gb * 1024,
                }
                rows.append(row)
                note = {
                    "status": "skipped_infeasible_exact_retrieval",
                    "reason": "Exact full-token softmax retrieval would require per-image similarity over all source patch tokens.",
                    "estimated_entries": entries,
                    "estimated_key_memory_gb_fp32": key_gb,
                    "max_exact_token_entries": max_entries,
                }
                save_metrics(run_dir, {"mean": note, "per_class": {}}, {"memory_size": {"image": 0, "patch": entries}})
                all_metrics[label] = note
                all_retrieval[label] = note
                logs.append(f"{label}: skipped exact full-token retrieval; entries={entries}, key_memory_fp32={key_gb:.2f}GB")
                continue
        logs.append(f"{label}: building memory from {cfg['source_dataset']} and evaluating on {cfg['dataset']}")
        memory = build_prototype_memory(model, source_loader, run_cfg, memory_type, cfg.get("device", "cuda:0"))
        metrics, retrieval, maps = run_visual_retrieval(
            model, memory, target_loader, run_cfg, int(run_cfg.get("layer", 24)), cfg.get("device", "cuda:0")
        )
        save_metrics(run_dir, metrics, retrieval)
        save_anomaly_maps(run_dir, maps)
        all_metrics[label] = metrics
        all_retrieval[label] = retrieval
        mean = metrics["mean"]
        row = {
            "Memory Type": label,
            "Entries": int(memory["image_keys"].shape[0] + memory["patch_keys"].shape[0]),
            "I-AUROC": mean["I-AUROC"],
            "P-AUROC": mean["P-AUROC"],
            "PRO": mean.get("PRO"),
            "GPU Memory": memory_gpu_mb(memory),
        }
        rows.append(row)
        logs.append(
            "{label}: entries={entries}, I-AUROC={ia:.4f}, P-AUROC={pa:.4f}, PRO={pro:.4f}".format(
                label=label,
                entries=row["Entries"],
                ia=row["I-AUROC"],
                pa=row["P-AUROC"],
                pro=row["PRO"] if row["PRO"] is not None else float("nan"),
            )
        )
        torch.cuda.empty_cache()

    write_csv(output_dir / "prototype_metrics.csv", rows, ["Memory Type", "Entries", "I-AUROC", "P-AUROC", "PRO", "GPU Memory"])
    save_metrics(output_dir, {"prototype_runs": all_metrics, "summary_rows": rows}, all_retrieval)
    copy_log(output_dir, logs)


if __name__ == "__main__":
    main()
