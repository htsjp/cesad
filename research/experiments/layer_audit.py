from pathlib import Path

from mrad_audit_utils import (
    build_layer_memory,
    copy_log,
    default_source_dataset,
    load_config,
    load_model,
    make_loader,
    parse_args,
    plot_layer_figures,
    run_visual_retrieval,
    save_anomaly_maps,
    save_metrics,
    save_run_metadata,
    set_seed,
    write_csv,
)


def main():
    args = parse_args("MRAD multi-layer CLIP feature audit")
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

    output_dir = Path(cfg.get("output_dir", "results/layer_audit"))
    save_run_metadata(output_dir, cfg, int(cfg.get("seed", 0)))
    logs = []

    model = load_model(cfg, cfg.get("device", "cuda:0"))
    _, source_loader = make_loader(cfg, cfg["source_dataset"], mode=cfg.get("source_mode", "test"), shuffle=False)
    _, target_loader = make_loader(cfg, cfg["dataset"], mode=cfg.get("target_mode", "test"), shuffle=False)

    rows = []
    all_metrics = {}
    all_retrieval = {}
    for layer in cfg["layers"]:
        layer_dir = output_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        logs.append(f"Layer {layer}: building memory from {cfg['source_dataset']} and evaluating on {cfg['dataset']}")
        memory = build_layer_memory(model, source_loader, cfg, int(layer), cfg.get("device", "cuda:0"))
        metrics, retrieval, maps = run_visual_retrieval(
            model, memory, target_loader, cfg, int(layer), cfg.get("device", "cuda:0")
        )
        save_metrics(layer_dir, metrics, retrieval)
        save_anomaly_maps(layer_dir, maps)
        all_metrics[str(layer)] = metrics
        all_retrieval[str(layer)] = retrieval
        mean = metrics["mean"]
        row = {
            "Layer": int(layer),
            "I-AUROC": mean["I-AUROC"],
            "I-AP": mean["I-AP"],
            "P-AUROC": mean["P-AUROC"],
            "PRO": mean.get("PRO"),
            "Memory Size": retrieval["memory_size"]["image"] + retrieval["memory_size"]["patch"],
        }
        rows.append(row)
        logs.append(
            "Layer {layer}: I-AUROC={ia:.4f}, I-AP={iap:.4f}, P-AUROC={pa:.4f}, PRO={pro:.4f}".format(
                layer=layer,
                ia=row["I-AUROC"],
                iap=row["I-AP"],
                pa=row["P-AUROC"],
                pro=row["PRO"] if row["PRO"] is not None else float("nan"),
            )
        )

    write_csv(output_dir / "layer_metrics.csv", rows, ["Layer", "I-AUROC", "I-AP", "P-AUROC", "PRO", "Memory Size"])
    save_metrics(output_dir, {"layers": all_metrics, "summary_rows": rows}, all_retrieval)
    plot_layer_figures(output_dir, rows)
    copy_log(output_dir, logs)


if __name__ == "__main__":
    main()
