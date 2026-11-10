import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv

OUT = ROOT / "results" / "phase_xdg_extended_generalization" / "fill"

BASE_PAIRS = (
    ("Full VisA", "MVTec"),
    ("Full VisA", "BTAD"),
    ("Full VisA", "MPDD"),
    ("Full VisA", "DTD-Synthetic"),
    ("Full VisA", "CVC-ClinicDB"),
    ("Full VisA", "CVC-ColonDB"),
    ("Full VisA", "Kvasir"),
    ("Full VisA", "ISBI"),
    ("Full MVTec", "VisA"),
    ("Full MVTec", "BTAD"),
    ("Full MVTec", "MPDD"),
    ("Full MVTec", "DTD-Synthetic"),
    ("Full MVTec", "CVC-ClinicDB"),
    ("Full MVTec", "CVC-ColonDB"),
    ("Full MVTec", "Kvasir"),
    ("Full MVTec", "ISBI"),
)

METHOD_TO_TABLE_NAME = {
    "ours": "Ours",
    "anomalyclip": "AnomalyCLIP",
    "visualad": "VisualAD",
    "fbclip": "FB-CLIP",
}


def json_default(obj):
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=json_default), encoding="utf-8")


def read_csv(path):
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fnum(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def pair_key(source, target):
    return f"{source}->{target}"


def available_pairs(smoke=False):
    from experiments.phase_final_cross_dataset_benchmark import protocol_cells

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ok = set()
    for source in ("Full VisA", "Full MVTec"):
        cells, _audits = protocol_cells(source, device, smoke)
        names = {target for _cid, target, _dataset, _refs, _patch, _kind in cells}
        for source2, target in BASE_PAIRS:
            if source2 == source and target in names:
                ok.add((source2, target))
    return [p for p in BASE_PAIRS if p in ok]


def filter_pairs(requested, smoke=False):
    if requested:
        wanted = set()
        for item in requested:
            if "->" not in item:
                raise ValueError(f"pair must be Source->Target, got {item}")
            s, t = item.split("->", 1)
            wanted.add((s.strip(), t.strip()))
        return [p for p in BASE_PAIRS if p in wanted]
    return list(BASE_PAIRS)


def rows_done(path, variant):
    done = set()
    for r in read_csv(path):
        if r.get("variant") == variant:
            done.add((r.get("source"), r.get("dataset")))
    return done


def load_protocol_cell(source, target, device, smoke):
    from experiments.phase_final_cross_dataset_benchmark import protocol_cells

    cells, _audits = protocol_cells(source, device, smoke)
    for cell_id, tname, dataset, refs, patch, kind in cells:
        if tname == target:
            return cell_id, dataset, refs, patch, kind
    raise KeyError(f"No protocol cell for {source}->{target}")


def run_ours(pairs, args):
    import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
    from experiments.phase_parameter_free_local_zoom_pyramid_fullres import (
        CHECKPOINT,
        VARIANTS as LZP_VARIANTS,
        build_maps as lzp_build_maps,
        evaluate_maps as lzp_evaluate_maps,
        load_components,
        load_mvtec_source,
        load_visa_source,
    )
    from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
    from utils.transforms import get_transform

    raw_path = OUT / "RAW_METRICS_ours.csv"
    per_path = OUT / "PER_CATEGORY_ours.csv"
    scale_path = OUT / "SCORE_SCALE_ours.csv"
    raw, percat, scale = read_csv(raw_path), read_csv(per_path), read_csv(scale_path)
    done = rows_done(raw_path, "E2_GlobalLocalZoomPyramid")
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    models = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    old_variants = lzp.VARIANTS
    lzp.VARIANTS = LZP_VARIANTS
    try:
        for source, target in pairs:
            if (source, target) in done and not args.force:
                print(f"[XDGFill][Ours] skip existing {source}->{target}", flush=True)
                continue
            _cell_id, dataset, refs, patch, kind = load_protocol_cell(source, target, device, args.smoke)
            n = min(24, len(refs)) if args.smoke else len(refs)
            print(f"[XDGFill][Ours] {source}->{target} n={n}", flush=True)
            maps, metas, scale_rows = lzp_build_maps(models[source], clip_model, preprocess, patch[:n], refs[:n], dataset, kind, target, device)
            rr, cc = lzp_evaluate_maps(source, target, maps, metas)
            raw = [r for r in raw if not (r.get("source") == source and r.get("dataset") == target)]
            percat = [r for r in percat if not (r.get("source") == source and r.get("dataset") == target)]
            scale = [r for r in scale if not (r.get("source") == source and r.get("dataset") == target)]
            raw.extend(rr)
            percat.extend(cc)
            scale.extend(scale_rows)
            write_csv(raw_path, raw)
            write_csv(per_path, percat)
            write_csv(scale_path, scale)
    finally:
        lzp.VARIANTS = old_variants
    return raw


def run_anomalyclip(pairs, args):
    import experiments.phase_anomalyclip_same_protocol_audit as ac

    raw_path = OUT / "RAW_METRICS_anomalyclip.csv"
    per_path = OUT / "PER_CATEGORY_anomalyclip.csv"
    raw, percat = read_csv(raw_path), read_csv(per_path)
    done = rows_done(raw_path, ac.VARIANT)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    by_source = {}
    for source, target in pairs:
        by_source.setdefault(source, []).append(target)
    for source, targets in by_source.items():
        if not ac.WEIGHTS[source].exists():
            print(f"[XDGFill][AnomalyCLIP] missing checkpoint {source}: {ac.WEIGHTS[source]}", flush=True)
            continue
        print(f"[XDGFill][AnomalyCLIP] load {source}", flush=True)
        model, prompt, text_features, ac_args = ac.load_model(source, device, smoke=args.smoke)
        if args.batch_size:
            ac_args.test_batch_size = args.batch_size
        for target in targets:
            if (source, target) in done and not args.force:
                print(f"[XDGFill][AnomalyCLIP] skip existing {source}->{target}", flush=True)
                continue
            print(f"[XDGFill][AnomalyCLIP] {source}->{target}", flush=True)
            rr, cc = ac.evaluate_pair(source, target, model, text_features, ac_args, device, args.smoke)
            raw = [r for r in raw if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == ac.VARIANT)]
            percat = [r for r in percat if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == ac.VARIANT)]
            raw.extend(rr)
            percat.extend(cc)
            write_csv(raw_path, raw)
            write_csv(per_path, percat)
        del model, prompt, text_features
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return raw


def run_visualad(pairs, args):
    import experiments.phase_visualad_exact_protocol_audit as va

    raw_path = OUT / "RAW_METRICS_visualad.csv"
    per_path = OUT / "PER_CATEGORY_visualad.csv"
    att_path = OUT / "ATTENTION_visualad.csv"
    ck_path = OUT / "CHECKSUM_visualad.csv"
    raw, percat, att, ck = read_csv(raw_path), read_csv(per_path), read_csv(att_path), read_csv(ck_path)
    done = rows_done(raw_path, "E1c_FullVisualAD")
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    official = va.ensure_official_imports()
    for source, target in pairs:
        if (source, target) in done and not args.force:
            print(f"[XDGFill][VisualAD] skip existing {source}->{target}", flush=True)
            continue
        print(f"[XDGFill][VisualAD] {source}->{target} E1c_FullVisualAD", flush=True)
        rr, cc, aa, kk = va.build_visualad_maps(source, target, "E1c_FullVisualAD", official, device, args.smoke, args.batch_size or 2)
        raw = [r for r in raw if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == "E1c_FullVisualAD")]
        percat = [r for r in percat if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == "E1c_FullVisualAD")]
        att = [r for r in att if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == "E1c_FullVisualAD")]
        ck = [r for r in ck if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == "E1c_FullVisualAD")]
        raw.extend(rr)
        percat.extend(cc)
        att.extend(aa)
        ck.extend(kk)
        write_csv(raw_path, raw)
        write_csv(per_path, percat)
        write_csv(att_path, att)
        write_csv(ck_path, ck)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return raw


def run_fbclip(pairs, args):
    import experiments.phase_fbclip_pretrained_headroom_audit as fb

    raw_path = OUT / "RAW_METRICS_fbclip.csv"
    per_path = OUT / "PER_CATEGORY_fbclip.csv"
    raw, percat = read_csv(raw_path), read_csv(per_path)
    done = rows_done(raw_path, fb.VARIANT)
    device = "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    by_source = {}
    for source, target in pairs:
        by_source.setdefault(source, []).append(target)
    for source, targets in by_source.items():
        if not fb.WEIGHTS[source].exists():
            print(f"[XDGFill][FBCLIP] missing checkpoint {source}: {fb.WEIGHTS[source]}", flush=True)
            continue
        print(f"[XDGFill][FBCLIP] load {source}", flush=True)
        model, prompt, fb_args, _info = fb.load_fb_model(source, device)
        if args.batch_size:
            fb_args.test_batch_size = args.batch_size
        for target in targets:
            if (source, target) in done and not args.force:
                print(f"[XDGFill][FBCLIP] skip existing {source}->{target}", flush=True)
                continue
            print(f"[XDGFill][FBCLIP] {source}->{target}", flush=True)
            rr, cc = fb.evaluate_fb(source, target, model, prompt, fb_args, device, args.smoke)
            raw = [r for r in raw if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == fb.VARIANT)]
            percat = [r for r in percat if not (r.get("source") == source and r.get("dataset") == target and r.get("variant") == fb.VARIANT)]
            raw.extend(rr)
            percat.extend(cc)
            write_csv(raw_path, raw)
            write_csv(per_path, percat)
        del model, prompt
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return raw


def summarize():
    rows = []
    for method, table_name in METHOD_TO_TABLE_NAME.items():
        raw_path = OUT / f"RAW_METRICS_{method}.csv"
        for r in read_csv(raw_path):
            variant = r.get("variant", "")
            if method == "ours" and variant != "E2_GlobalLocalZoomPyramid":
                continue
            if method == "anomalyclip" and variant != "AnomalyCLIP_OfficialCheckpoint":
                continue
            if method == "visualad" and variant != "E1c_FullVisualAD":
                continue
            if method == "fbclip" and variant != "E1_Official_FBCLIP_Pretrained":
                continue
            rows.append({
                "method": table_name,
                "source": r.get("source"),
                "dataset": r.get("dataset"),
                "variant": variant,
                "M_FR_P_AUROC": fnum(r, "M_FR_P_AUROC"),
                "M_FR_P_AP": fnum(r, "M_FR_P_AP"),
                "M_FR_P_F1max": fnum(r, "M_FR_P_F1max"),
                "M_FR_PRO": fnum(r, "M_FR_PRO"),
            })
    write_csv(OUT / "FILLED_COMPARISON_SUMMARY.csv", rows)
    return rows


def write_report(summary_rows, runtime, pairs, methods):
    lines = [
        "# XDG Comparison Fill Run",
        "",
        f"Runtime: {runtime:.1f}s",
        "",
        f"Methods requested: `{', '.join(methods)}`",
        "",
        "## Filled Cells",
        "",
        "| Method | Source | Target | P-AUROC | P-AP | P-F1max | PRO |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in sorted(summary_rows, key=lambda x: (x["method"], x["source"], x["dataset"])):
        lines.append(
            f"| {r['method']} | {r['source']} | {r['dataset']} | "
            f"{fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | "
            f"{fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |"
        )
    lines += [
        "",
        "## Protocol Notes",
        "",
        "- No target training, target statistics, target tuning, or threshold tuning was added.",
        "- `Ours` uses the frozen Canonical Two-State + NTR + Fixed LRP path.",
        "- Official comparison wrappers are evaluated through the same full-resolution evaluator when their checkpoint/code path can be applied.",
        "- MRAD official wrapper is not extended here because its current same-protocol implementation has dataset-specific loaders for MVTec/VisA/BTAD/MPDD; unsupported cells are left blank rather than fabricated.",
        "",
        "Requested pairs:",
    ]
    for source, target in pairs:
        lines.append(f"- {source} -> {target}")
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["anomalyclip", "visualad", "fbclip"], choices=["ours", "anomalyclip", "visualad", "fbclip"])
    p.add_argument("--pairs", nargs="*", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = filter_pairs(args.pairs, args.smoke)
    print(f"[XDGFill] pairs={len(pairs)} models={args.models}", flush=True)
    for method in args.models:
        if method == "ours":
            run_ours(pairs, args)
        elif method == "anomalyclip":
            run_anomalyclip(pairs, args)
        elif method == "visualad":
            run_visualad(pairs, args)
        elif method == "fbclip":
            run_fbclip(pairs, args)
    summary_rows = summarize()
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {"outcome": "XDG_COMPARISON_FILL_COMPLETE", "methods": args.models, "pairs": [pair_key(*p) for p in pairs], "filled_rows": len(summary_rows), "runtime_sec": runtime})
    write_report(summary_rows, runtime, pairs, args.models)
    print(json.dumps({"outcome": "XDG_COMPARISON_FILL_COMPLETE", "filled_rows": len(summary_rows), "runtime_sec": runtime}, indent=2), flush=True)


if __name__ == "__main__":
    main()
