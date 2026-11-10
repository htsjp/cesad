import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.phase_parameter_free_local_zoom_pyramid_fullres as lzp
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE
from utils.transforms import get_transform


OUT = ROOT / "results" / "paper_assets" / "metric_extension" / "dtd_completion"


def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_path = OUT / "DTD_ONLY_FULL_METRICS.csv"
    cat_path = OUT / "DTD_ONLY_PER_CATEGORY_FULL_RES.csv"
    if raw_path.exists() and cat_path.exists():
        print(f"[DTDComplete] existing outputs found: {raw_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DTDComplete] device={device}", flush=True)
    preprocess, _ = get_transform(__import__("argparse").Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)

    models = [lzp.load_visa_source(device), lzp.load_mvtec_source()]
    raw, cats, loc, stitch, scale = [], [], [], [], []
    for model_info in models:
        cells, _audits = lzp.protocol_cells(model_info["source"], device, False)
        for cell in cells:
            _cell_id, target, dataset, idx_or_rows, patch, kind = cell
            if target != "DTD-Synthetic":
                continue
            refs = idx_or_rows[: patch.shape[0]]
            print(f"[DTDComplete] evaluate {model_info['source']} -> {target} n={len(refs)}", flush=True)
            maps, metas, scale_rows = lzp.build_maps(model_info, clip_model, preprocess, patch, refs, dataset, kind, target, device)
            rr, cc = lzp.evaluate_maps(model_info["source"], target, maps, metas)
            raw.extend(rr)
            cats.extend(cc)
            loc.extend(lzp.localization_diagnostics(model_info["source"], target, maps, metas))
            stitch.extend(lzp.stitch_boundary_diagnostics(model_info["source"], target, maps, metas))
            scale.extend(scale_rows)
            write_csv(raw_path, raw)
            write_csv(cat_path, cats)
            write_csv(OUT / "DTD_ONLY_LOCALIZATION_DIAGNOSTICS.csv", loc)
            write_csv(OUT / "DTD_ONLY_STITCH_BOUNDARY_DIAGNOSTICS.csv", stitch)
            write_csv(OUT / "DTD_ONLY_SCORE_SCALE_DIAGNOSTICS.csv", scale)

    (OUT / "REPORT.md").write_text(
        "# DTD-Synthetic Full Metric Completion\n\n"
        "Completed image-level and pixel-level full-resolution metrics for Ours on DTD-Synthetic only. "
        "No training, target tuning, or method changes were performed.\n\n"
        f"Runtime: {time.time() - start:.1f}s.\n",
        encoding="utf-8",
    )
    print(f"[DTDComplete] wrote {raw_path} runtime={time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
