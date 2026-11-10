"""Reuse the frozen paper evaluator; do not implement a new metric variant."""

from pathlib import Path
import sys
import tempfile
import gc
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from .config import research_root


def run(manifest, output):
    sys.path.insert(0, str(research_root()))
    from experiments import final_unified_evaluation as ev

    manifest = Path(manifest).resolve()
    rows = ev.readcsv(manifest)
    if not rows:
        raise ValueError("Empty prediction manifest")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample IDs")
    for row in rows:
        for key in ("map_path", "img_path", "gt_path"):
            if row.get(key):
                path = Path(row[key])
                row[key] = str(path if path.is_absolute() else manifest.parent / path)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for category in sorted({r["category"] for r in rows}):
        cell = [r for r in rows if r["category"] == category]
        count = sum(int(r["gt_height"]) * int(r["gt_width"]) for r in cell)
        with tempfile.TemporaryDirectory(dir=output, prefix="category_") as temporary:
            yp = Path(temporary) / "labels.npy"
            sp = Path(temporary) / "scores.npy"
            y = np.lib.format.open_memmap(yp, mode="w+", dtype=bool, shape=(count,))
            s = np.lib.format.open_memmap(
                sp, mode="w+", dtype=np.float32, shape=(count,)
            )
            masks = []
            scores = []
            image_scores = []
            image_labels = []
            offset = 0
            for row in cell:
                mask = ev.native_mask(row)
                raw = np.load(row["map_path"])
                if raw.ndim != 2 or not np.isfinite(raw).all():
                    raise ValueError("Invalid anomaly map")
                image_scores.append(ev.topmean(ev.resize(raw, (74, 74))))
                image_labels.append(int(row["label"]))
                end = offset + mask.size
                y[offset:end] = mask.ravel()
                s[offset:end] = ev.resize(raw, mask.shape).ravel()
                masks.append(y[offset:end].reshape(mask.shape))
                scores.append(s[offset:end].reshape(mask.shape))
                offset = end
            value = ev.weighted_pixel_metrics(y, s)
            value["AUPRO"] = ev.aupro(masks, scores)
            valid = len(set(image_labels)) == 2
            value.update(
                I_AUROC=(
                    100 * roc_auc_score(image_labels, image_scores) if valid else None
                ),
                I_AP=(
                    100 * average_precision_score(image_labels, image_scores)
                    if valid
                    else None
                ),
                category=category,
            )
            results.append(value)
            masks.clear()
            scores.clear()
            del y, s
            gc.collect()
    ev.writecsv(output / "per_category.csv", results)
    summary = ev.macro_metrics(
        results, ["I_AUROC", "I_AP", "P_AUROC", "P_AP", "P_F1max", "AUPRO"]
    )
    ev.savejson(
        output / "summary.json", dict(summary, evaluator_hash=ev.evaluator_hash())
    )
    print(summary)
