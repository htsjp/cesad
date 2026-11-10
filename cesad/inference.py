"""Target inference only: frozen head and frozen source preparation."""

from pathlib import Path
import csv
import numpy as np
import torch
from .config import asset, identity, save_json
from .geometry import load_head
from .features import SourceTransform
from .lrp import fuse_views


def run(cfg, output, device, dry_run=False):
    manifest = asset(cfg, "manifest")
    checkpoint = asset(cfg, "checkpoint")
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Empty target manifest")
    pc1 = asset(cfg, "pc1", required=False)
    adapter = asset(cfg, "adapter", required=False)
    if cfg["source"] == "Full VisA" and (pc1 is None or adapter is None):
        raise ValueError("VisA needs explicit frozen PC1 and E2 adapter")
    for path in (pc1, adapter):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    cache = asset(cfg, "cached_features", required=False)
    if cache is not None:
        raw = np.load(cache, mmap_mode="r")
        if raw.shape != (len(rows), 5, 1369, 1024):
            raise ValueError("Cached features must match manifest row order and5 views")
    else:
        asset(cfg, "base_weights")
        asset(cfg, "ancillary_checkpoint")
    print(
        f'Inference preflight: {len(rows)} images, source={cfg["source"]}, fixed2x2 LRP'
    )
    if dry_run:
        return
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    head, _ = load_head(checkpoint, device)
    head.eval()
    head.requires_grad_(False)
    transform = SourceTransform(pc1, adapter).to(device)
    encoder = None
    if cache is None:
        from .backbone import FrozenEncoder

        encoder = FrozenEncoder(
            asset(cfg, "base_weights"), asset(cfg, "ancillary_checkpoint"), device
        )
    completed = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            image = Path(row["img_path"])
            if not image.is_absolute():
                image = manifest.parent / image
            feature = (
                np.asarray(raw[i], np.float32)
                if cache is not None
                else encoder.extract(image)
            )
            z = transform(torch.from_numpy(feature).to(device))
            q = head(z).reshape(1, 5, 37, 37)
            score = fuse_views(q)[0].cpu().numpy()
            if not np.isfinite(score).all():
                raise ValueError("Nonfinite raw logits")
            path = output / f"{i:06d}.npy"
            np.save(path, score)
            record = dict(
                row, img_path=str(image.resolve()), map_path=str(path.resolve())
            )
            if record.get("gt_path"):
                gt = Path(record["gt_path"])
                record["gt_path"] = str(
                    (gt if gt.is_absolute() else manifest.parent / gt).resolve()
                )
            completed.append(record)
    with (output / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(completed[0]))
        writer.writeheader()
        writer.writerows(completed)
    save_json(
        output / "inputs.json",
        dict(
            checkpoint=identity(checkpoint),
            manifest=identity(manifest),
            pc1=identity(pc1) if pc1 else None,
            adapter=identity(adapter) if adapter else None,
        ),
    )
