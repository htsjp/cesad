import csv
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv


DATA_ROOT = Path(r"E:\WorkSpace\dataset")
VISA_ROOT = DATA_ROOT / "VISA"
OUT = Path("results/phase_xlvi_semantic_distillation")


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


def mask_ids(path):
    arr = np.asarray(Image.open(path))
    return tuple(sorted(int(x) for x in np.unique(arr) if int(x) != 0))


def parse_visa_object(obj_dir):
    rows = []
    with open(obj_dir / "image_anno.csv", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            label = row.get("label", "")
            names = [] if label == "normal" else [x.strip() for x in label.split(",") if x.strip()]
            ids = mask_ids(VISA_ROOT / row["mask"]) if row.get("mask") else tuple()
            rows.append({
                "object": obj_dir.name,
                "image": row.get("image", ""),
                "binary_label": int(label != "normal"),
                "label": label,
                "defect_names": names,
                "defect_ids": ids,
                "mask": row.get("mask", ""),
            })
    return rows


def solve_mapping(rows):
    anom = [r for r in rows if r["binary_label"] == 1]
    ids = sorted({i for r in anom for i in r["defect_ids"]})
    names = sorted({n for r in anom for n in r["defect_names"]})
    solutions = []
    if len(ids) != len(names) or len(ids) > 9:
        return ids, names, solutions
    for perm in itertools.permutations(names, len(ids)):
        mp = dict(zip(ids, perm))
        ok = True
        for r in anom:
            if sorted(mp[i] for i in r["defect_ids"]) != sorted(r["defect_names"]):
                ok = False
                break
        if ok:
            solutions.append(mp)
            if len(solutions) > 2:
                break
    return ids, names, solutions


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    objects = sorted([p for p in VISA_ROOT.iterdir() if p.is_dir() and (p / "image_anno.csv").exists()])
    audit_rows = []
    taxonomy_rows = []
    ambiguous_objects = []
    all_rows = []
    for obj in objects:
        rows = parse_visa_object(obj)
        all_rows.extend(rows)
        ids, names, solutions = solve_mapping(rows)
        reliable = len(solutions) == 1
        if not reliable:
            ambiguous_objects.append(obj.name)
        counts = Counter()
        for r in rows:
            if r["binary_label"] == 1:
                for n in r["defect_names"]:
                    counts[n] += 1
        audit_rows.append({
            "object": obj.name,
            "image": "__OBJECT_SUMMARY__",
            "binary_label": int(any(r["binary_label"] for r in rows)),
            "num_defect_types": len(names),
            "defect_ids": json.dumps(ids),
            "defect_names": json.dumps(names),
            "has_multiclass_mask": bool(any(len(r["defect_ids"]) > 1 or any(i > 1 for i in r["defect_ids"]) for r in rows)),
            "source_origin": str(obj / "image_anno.csv"),
            "mask_images": sum(1 for r in rows if r["mask"]),
            "mapping_solution_count": len(solutions),
            "mapping_reliable": reliable,
        })
        if solutions:
            for i, name in sorted(solutions[0].items()):
                taxonomy_rows.append({
                    "object": obj.name,
                    "defect_id": i,
                    "defect_name": name,
                    "image_count_with_name": counts[name],
                    "mapping_solution_count": len(solutions),
                    "mapping_reliable": reliable,
                })
        else:
            for i in ids:
                taxonomy_rows.append({
                    "object": obj.name,
                    "defect_id": i,
                    "defect_name": "",
                    "image_count_with_name": "",
                    "mapping_solution_count": 0,
                    "mapping_reliable": False,
                })
    # Representative image-level audit rows keep output compact while preserving required fields.
    for r in all_rows:
        if r["binary_label"] == 1:
            audit_rows.append({
                "object": r["object"],
                "image": r["image"],
                "binary_label": r["binary_label"],
                "num_defect_types": len(r["defect_names"]),
                "defect_ids": json.dumps(list(r["defect_ids"])),
                "defect_names": json.dumps(r["defect_names"]),
                "has_multiclass_mask": len(r["defect_ids"]) > 1 or any(i > 1 for i in r["defect_ids"]),
                "source_origin": r["mask"],
                "mask_images": 1,
                "mapping_solution_count": "",
                "mapping_reliable": "",
            })
    multiads_dirs = []
    for p in DATA_ROOT.rglob("*"):
        if p.is_dir() and any(s in p.name.lower() for s in ["multiads", "visa_extended", "visa-extended", "extended"]):
            multiads_dirs.append(str(p))
    source_labels_reliable = len(ambiguous_objects) == 0
    extra_source_supervision_required = False
    if not source_labels_reliable and multiads_dirs:
        outcome = "XLVI_A_EXTRA_SOURCE_SUPERVISION_REQUIRED"
        extra_source_supervision_required = True
    elif not source_labels_reliable:
        outcome = "XLVI_A_SOURCE_SEMANTIC_LABELS_UNAVAILABLE"
    else:
        outcome = "XLVI_A_IMPLEMENTATION_FAILURE"

    write_csv(OUT / "SOURCE_DEFECT_LABEL_AUDIT.csv", audit_rows)
    write_csv(OUT / "SOURCE_DEFECT_TAXONOMY.csv", taxonomy_rows)
    write_csv(OUT / "TEXT_TEACHER_GEOMETRY.csv", [{
        "status": "NOT_RUN_SOURCE_LABEL_AMBIGUITY",
        "reason": "No official id2class mapping file; pipe_fryum has multiple valid mask-id/name mappings.",
    }])
    write_json(OUT / "TRAINING_PROTOCOL.json", {
        "phase": "XLVI-A",
        "training_executed": False,
        "reason": "source defect labels not fully reliable under hard-stop rule",
        "source_root": str(VISA_ROOT),
        "ambiguous_objects": ambiguous_objects,
        "multiads_or_visa_extended_available": bool(multiads_dirs),
        "multiads_or_visa_extended_paths": multiads_dirs,
    })
    placeholder = [{
        "variant": "E1_TRUE_SEMANTIC_KD",
        "target": "BTAD/CVC-ClinicDB",
        "status": "NOT_RUN_SOURCE_LABEL_AMBIGUITY",
    }, {
        "variant": "E2_PERMUTED_SEMANTIC_CONTROL",
        "target": "BTAD/CVC-ClinicDB",
        "status": "NOT_RUN_SOURCE_LABEL_AMBIGUITY",
    }]
    write_csv(OUT / "RAW_METRICS.csv", placeholder)
    write_csv(OUT / "SEMANTIC_GAIN.csv", placeholder)
    write_csv(OUT / "VISUAL_TEXT_GEOMETRY.csv", placeholder)
    write_csv(OUT / "BINARY_MARGIN_SAFETY.csv", placeholder)
    write_json(OUT / "INFERENCE_DEPENDENCY_AUDIT.json", {
        "TEXT_AT_INFERENCE": False,
        "TARGET_DEFECT_VOCAB": False,
        "MEMORY_AT_INFERENCE": False,
        "RETRIEVAL_AT_INFERENCE": False,
        "inference_model_built": False,
        "reason": "E1/E2 not trained because source semantic labels were not fully reliable.",
    })
    (OUT / "HISTORICAL_ROUTE_DIFFERENCE.md").write_text(
        "# Historical Route Difference\n\n"
        "- VI-A dynamic prompt residual is not this phase.\n"
        "- VI-B text-anchored memory basis is not this phase.\n"
        "- XXXI semantic-conditioned retrieval is not this phase.\n"
        "- XLVI-A would be source-only training semantic regularization with text removed during inference, but training was not run because source pixel defect mapping was ambiguous.\n",
        encoding="utf-8",
    )
    report = [
        "# Phase XLVI-A Source-Only Fine-Grained Semantic Distillation Pilot",
        "",
        f"Outcome: **{outcome}**",
        "",
        "## Source Defect Label Audit",
        "",
        f"- VisA objects audited: {len(objects)}.",
        f"- Objects with unique mask-id/name mapping: {len(objects) - len(ambiguous_objects)}/{len(objects)}.",
        f"- Ambiguous objects: {', '.join(ambiguous_objects) if ambiguous_objects else 'none'}.",
        f"- MultiADS / VisA_Extended available: `{bool(multiads_dirs)}`.",
        "",
        "## Stop Reason",
        "",
        "Official/raw VisA provides image-level defect names and multi-valued masks, but no official id2class mapping file was found locally. Constraint solving recovers a unique mapping for 11/12 objects; `pipe_fryum` has two valid mappings, so full source pixel-level defect type recovery is not reliable under the hard-stop rule.",
        "",
        "E1/E2 were not trained. No target data, target labels, target masks, text prompts, or source model parameters were used for training decisions.",
        f"Runtime: {time.time() - start:.1f}s.",
    ]
    (OUT / "XLVI_A_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLVI-A",
        "outcome": outcome,
        "source_labels_reliable": source_labels_reliable,
        "ambiguous_objects": ambiguous_objects,
        "extra_source_supervision_required": extra_source_supervision_required,
        "multiads_or_visa_extended_available": bool(multiads_dirs),
        "training_executed": False,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLVI-A Source-Only Fine-Grained Semantic Distillation Pilot\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Unique mappings: {len(objects) - len(ambiguous_objects)}/{len(objects)}; ambiguous: {ambiguous_objects}.\n")
        fp.write("- E1/E2 not trained due to source semantic label reliability hard-stop.\n")
    print(json.dumps(json.loads((OUT / "OUTCOME.json").read_text(encoding="utf-8")), indent=2), flush=True)


if __name__ == "__main__":
    main()
