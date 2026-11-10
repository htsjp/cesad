import ast
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_xlvi_semantic_distillation/official_mapping_recovery")
SOURCE = OUT / "source" / "id2class.py"
VISA_ROOT = Path(r"E:\WorkSpace\dataset\VISA")
REPO = "amazon-science/spot-diff"
COMMIT = "2a692ab575001cbde74d402d897a7286086c6199"
RAW_URL = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/utils/id2class.py"


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


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_name(s):
    return " ".join(str(s).strip().lower().split())


def load_official_mapping():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "id2class_map":
                    raw = ast.literal_eval(node.value)
                    return {
                        obj: {int(k): norm_name(v) for k, v in mp.items() if int(k) != 0}
                        for obj, mp in raw.items()
                    }
    raise RuntimeError("id2class_map not found")


def mask_ids(path):
    arr = np.asarray(Image.open(path))
    return tuple(sorted(int(x) for x in np.unique(arr) if int(x) != 0))


def object_rows(obj):
    rows = []
    with open(obj / "image_anno.csv", newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            if row["label"] == "normal":
                continue
            rows.append({
                "object": obj.name,
                "image": row["image"],
                "mask": row["mask"],
                "names": tuple(sorted(norm_name(x) for x in row["label"].split(",") if x.strip())),
                "ids": mask_ids(VISA_ROOT / row["mask"]),
            })
    return rows


def solve_candidates(rows):
    ids = sorted({i for r in rows for i in r["ids"]})
    names = sorted({n for r in rows for n in r["names"]})
    out = []
    if len(ids) != len(names):
        return []
    for perm in itertools.permutations(names, len(ids)):
        mp = dict(zip(ids, perm))
        if all(tuple(sorted(mp[i] for i in r["ids"])) == r["names"] for r in rows):
            out.append(mp)
    return out


def read_previous_taxonomy():
    path = Path("results/phase_xlvi_semantic_distillation/SOURCE_DEFECT_TAXONOMY.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as fp:
        for row in csv.DictReader(fp):
            rows.append({
                "object": row["object"],
                "mask_id": int(row["defect_id"]),
                "defect_name": norm_name(row["defect_name"]),
                "mapping_reliable": str(row["mapping_reliable"]).lower() == "true",
                "mapping_solution_count": int(row["mapping_solution_count"]),
            })
    return rows


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    official = load_official_mapping()
    write_json(OUT / "OFFICIAL_MAPPING_PROVENANCE.json", {
        "repository": REPO,
        "commit_sha": COMMIT,
        "file_path": "utils/id2class.py",
        "raw_url": RAW_URL,
        "retrieval_time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "local_copy": str(SOURCE),
        "sha256": sha256(SOURCE),
    })
    off_rows = []
    for obj, mp in sorted(official.items()):
        for mid, name in sorted(mp.items()):
            off_rows.append({
                "object": obj,
                "mask_id": mid,
                "defect_name": name,
                "official_source": RAW_URL,
                "commit_sha": COMMIT,
            })
    write_csv(OUT / "OFFICIAL_ID2CLASS.csv", off_rows)

    prev = read_previous_taxonomy()
    prev_by = {(r["object"], r["mask_id"]): r for r in prev}
    consistency = []
    objects = sorted(official)
    exact_matches = mismatches = missing_ids = extra_ids = 0
    historical_mismatch = []
    for obj in objects:
        ids = sorted(set(official[obj]) | {r["mask_id"] for r in prev if r["object"] == obj})
        for mid in ids:
            p = prev_by.get((obj, mid))
            o = official[obj].get(mid)
            if p and o:
                match = p["defect_name"] == o
                exact_matches += int(match)
                mismatches += int(not match)
                if p["mapping_reliable"] and not match:
                    historical_mismatch.append({"object": obj, "mask_id": mid, "previous": p["defect_name"], "official": o})
                status = "match" if match else "mismatch"
            elif p and not o:
                extra_ids += 1
                status = "extra_in_constraint_solver"
            else:
                missing_ids += 1
                status = "extra_in_official_unused_locally"
            consistency.append({
                "object": obj,
                "mask_id": mid,
                "constraint_solver_mapping": p["defect_name"] if p else "",
                "official_mapping": o if o else "",
                "historical_mapping_reliable": p["mapping_reliable"] if p else "",
                "status": status,
            })
    write_csv(OUT / "MAPPING_CONSISTENCY.csv", consistency)

    all_obj_rows = {obj.name: object_rows(obj) for obj in sorted([p for p in VISA_ROOT.iterdir() if (p / "image_anno.csv").exists()])}
    candidates = solve_candidates(all_obj_rows["pipe_fryum"])
    official_pipe = official["pipe_fryum"]
    pipe_local_ids = sorted({i for r in all_obj_rows["pipe_fryum"] for i in r["ids"]})
    official_pipe_local = {i: official_pipe[i] for i in pipe_local_ids if i in official_pipe}
    matched = []
    for i, cand in enumerate(candidates):
        if cand == official_pipe_local:
            matched.append(i)
    pipe_json = {
        "candidate_A": candidates[0] if len(candidates) > 0 else None,
        "candidate_B": candidates[1] if len(candidates) > 1 else None,
        "official_mapping": official_pipe,
        "official_mapping_restricted_to_local_mask_ids": official_pipe_local,
        "matched_candidate": f"candidate_{chr(ord('A') + matched[0])}" if len(matched) == 1 else "",
        "resolved": len(matched) == 1,
    }
    if not pipe_json["resolved"]:
        pipe_json["status"] = "PIPE_FRYUM_LOCAL_DATA_MISMATCH"
    write_json(OUT / "PIPE_FRYUM_RESOLUTION.json", pipe_json)

    val_rows = []
    total = valid = invalid_mask_id_images = unknown_count = 0
    per_object = {}
    for obj, rows in all_obj_rows.items():
        obj_total = obj_valid = obj_unknown = 0
        for r in rows:
            total += 1
            obj_total += 1
            unknown = [i for i in r["ids"] if i not in official[obj]]
            mapped = tuple(sorted(official[obj][i] for i in r["ids"] if i in official[obj]))
            ok = len(unknown) == 0 and mapped == r["names"]
            valid += int(ok)
            obj_valid += int(ok)
            invalid_mask_id_images += int(bool(unknown))
            unknown_count += len(unknown)
            obj_unknown += len(unknown)
            val_rows.append({
                "object": obj,
                "image": r["image"],
                "mask": r["mask"],
                "mask_ids": json.dumps(list(r["ids"])),
                "official_defect_names": json.dumps(list(mapped)),
                "image_anno_defect_names": json.dumps(list(r["names"])),
                "unknown_ids": json.dumps(unknown),
                "valid": ok,
            })
        per_object[obj] = {
            "total_anomaly_images": obj_total,
            "valid_multi_class_mask_images": obj_valid,
            "unknown_mask_id_count": obj_unknown,
            "official_mapping_coverage": obj_valid / max(obj_total, 1),
        }
    write_csv(OUT / "LOCAL_MASK_VALIDATION.csv", val_rows)

    historical_ok = len(historical_mismatch) == 0
    unknown_ok = unknown_count == 0
    pipe_ok = bool(pipe_json["resolved"])
    source_reliable = historical_ok and pipe_ok and unknown_ok
    outcome = "XLVI_A1_OFFICIAL_MAPPING_RECOVERED" if source_reliable else (
        "XLVI_A1_MAPPING_INCONSISTENT" if not historical_ok else "PIPE_FRYUM_LOCAL_DATA_MISMATCH"
    )
    availability = {
        "SOURCE_PIXEL_DEFECT_LABELS_RELIABLE": source_reliable,
        "EXTRA_SOURCE_SUPERVISION": False,
        "TARGET_INFORMATION_USED": False,
        "TOTAL_ANOMALY_IMAGES": total,
        "VALID_MULTI_CLASS_MASK_IMAGES": valid,
        "INVALID_MASK_ID_IMAGES": invalid_mask_id_images,
        "UNKNOWN_MASK_ID_COUNT": unknown_count,
        "OFFICIAL_MAPPING_COVERAGE": valid / max(total, 1),
        "per_object": per_object,
        "mapping_consistency": {
            "exact_matches": exact_matches,
            "mismatches": mismatches,
            "missing_ids": missing_ids,
            "extra_ids": extra_ids,
            "historical_reliable_mismatches": historical_mismatch,
        },
        "pipe_fryum_resolved": pipe_ok,
    }
    write_json(OUT / "SOURCE_SEMANTIC_AVAILABILITY.json", availability)
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLVI-A1",
        "outcome": outcome,
        "source_pixel_defect_labels_reliable": source_reliable,
        "runtime_sec": time.time() - start,
    })
    report = [
        "# Phase XLVI-A1 Official VisA ID2Class Recovery Audit",
        "",
        f"Outcome: **{outcome}**",
        "",
        "## Provenance",
        "",
        f"- Repository: `{REPO}`.",
        f"- Commit: `{COMMIT}`.",
        f"- File: `utils/id2class.py`.",
        f"- Local SHA256: `{sha256(SOURCE)}`.",
        "",
        "## Mapping Consistency",
        "",
        f"- Historical reliable mapping mismatches: {len(historical_mismatch)}.",
        f"- Exact compared matches: {exact_matches}.",
        f"- Official IDs unused in local masks: {missing_ids}.",
        "",
        "## Pipe Fryum",
        "",
        f"- Candidate count from local constraints: {len(candidates)}.",
        f"- Official mapping matched: `{pipe_json['matched_candidate']}`.",
        f"- Resolved: `{pipe_ok}`.",
        "",
        "## Local Mask Validation",
        "",
        f"- Total anomaly images: {total}.",
        f"- Valid images: {valid}.",
        f"- Unknown mask ids: {unknown_count}.",
        f"- Official mapping coverage: {availability['OFFICIAL_MAPPING_COVERAGE']:.6f}.",
        "",
        "Stop: XLVI-A1 only recovered source annotation metadata. No E1/E2 training was run.",
    ]
    (OUT / "XLVI_A1_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLVI-A1 Official VisA ID2Class Recovery Audit\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- Source pixel defect labels reliable: {source_reliable}; unknown mask ids: {unknown_count}; pipe_fryum resolved: {pipe_ok}.\n")
    print(json.dumps(json.loads((OUT / "OUTCOME.json").read_text(encoding="utf-8")), indent=2), flush=True)


if __name__ == "__main__":
    main()
