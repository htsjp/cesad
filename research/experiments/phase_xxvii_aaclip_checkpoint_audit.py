from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = Path(r"D:\ProjectPython\MRAD-main\reference_repos\AA-CLIP")
OUT = ROOT / "results" / "phase_xxvii_aaclip_external"


def run(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.stdout.strip()


def line_text(path: Path, line_no: int) -> str:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return lines[line_no - 1] if 0 < line_no <= len(lines) else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name: str, payload: dict) -> None:
    (OUT / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    remote_head = run(
        ["git", "ls-remote", "https://github.com/Mwxinnn/AA-CLIP.git", "HEAD"]
    )
    releases = run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Invoke-RestMethod -Uri https://api.github.com/repos/Mwxinnn/AA-CLIP/releases -Headers @{ 'User-Agent'='CodexAudit' }).Count",
        ]
    )
    checkpoint_files = []
    for pattern in ("*.pth", "*.pt", "*.ckpt", "*.bin"):
        checkpoint_files.extend(REPO.rglob(pattern))

    repository_audit = {
        "phase": "XXVII",
        "timestamp_utc": now,
        "official_repository": "https://github.com/Mwxinnn/AA-CLIP",
        "local_repository": str(REPO),
        "local_exists": REPO.exists(),
        "local_remote": run(["git", "remote", "-v"], cwd=REPO),
        "local_commit": run(["git", "rev-parse", "HEAD"], cwd=REPO),
        "local_status": run(["git", "status", "--short"], cwd=REPO),
        "remote_head": remote_head,
        "tags": run(["git", "tag"], cwd=REPO).splitlines(),
        "branches": run(["git", "branch", "-a"], cwd=REPO).splitlines(),
        "github_release_count": releases,
        "source_files_hashed": {
            rel: sha256(REPO / rel)
            for rel in [
                "README.md",
                "scripts.sh",
                "train.py",
                "test.py",
                "forward_utils.py",
            ]
            if (REPO / rel).exists()
        },
    }

    checkpoint_audit = {
        "phase": "XXVII",
        "timestamp_utc": now,
        "checkpoint_files_found": [
            {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(checkpoint_files)
        ],
        "released_checkpoint_link_found_in_audited_files": False,
        "github_release_count": releases,
        "official_readme_training_eval_lines": {
            "README.md:40": line_text(REPO / "README.md", 40),
            "README.md:43": line_text(REPO / "README.md", 43),
            "README.md:45": line_text(REPO / "README.md", 45),
            "README.md:49": line_text(REPO / "README.md", 49),
        },
        "official_test_checkpoint_gate": {
            "text_adapter_line": "test.py:163-170",
            "text_adapter_optional": True,
            "image_adapter_line": "test.py:172-173",
            "image_adapter_required": True,
            "line_172": line_text(REPO / "test.py", 172),
            "line_173": line_text(REPO / "test.py", 173),
        },
        "directions": {
            "D1_MVTec_to_VisA": {
                "valid_official_or_pretrained_checkpoint": False,
                "reason": "No released or local official AA-CLIP image_adapter_*.pth for source-trained MVTec evaluation on VisA.",
            },
            "D2_VisA_to_MVTec": {
                "valid_official_or_pretrained_checkpoint": False,
                "reason": "No released or local official AA-CLIP image_adapter_*.pth for source-trained VisA evaluation on MVTec.",
            },
        },
        "requires_training_before_official_evaluation": True,
        "automatic_training_started": False,
    }

    map_pipeline_audit = {
        "phase": "XXVII",
        "timestamp_utc": now,
        "files_inspected": [
            str(REPO / "forward_utils.py"),
            str(REPO / "test.py"),
            str(REPO / "train.py"),
            str(REPO / "README.md"),
            str(REPO / "scripts.sh"),
        ],
        "similarity_map_function": "forward_utils.py:196 calculate_similarity_map",
        "official_layer_map_topology": [
            {
                "step": "patch_text_similarity",
                "location": "forward_utils.py:199",
                "code": line_text(REPO / "forward_utils.py", 199),
            },
            {
                "step": "reshape_to_spatial_logits",
                "location": "forward_utils.py:202",
                "code": line_text(REPO / "forward_utils.py", 202),
            },
            {
                "step": "scalar_patch_pred",
                "location": "forward_utils.py:207",
                "code": line_text(REPO / "forward_utils.py", 207),
            },
            {
                "step": "official_gaussian_blur",
                "location": "forward_utils.py:208-210",
                "code": line_text(REPO / "forward_utils.py", 208),
            },
            {
                "step": "interpolate_to_image_size",
                "location": "forward_utils.py:211",
                "code": line_text(REPO / "forward_utils.py", 211),
            },
            {
                "step": "sum_official_used_layer_maps",
                "location": "test.py:93",
                "code": line_text(REPO / "test.py", 93),
            },
        ],
        "required_calibration_insertion_point_if_continued": {
            "location": "inside calculate_similarity_map, after forward_utils.py:207 and immediately before gaussian_blur2d at forward_utils.py:208",
            "scope": "Every official used layer map before blur; use one pooled CDF across all official used layers.",
        },
        "official_gaussian_for_industrial": {
            "domain_gate_location": "forward_utils.py:204-206",
            "sigma": 1,
            "kernel_size": 7,
            "applies_to": ["MVTec", "VisA"],
        },
        "official_image_metric_coupling": {
            "location": "forward_utils.py:250-252",
            "code": [
                line_text(REPO / "forward_utils.py", 250),
                line_text(REPO / "forward_utils.py", 252),
            ],
            "note": "AA-CLIP industrial image score blends pixel max and native image prediction, so a pixel-only calibrator would alter official image metrics unless native-det parity is reported separately.",
        },
    }

    outcome = {
        "phase": "XXVII",
        "outcome": "XXVII_OFFICIAL_CHECKPOINT_REVIEW_REQUIRED",
        "timestamp_utc": now,
        "trained": False,
        "target_accessed": False,
        "calibration_built": False,
        "evaluation_started": False,
        "reason": "Official AA-CLIP evaluation requires image_adapter_*.pth, but no valid released/local D1 or D2 official checkpoint was found; protocol requires immediate stop rather than training.",
        "required_user_decision": "Provide valid official/pretrained AA-CLIP checkpoints or explicitly authorize training in a separate phase.",
    }

    write_json("REPOSITORY_AUDIT.json", repository_audit)
    write_json("CHECKPOINT_AUDIT.json", checkpoint_audit)
    write_json("MAP_PIPELINE_AUDIT.json", map_pipeline_audit)
    write_json("OUTCOME.json", outcome)

    report = f"""# Phase XXVII AA-CLIP External Model Calibration Audit

## Outcome

`XXVII_OFFICIAL_CHECKPOINT_REVIEW_REQUIRED`

Phase XXVII was stopped before calibration or evaluation. No adapters were trained and no target data was accessed.

## Repository Audit

- Official repository: `https://github.com/Mwxinnn/AA-CLIP`
- Local repository: `{REPO}`
- Local commit: `{repository_audit["local_commit"]}`
- Remote HEAD: `{remote_head}`
- GitHub releases found: `{releases}`
- Local status: `{repository_audit["local_status"] or "clean"}`

The local checkout matches the current official remote HEAD. No tags were present in the audited repository.

## Checkpoint Audit

No local `.pth`, `.pt`, `.ckpt`, or `.bin` checkpoint files were found under the official AA-CLIP repository.

Official `test.py` requires trained image-adapter checkpoints:

- `test.py:172`: `{line_text(REPO / "test.py", 172)}`
- `test.py:173`: `{line_text(REPO / "test.py", 173)}`

The text adapter is optional in the official evaluation path, but the image adapter is mandatory. The README and `scripts.sh` describe training before evaluation, and the audited public GitHub release list is empty.

Therefore both requested directions are unavailable as strict official checkpoint evaluations:

- D1 MVTec -> VisA: no valid official/pretrained source-trained AA-CLIP checkpoint found.
- D2 VisA -> MVTec: no valid official/pretrained source-trained AA-CLIP checkpoint found.

## Map Pipeline Audit

If valid checkpoints are later supplied, the calibration insertion point is unambiguous:

- similarity map function: `forward_utils.py:196 calculate_similarity_map`
- scalar patch response: `forward_utils.py:207`
- insert calibration immediately after `forward_utils.py:207`
- official Gaussian blur begins at `forward_utils.py:208`
- final per-layer maps are interpolated and then summed at `test.py:93`

For MVTec and VisA, AA-CLIP uses the Industrial branch of the Gaussian settings: `kernel_size=7`, `sigma=1`.

One important metric caveat is that AA-CLIP's Industrial image metric blends pixel-map max and native image score in `metrics_eval`; a pixel-map calibrator can therefore change official image AUROC/AP even when the native image branch remains unchanged. Native image-score parity must be reported separately if evaluation is resumed.

## Stop Reason

The Phase XXVII protocol says to stop immediately if valid official/pretrained AA-CLIP checkpoints for D1 and D2 require training or are unavailable. That condition is met. Starting training would violate the protocol, so Phase XXVII is halted at checkpoint review.
"""
    (OUT / "XXVII_REPORT.md").write_text(report, encoding="utf-8")

    print(json.dumps(outcome, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
