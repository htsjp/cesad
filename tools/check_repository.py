"""Check Python syntax, local documentation links, snapshot hashes and payloads."""

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=None,
        help="Optional external manifest for checking original snapshot hashes",
    )
    args = parser.parse_args()
    files = [
        p
        for p in ROOT.rglob("*")
        if p.is_file()
        and not set(p.relative_to(ROOT).parts) & {".git", "__pycache__", "outputs"}
    ]
    errors = []
    for p in files:
        if p.suffix == ".py":
            try:
                ast.parse(p.read_text(encoding="utf-8-sig"), filename=str(p))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"{p.relative_to(ROOT)}: {exc}")
        if p.suffix.lower() in {
            ".pt",
            ".pth",
            ".npy",
            ".npz",
            ".pem",
            ".key",
            ".docx",
            ".zip",
        }:
            errors.append(f"Unexpected payload: {p.relative_to(ROOT)}")
        if p.stat().st_size > 10 * 1024 * 1024:
            errors.append(f"Large file: {p.relative_to(ROOT)}")
        if p.suffix in {".py", ".md", ".yaml", ".yml", ".json"}:
            content = p.read_text(encoding="utf-8-sig")
            if re.search(
                r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
                content,
            ):
                errors.append(f"Possible credential material: {p.relative_to(ROOT)}")
    for p in [
        ROOT / "README.md",
        ROOT / "README_CN.md",
        ROOT / "THIRD_PARTY_NOTICES.md",
    ]:
        for destination in re.findall(r"\]\(([^)]+)\)", p.read_text(encoding="utf-8")):
            if "://" not in destination and not destination.startswith("#"):
                if not (p.parent / destination.split("#")[0]).exists():
                    errors.append(f"Broken link in {p.name}: {destination}")
    verified = 0
    if args.source_manifest is not None:
        with args.source_manifest.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                path = ROOT / "research" / row["path"]
                if (
                    not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
                ):
                    errors.append(f'Original snapshot mismatch: {row["path"]}')
                else:
                    verified += 1
    result = dict(
        files=len(files),
        python_files=sum(p.suffix == ".py" for p in files),
        original_files_verified=verified,
        errors=errors,
    )
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
