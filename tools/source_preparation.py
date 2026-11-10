"""Dispatch the original gated source preparation without changing its guards."""

from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=ROOT / "research")
    p.add_argument(
        "--task", choices=("preflight", "prepare", "source-build"), required=True
    )
    a = p.parse_args()
    script = (
        "rebuild_traceable_visa_source.py"
        if a.task == "preflight"
        else "source_build_traceable_visa.py"
    )
    path = a.project_root / "experiments" / script
    if not path.is_file():
        raise FileNotFoundError(path)
    # An archived review binds code/input hashes and may reject a relocated
    # workspace. Do not replace approval files or bypass the original gate.
    raise SystemExit(
        subprocess.call(
            [sys.executable, str(path), "--stage", a.task], cwd=a.project_root
        )
    )


if __name__ == "__main__":
    main()
