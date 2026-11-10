"""Read-only parity checks against existing source features and endpoints.

No encoder forward, training or target evaluation is performed.
"""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cesad.features import SourceTransform
from cesad.geometry import load_head
from cesad.config import identity, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--pc1", type=Path, required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--initial", type=Path, required=True)
    p.add_argument("--final", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    inputs = {
        k: identity(getattr(a, k))
        for k in ("data", "pc1", "adapter", "initial", "final")
    }
    with np.load(a.data) as data, torch.inference_mode():
        raw = torch.from_numpy(data["F_raw"])
        prepared = torch.from_numpy(data["F_prep"])
        new = SourceTransform(a.pc1, a.adapter)(raw)
        transform_error = float((new - prepared).abs().max())
        errors = {}
        for name, path, index in [("initial", a.initial, 1), ("final", a.final, 2)]:
            head, _ = load_head(path)
            q = head(prepared).numpy()
            errors[name] = float(np.max(np.abs(q - data[f"panel_{index}_scores"])))
        result = dict(
            inputs=inputs,
            patches=len(raw),
            transform_max_absolute_error=transform_error,
            score_max_absolute_errors=errors,
            tolerance=1e-6,
            pass_checks=transform_error < 1e-6 and max(errors.values()) < 1e-6,
        )
    save_json(a.output, result)
    print(json.dumps(result, indent=2))
    if not result["pass_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
