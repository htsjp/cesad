"""Explicit path configuration; no implicit old-cache or checkpoint fallback."""

from pathlib import Path
import hashlib
import json
import yaml


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    return cfg


def asset(cfg, key, required=True):
    value = cfg.get(key)
    if not value:
        if required:
            raise ValueError(f"Missing configuration field: {key}")
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(cfg["_config_path"]).parent / path
    if required and not path.is_file():
        raise FileNotFoundError(f"{key}: {path}")
    return path.resolve()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def research_root():
    path = Path(__file__).resolve().parents[1] / "research"
    if not (path / "experiments/final_unified_evaluation.py").is_file():
        raise FileNotFoundError("Missing original research implementation directory")
    return path
