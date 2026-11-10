from pathlib import Path


def _count_files(path: Path):
    return sum(1 for p in path.rglob("*") if p.is_file())


def audit_dataset_root(root, dataset_name):
    root = Path(root)
    info = {
        "dataset": dataset_name,
        "root": str(root),
        "available": root.exists(),
        "categories": [],
        "normal_file_count": 0,
        "anomaly_file_count": 0,
        "loader_implementation": "utils.dataset.Dataset",
    }
    if not root.exists():
        return info
    for cat in sorted([p for p in root.iterdir() if p.is_dir() and p.name != "split_csv"]):
        normal = 0
        anomaly = 0
        if dataset_name.lower() == "mvtec":
            normal += _count_files(cat / "test" / "good")
            for defect in (cat / "test").iterdir() if (cat / "test").exists() else []:
                if defect.is_dir() and defect.name != "good":
                    anomaly += _count_files(defect)
        else:
            data_dir = cat / "Data"
            for p in data_dir.rglob("*") if data_dir.exists() else []:
                if not p.is_file():
                    continue
                lower = str(p).lower()
                if "normal" in lower or "/good/" in lower:
                    normal += 1
                elif "anomaly" in lower or "abnormal" in lower:
                    anomaly += 1
        info["categories"].append({"name": cat.name, "normal_files": normal, "anomaly_files": anomaly})
        info["normal_file_count"] += normal
        info["anomaly_file_count"] += anomaly
    return info


def build_dataset_audit(cfg):
    paths = cfg.get("data_paths", {})
    datasets = {}
    for name in ("visa", "mvtec"):
        datasets[name] = audit_dataset_root(paths.get(name, f"./data/{name}"), name)
    return {
        "datasets": datasets,
        "required_transfer_directions": [
            {"source": "visa", "target": "mvtec"},
            {"source": "mvtec", "target": "visa"},
        ],
        "unavailable_datasets": [k for k, v in datasets.items() if not v["available"]],
        "no_download_confirmation": True,
    }

