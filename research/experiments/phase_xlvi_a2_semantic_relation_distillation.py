import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.mrad_audit_utils import encode_plain_text, tokenize_texts
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, EPOCHS, LR, RANK, SEED, WEIGHT_DECAY, balanced_loss
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, load_full_visa_source, score_v1
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, SIDE, norm_np, upsample_map


OUT = Path("results/phase_xlvi_semantic_relation_distillation")
A1 = Path("results/phase_xlvi_semantic_distillation/official_mapping_recovery")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
CHECKPOINT = Path("checkpoints/test_on_mvtec.pth")
OFFICIAL_COMMIT = "2a692ab575001cbde74d402d897a7286086c6199"
OFFICIAL_SOURCE = f"https://raw.githubusercontent.com/amazon-science/spot-diff/{OFFICIAL_COMMIT}/utils/id2class.py"


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


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


def normalize_defect_name(name):
    return str(name).lower().replace("_", " ").strip()


def ensure_a1_ready():
    outcome = read_json(A1 / "OUTCOME.json")
    avail = read_json(A1 / "SOURCE_SEMANTIC_AVAILABILITY.json")
    ok = (
        outcome.get("outcome") == "XLVI_A1_OFFICIAL_MAPPING_RECOVERED"
        and bool(avail.get("SOURCE_PIXEL_DEFECT_LABELS_RELIABLE"))
        and not bool(avail.get("TARGET_INFORMATION_USED"))
    )
    if not ok:
        raise SystemExit("XLVI_A2_PATCH_LABEL_MAPPING_FAILURE")


def official_mapping():
    rows = read_csv(A1 / "OFFICIAL_ID2CLASS.csv")
    by_obj = defaultdict(dict)
    names = {}
    for r in rows:
        obj = r["object"]
        mid = int(r["mask_id"])
        name = normalize_defect_name(r["defect_name"])
        by_obj[obj][mid] = name
        names[name] = None
    identities = {name: i for i, name in enumerate(sorted(names))}
    out_rows = []
    for obj in sorted(by_obj):
        for mid, name in sorted(by_obj[obj].items()):
            out_rows.append({
                "object": obj,
                "mask_id": mid,
                "defect_name": name,
                "semantic_identity_id": identities[name],
                "official_source": OFFICIAL_SOURCE,
                "commit_sha": OFFICIAL_COMMIT,
            })
    write_csv(OUT / "SOURCE_TAXONOMY_FINAL.csv", out_rows)
    return dict(by_obj), identities, out_rows


def raw_mask_path(dataset, idx):
    row = dataset.data_all[idx]
    mask = row.get("mask_path") or row.get("mask") or ""
    if not mask:
        return None
    return Path(dataset.root) / mask


def multiclass_patch_ids(dataset, idx):
    p = raw_mask_path(dataset, idx)
    if p is None or not p.exists():
        return np.zeros((SIDE * SIDE,), dtype=np.int64)
    arr = Image.open(p)
    arr = arr.resize((SIDE, SIDE), Image.Resampling.NEAREST)
    return np.asarray(arr, dtype=np.int64).reshape(-1)


def source_samples_with_defects(dataset, indices, patches, mapping_by_obj, identities):
    rng = np.random.default_rng(SEED)
    cap = 512
    by_key = defaultdict(list)
    audit = defaultdict(lambda: {
        "source_images": 0,
        "anomaly_images": 0,
        "total_anomaly_patches": 0,
        "labeled_anomaly_patches": 0,
        "unknown_patch_ids": 0,
        "defect_names": Counter(),
    })
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        obj = str(item["cls_name"])
        audit[obj]["source_images"] += 1
        patch = np.asarray(patches[pos], dtype=np.float32)
        ids = multiclass_patch_ids(dataset, idx)
        if int(item["anomaly"]) == 0:
            pool = np.arange(ids.size)
            key = (obj, 0)
            room = cap - len(by_key[key])
            if room > 0:
                take = pool if pool.size <= room else rng.choice(pool, size=room, replace=False)
                by_key[key].append((norm_np(patch[take]), np.zeros(take.size, dtype=np.int64), np.full(take.size, -1, dtype=np.int64)))
            continue
        audit[obj]["anomaly_images"] += 1
        normal_pool = np.flatnonzero(ids == 0)
        anomaly_pool = np.flatnonzero(ids != 0)
        audit[obj]["total_anomaly_patches"] += int(anomaly_pool.size)
        known = [pidx for pidx in anomaly_pool if int(ids[pidx]) in mapping_by_obj[obj]]
        audit[obj]["labeled_anomaly_patches"] += int(len(known))
        audit[obj]["unknown_patch_ids"] += int(anomaly_pool.size - len(known))
        for pidx in known:
            audit[obj]["defect_names"][mapping_by_obj[obj][int(ids[pidx])]] += 1
        for cls, pool in [(0, normal_pool), (1, np.asarray(known, dtype=np.int64))]:
            if pool.size == 0:
                continue
            key = (obj, cls)
            room = cap - len(by_key[key])
            if room <= 0:
                continue
            take = pool if pool.size <= room else rng.choice(pool, size=room, replace=False)
            sem = np.full(take.size, -1, dtype=np.int64)
            if cls == 1:
                sem = np.asarray([identities[mapping_by_obj[obj][int(ids[pidx])]] for pidx in take], dtype=np.int64)
            by_key[key].append((norm_np(patch[take]), np.full(take.size, cls, dtype=np.int64), sem))
    feats, labels, semids, cats = [], [], [], []
    for (cat, _cls), parts in sorted(by_key.items()):
        if not parts:
            continue
        xs, ys, ss = zip(*parts)
        x = np.concatenate(xs, axis=0).astype(np.float32)
        y = np.concatenate(ys, axis=0).astype(np.int64)
        s = np.concatenate(ss, axis=0).astype(np.int64)
        feats.append(x)
        labels.append(y)
        semids.append(s)
        cats.extend([cat] * x.shape[0])
    audit_rows = []
    total_anom = total_labeled = total_unknown = 0
    for obj, d in sorted(audit.items()):
        coverage = d["labeled_anomaly_patches"] / max(1, d["total_anomaly_patches"])
        total_anom += d["total_anomaly_patches"]
        total_labeled += d["labeled_anomaly_patches"]
        total_unknown += d["unknown_patch_ids"]
        audit_rows.append({
            "object": obj,
            "source_images": d["source_images"],
            "anomaly_images": d["anomaly_images"],
            "TOTAL_ANOMALY_PATCHES": d["total_anomaly_patches"],
            "LABELED_ANOMALY_PATCHES": d["labeled_anomaly_patches"],
            "UNKNOWN_PATCH_IDS": d["unknown_patch_ids"],
            "DEFECT_LABEL_COVERAGE": coverage,
            "defect_patch_counts": json.dumps(dict(sorted(d["defect_names"].items())), ensure_ascii=True),
        })
    audit_rows.append({
        "object": "__TOTAL__",
        "source_images": sum(d["source_images"] for d in audit.values()),
        "anomaly_images": sum(d["anomaly_images"] for d in audit.values()),
        "TOTAL_ANOMALY_PATCHES": total_anom,
        "LABELED_ANOMALY_PATCHES": total_labeled,
        "UNKNOWN_PATCH_IDS": total_unknown,
        "DEFECT_LABEL_COVERAGE": total_labeled / max(1, total_anom),
        "defect_patch_counts": "",
    })
    write_csv(OUT / "PATCH_DEFECT_LABEL_AUDIT.csv", audit_rows)
    if not feats:
        raise SystemExit("XLVI_A2_PATCH_LABEL_MAPPING_FAILURE")
    return norm_np(np.concatenate(feats, axis=0)), np.concatenate(labels), np.concatenate(semids), np.asarray(cats), audit_rows


@torch.no_grad()
def build_text_relation(identities, device):
    model, *_ = load_components(str(CHECKPOINT), device)
    names = [None] * len(identities)
    for name, idx in identities.items():
        names[idx] = name
    prompts = [f"a photo of an object with {name}." for name in names]
    tokens = tokenize_texts(prompts).to(device)
    feats = encode_plain_text(model, tokens).float()
    feats = F.normalize(feats, dim=-1)
    rel = (feats @ feats.T).detach().cpu().numpy().astype(np.float32)
    rows = []
    for i, ni in enumerate(names):
        rows.append({"defect_i": ni, "defect_j": ni, "i": i, "j": i, "R_text": float(rel[i, i]), "prompt_i": prompts[i], "prompt_j": prompts[i]})
        for j in range(i + 1, len(names)):
            rows.append({"defect_i": ni, "defect_j": names[j], "i": i, "j": j, "R_text": float(rel[i, j]), "prompt_i": prompts[i], "prompt_j": prompts[j]})
    write_csv(OUT / "TEXT_TEACHER_RELATION.csv", rows)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rel, names


def derangement(n, seed=0):
    rng = np.random.default_rng(seed)
    if n < 2:
        return np.arange(n)
    while True:
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p


def semantic_loss(z, sem, relation, mode, perm=None):
    valid = sem >= 0
    if int(valid.sum().item()) == 0:
        return z.sum() * 0.0, 0
    present = torch.unique(sem[valid])
    if present.numel() < 2:
        return z.sum() * 0.0, int(present.numel())
    cents = []
    ids = []
    for sid in present.tolist():
        c = F.normalize(z[sem == int(sid)].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        cents.append(c)
        ids.append(int(sid))
    cents = torch.stack(cents, dim=0)
    rv = cents @ cents.T
    ii, jj = torch.triu_indices(len(ids), len(ids), offset=1, device=z.device)
    pred = rv[ii, jj]
    if mode == "E2_LABEL_ONLY_TYPE_SEPARATION":
        target = torch.zeros_like(pred)
    else:
        ids_np = np.asarray(ids, dtype=np.int64)
        if mode == "E3_PERMUTED_SEMANTIC_CONTROL":
            ids_np = perm[ids_np]
        tgt = relation[np.ix_(ids_np, ids_np)]
        target = torch.tensor(tgt, dtype=torch.float32, device=z.device)[ii, jj]
    return F.mse_loss(pred, target.detach()), int(present.numel())


def train_variant(variant, x_np, y_np, sem_np, states, relation, device, perm=None):
    ckpt = OUT / "checkpoints" / f"{variant}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        model = AdapterToState(x_np.shape[1], states).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        return model, payload["curves"], payload["coverage"]
    model = AdapterToState(x_np.shape[1], states).to(device)
    x_all = norm_np(x_np)
    y_all = y_np.astype(np.float32)
    sem_all = sem_np.astype(np.int64)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves, coverage_rows = [], []
    active = total = 0
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all))
        vals = []
        for step, start in enumerate(range(0, len(order), BATCH_SIZE)):
            idx = order[start:start + BATCH_SIZE]
            x = torch.tensor(x_all[idx], dtype=torch.float32, device=device)
            y = torch.tensor(y_all[idx], dtype=torch.float32, device=device)
            sem = torch.tensor(sem_all[idx], dtype=torch.long, device=device)
            opt.zero_grad()
            logits, z = model.logits(x)
            base, bce, dice = balanced_loss(logits, y)
            sem_l, ntypes = semantic_loss(z, sem, relation, variant, perm=perm)
            loss = base + sem_l
            loss.backward()
            opt.step()
            is_active = ntypes >= 2
            active += int(is_active)
            total += 1
            vals.append((float(loss.detach().cpu()), float(base.detach().cpu()), float(sem_l.detach().cpu()), float(bce.cpu()), float(dice.cpu()), ntypes))
            coverage_rows.append({"variant": variant, "epoch": epoch, "step": step, "num_defect_types": ntypes, "semantic_active": is_active})
        arr = np.asarray(vals, dtype=np.float64)
        curves.append({
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "base_loss": float(arr[:, 1].mean()),
            "semantic_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
            "mean_defect_types_per_step": float(arr[:, 5].mean()),
        })
    coverage = {"variant": variant, "SEMANTIC_ACTIVE_STEP_RATIO": active / max(1, total), "active_steps": active, "total_steps": total, "rows": coverage_rows}
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "curves": curves, "coverage": coverage}, ckpt)
    return model.eval(), curves, coverage


def top1_image_score(map37):
    flat = np.asarray(map37, dtype=np.float32).reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


def score_source_model(patch, adapter, device):
    maps = np.zeros((patch.shape[0], SIDE, SIDE), dtype=np.float16)
    image_scores = np.zeros((patch.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for i in range(patch.shape[0]):
            x = torch.tensor(norm_np(np.asarray(patch[i], dtype=np.float32)), dtype=torch.float32, device=device)
            logits, _ = adapter.logits(x)
            m = torch.sigmoid(logits).detach().cpu().numpy().reshape(SIDE, SIDE).astype(np.float32)
            maps[i] = m.astype(np.float16)
            image_scores[i] = top1_image_score(m)
    return image_scores, maps


def evaluate_btad(adapter, device):
    dataset = ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test")
    c = np.load(XLV_A / "feature_cache" / "BTAD_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1(c["patch"], adapter, device)
    mean, per = evaluate_external(dataset, [dict(r) for r in c["rows"]], image_scores, maps)
    return mean, per


def evaluate_clinicdb(adapter, device):
    dataset = MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test")
    c = np.load(XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1(c["patch"], adapter, device)
    metric, _per_img, _pixel = eval_scores(dataset, [dict(r) for r in c["rows"]], image_scores, maps)
    return metric, {}


def load_e0_adapter(states, device):
    adapter = AdapterToState(1024, states).to(device).eval()
    payload = torch.load(XLV_A / "source_model" / "full_visa_adapter.pt", map_location=device)
    adapter.load_state_dict(payload["model"], strict=True)
    return adapter


def e0_metrics():
    out = []
    for r in read_csv(XLV_A / "PILOT_METRICS.csv"):
        if r["target"] == "BTAD" and r["variant"] == "V1_STATE_ALIGNED":
            out.append({"target": "BTAD", "variant": "E0_CURRENT_STATE_ALIGNED", **{k: float(r[k]) for k in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"]}})
    for r in read_csv(XLV_B / "RAW_METRICS.csv"):
        if r["target"] == "CVC-ClinicDB" and r["variant"] == "V1_STATE_ALIGNED":
            out.append({"target": "CVC-ClinicDB", "variant": "E0_CURRENT_STATE_ALIGNED", **{k: float(r[k]) for k in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"]}})
    return out


def metric_row(target, variant, mean):
    row = {"target": target, "variant": variant}
    for k in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "mask_images"]:
        row[k] = float(mean.get(k, float("nan")))
    row["G_pixel"] = gp(row)
    return row


def source_relation_geometry(variant, adapter, x_np, sem_np, relation, device):
    valid_ids = sorted(int(x) for x in np.unique(sem_np) if int(x) >= 0)
    if len(valid_ids) < 3:
        return {"variant": variant, "defect_types": len(valid_ids), "spearman_visual_text": float("nan")}
    with torch.no_grad():
        x = torch.tensor(norm_np(x_np.astype(np.float32)), dtype=torch.float32, device=device)
        _logits, z = adapter.logits(x)
        z = z.detach().cpu().numpy().astype(np.float32)
    cents = []
    for sid in valid_ids:
        cents.append(norm_np(z[sem_np == sid].mean(axis=0, keepdims=True))[0])
    cents = np.stack(cents)
    rv = cents @ cents.T
    rt = relation[np.ix_(valid_ids, valid_ids)]
    iu = np.triu_indices(len(valid_ids), k=1)
    rho = spearmanr(rv[iu], rt[iu]).correlation
    return {"variant": variant, "defect_types": len(valid_ids), "spearman_visual_text": float(rho), "pairs": int(len(iu[0]))}


def binary_margin_safety(variant, adapter, x_np, y_np, states, device):
    with torch.no_grad():
        x = torch.tensor(norm_np(x_np.astype(np.float32)), dtype=torch.float32, device=device)
        _logits, z = adapter.logits(x)
        s = torch.tensor(norm_np(states), dtype=torch.float32, device=device)
        m = (z @ s[1] - z @ s[0]).detach().cpu().numpy()
    rows = []
    for lab, name in [(0, "normal"), (1, "anomaly")]:
        vals = m[y_np == lab]
        rows.append({"variant": variant, "group": name, "mean": float(np.mean(vals)), "median": float(np.median(vals)), "count": int(vals.size)})
    rows.append({
        "variant": variant,
        "group": "anomaly_minus_normal",
        "mean": float(np.mean(m[y_np == 1]) - np.mean(m[y_np == 0])),
        "median": float(np.median(m[y_np == 1]) - np.median(m[y_np == 0])),
        "count": int(m.size),
    })
    return rows


def parameter_rows(variants):
    rows = []
    for v in variants:
        rows.append({
            "variant": v,
            "rank": RANK,
            "adapter_parameters": 8192,
            "state_parameters_inference": 0,
            "text_parameters_inference": 0,
            "total_inference_trainable_parameters": 8192,
        })
    write_csv(OUT / "PARAMETER_AUDIT.csv", rows)


def causal_rows(raw_rows):
    by = {(r["target"], r["variant"]): r for r in raw_rows}
    rows = []
    for target in sorted({r["target"] for r in raw_rows}):
        if (target, "E0_CURRENT_STATE_ALIGNED") not in by or (target, "E1_TRUE_SEMANTIC_RELATION_DISTILLATION") not in by:
            continue
        e0 = by[(target, "E0_CURRENT_STATE_ALIGNED")]
        e1 = by[(target, "E1_TRUE_SEMANTIC_RELATION_DISTILLATION")]
        rows.append({"target": target, "comparison": "E1_minus_E0", "delta_P-AUROC": float(e1["P-AUROC"]) - float(e0["P-AUROC"]), "delta_P-AP": float(e1["P-AP"]) - float(e0["P-AP"]), "delta_G_pixel": gp(e1) - gp(e0)})
        for ctrl in ["E2_LABEL_ONLY_TYPE_SEPARATION", "E3_PERMUTED_SEMANTIC_CONTROL"]:
            if (target, ctrl) in by:
                c = by[(target, ctrl)]
                rows.append({"target": target, "comparison": f"E1_minus_{ctrl[:2]}", "delta_P-AUROC": float(e1["P-AUROC"]) - float(c["P-AUROC"]), "delta_P-AP": float(e1["P-AP"]) - float(c["P-AP"]), "delta_G_pixel": gp(e1) - gp(c)})
    write_csv(OUT / "CAUSAL_COMPARISON.csv", rows)
    return rows


def determine_outcome(raw_rows, e1_gate, controls_run):
    if not e1_gate.get("patch_label_gate_passed", False):
        return "XLVI_A2_PATCH_LABEL_MAPPING_FAILURE"
    if not e1_gate.get("semantic_batch_gate_passed", False):
        return "XLVI_A2_SEMANTIC_BATCH_COVERAGE_INSUFFICIENT"
    by = {(r["target"], r["variant"]): r for r in raw_rows}
    e0b = by[("BTAD", "E0_CURRENT_STATE_ALIGNED")]
    e1b = by[("BTAD", "E1_TRUE_SEMANTIC_RELATION_DISTILLATION")]
    e0c = by[("CVC-ClinicDB", "E0_CURRENT_STATE_ALIGNED")]
    e1c = by[("CVC-ClinicDB", "E1_TRUE_SEMANTIC_RELATION_DISTILLATION")]
    d_btad = gp(e1b) - gp(e0b)
    d_clinic = gp(e1c) - gp(e0c)
    if d_btad < 0.50:
        return "XLVI_A2_SEMANTIC_RELATION_NO_SIGNAL"
    if d_btad >= 1.00 and d_clinic <= -1.00:
        return "XLVI_A2_INDUSTRIAL_SEMANTICS_HURT_CROSS_DOMAIN"
    if not controls_run:
        return "XLVI_A2_IMPLEMENTATION_FAILURE"
    e2b = by[("BTAD", "E2_LABEL_ONLY_TYPE_SEPARATION")]
    e3b = by[("BTAD", "E3_PERMUTED_SEMANTIC_CONTROL")]
    e1_e2 = gp(e1b) - gp(e2b)
    e1_e3 = gp(e1b) - gp(e3b)
    if d_btad >= 1.00 and e1_e2 >= 0.50 and e1_e3 >= 0.50 and d_clinic >= -0.50:
        return "XLVI_A2_SEMANTIC_TOPOLOGY_SUPPORTED"
    if d_btad > 0.50 and abs(e1_e2) < 0.20:
        return "XLVI_A2_FINE_GRAINED_PARTITION_ONLY"
    if d_btad > 0.50 and abs(e1_e3) < 0.20:
        return "XLVI_A2_SEMANTIC_IDENTITY_NOT_SPECIFIC"
    return "XLVI_A2_SEMANTIC_RELATION_NO_SIGNAL"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    ensure_a1_ready()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    mapping_by_obj, identities, _tax_rows = official_mapping()
    relation, identity_names = build_text_relation(identities, device)

    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img, _rows = load_full_visa_source()
    x_train, y_train, sem_train, cats_train, patch_audit = source_samples_with_defects(dataset, indices, patch, mapping_by_obj, identities)
    total_row = next(r for r in patch_audit if r["object"] == "__TOTAL__")
    patch_gate = float(total_row["DEFECT_LABEL_COVERAGE"]) >= 0.99 and int(total_row["UNKNOWN_PATCH_IDS"]) == 0
    if not patch_gate:
        write_json(OUT / "OUTCOME.json", {"phase": "XLVI-A2", "outcome": "XLVI_A2_PATCH_LABEL_MAPPING_FAILURE"})
        print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)
        return

    states = np.load(XLV_A / "source_model" / "full_visa_states.npz")["patch_states"]
    e0_adapter = load_e0_adapter(states, device)
    raw_rows = e0_metrics()
    geom_rows = [source_relation_geometry("E0_CURRENT_STATE_ALIGNED", e0_adapter, x_train, sem_train, relation, device)]
    margin_rows = binary_margin_safety("E0_CURRENT_STATE_ALIGNED", e0_adapter, x_train, y_train, states, device)

    e1, e1_curves, e1_cov = train_variant("E1_TRUE_SEMANTIC_RELATION_DISTILLATION", x_train, y_train, sem_train, states, relation, device)
    sem_cov_rows = e1_cov["rows"]
    write_csv(OUT / "SEMANTIC_BATCH_COVERAGE.csv", sem_cov_rows + [{"variant": "E1_TRUE_SEMANTIC_RELATION_DISTILLATION", "epoch": "__SUMMARY__", "step": "", "num_defect_types": "", "semantic_active": "", "SEMANTIC_ACTIVE_STEP_RATIO": e1_cov["SEMANTIC_ACTIVE_STEP_RATIO"]}])
    semantic_batch_gate = e1_cov["SEMANTIC_ACTIVE_STEP_RATIO"] >= 0.30

    for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
        mean, _per = fn(e1, device)
        raw_rows.append(metric_row(target, "E1_TRUE_SEMANTIC_RELATION_DISTILLATION", mean))
    geom_rows.append(source_relation_geometry("E1_TRUE_SEMANTIC_RELATION_DISTILLATION", e1, x_train, sem_train, relation, device))
    margin_rows.extend(binary_margin_safety("E1_TRUE_SEMANTIC_RELATION_DISTILLATION", e1, x_train, y_train, states, device))

    by = {(r["target"], r["variant"]): r for r in raw_rows}
    e0b = by[("BTAD", "E0_CURRENT_STATE_ALIGNED")]
    e1b = by[("BTAD", "E1_TRUE_SEMANTIC_RELATION_DISTILLATION")]
    e0c = by[("CVC-ClinicDB", "E0_CURRENT_STATE_ALIGNED")]
    e1c = by[("CVC-ClinicDB", "E1_TRUE_SEMANTIC_RELATION_DISTILLATION")]
    e1_gate = {
        "patch_label_gate_passed": patch_gate,
        "semantic_batch_gate_passed": semantic_batch_gate,
        "SEMANTIC_ACTIVE_STEP_RATIO": e1_cov["SEMANTIC_ACTIVE_STEP_RATIO"],
        "BTAD_E1_minus_E0_G_pixel": gp(e1b) - gp(e0b),
        "BTAD_E1_minus_E0_P-AUROC": float(e1b["P-AUROC"]) - float(e0b["P-AUROC"]),
        "BTAD_E1_minus_E0_P-AP": float(e1b["P-AP"]) - float(e0b["P-AP"]),
        "BTAD_continuation_pass": bool(gp(e1b) - gp(e0b) >= 1.00 and float(e1b["P-AP"]) > float(e0b["P-AP"]) and float(e1b["P-AUROC"]) - float(e0b["P-AUROC"]) >= -0.20),
        "BTAD_no_signal_stop": bool(gp(e1b) - gp(e0b) < 0.50),
        "ClinicDB_E1_minus_E0_G_pixel": gp(e1c) - gp(e0c),
        "ClinicDB_non_destruction_pass": bool(gp(e1c) - gp(e0c) >= -0.50),
    }
    write_json(OUT / "E1_CONTINUATION_GATE.json", e1_gate)

    controls_run = False
    curves = e1_curves
    if semantic_batch_gate and e1_gate["BTAD_continuation_pass"] and e1_gate["ClinicDB_non_destruction_pass"]:
        perm = derangement(len(identity_names), seed=SEED)
        write_json(OUT / "PERMUTED_SEMANTIC_CONTROL.json", {"seed": SEED, "derangement": perm.tolist(), "identity_names": identity_names, "no_fixed_point": bool(np.all(perm != np.arange(len(perm))))})
        for variant in ["E2_LABEL_ONLY_TYPE_SEPARATION", "E3_PERMUTED_SEMANTIC_CONTROL"]:
            model, c, cov = train_variant(variant, x_train, y_train, sem_train, states, relation, device, perm=perm)
            curves.extend(c)
            for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
                mean, _per = fn(model, device)
                raw_rows.append(metric_row(target, variant, mean))
            geom_rows.append(source_relation_geometry(variant, model, x_train, sem_train, relation, device))
            margin_rows.extend(binary_margin_safety(variant, model, x_train, y_train, states, device))
        controls_run = True
    else:
        for variant in ["E2_LABEL_ONLY_TYPE_SEPARATION", "E3_PERMUTED_SEMANTIC_CONTROL"]:
            raw_rows.append({"target": "BTAD", "variant": variant, "I-AUROC": float("nan"), "I-AP": float("nan"), "P-AUROC": float("nan"), "P-AP": float("nan"), "PRO": float("nan"), "G_pixel": float("nan"), "status": "NOT_RUN_E1_CONTINUATION_GATE_FAILED"})
            raw_rows.append({"target": "CVC-ClinicDB", "variant": variant, "I-AUROC": float("nan"), "I-AP": float("nan"), "P-AUROC": float("nan"), "P-AP": float("nan"), "PRO": float("nan"), "G_pixel": float("nan"), "status": "NOT_RUN_E1_CONTINUATION_GATE_FAILED"})
            geom_rows.append({"variant": variant, "defect_types": len(identity_names), "spearman_visual_text": float("nan"), "status": "NOT_RUN_E1_CONTINUATION_GATE_FAILED"})

    write_json(OUT / "TRAINING_PROTOCOL.json", {
        "phase": "XLVI-A2",
        "source": "FULL VisA source-labeled protocol reused from XLV source model",
        "source_cache": str(XLV_A / "source_model"),
        "rank": RANK,
        "adapter_parameters": 8192,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "lambda_sem": 1.0,
        "checkpoint_selection": "final_epoch_10_only",
        "target_data_used_for_training": False,
        "target_data_used_for_hyperparameter_selection": False,
        "text_teacher": "defect-defect relation matrix only",
    })
    write_csv(OUT / "RAW_METRICS.csv", raw_rows)
    comparisons = causal_rows(raw_rows)
    write_csv(OUT / "SOURCE_RELATION_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "BINARY_MARGIN_SAFETY.csv", margin_rows)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_json(OUT / "INFERENCE_DEPENDENCY_AUDIT.json", {
        "TEXT_AT_INFERENCE": False,
        "DEFECT_TAXONOMY_AT_INFERENCE": False,
        "TARGET_DEFECT_VOCAB": False,
        "MEMORY_AT_INFERENCE": False,
        "RETRIEVAL_AT_INFERENCE": False,
        "TARGET_ADAPTATION": False,
        "saved_inference_components": ["rank4_adapter", "s_N", "s_A"],
    })
    parameter_rows(["E0_CURRENT_STATE_ALIGNED", "E1_TRUE_SEMANTIC_RELATION_DISTILLATION", "E2_LABEL_ONLY_TYPE_SEPARATION", "E3_PERMUTED_SEMANTIC_CONTROL"])
    outcome = determine_outcome(raw_rows, e1_gate, controls_run)

    report = ["# Phase XLVI-A2 Source-Only Fine-Grained Semantic Relation Distillation", "", f"Outcome: **{outcome}**", "", "## E1 Continuation Gate", ""]
    report.append(f"- Patch defect coverage: {float(total_row['DEFECT_LABEL_COVERAGE']):.6f}; unknown patch ids: {int(total_row['UNKNOWN_PATCH_IDS'])}.")
    report.append(f"- Semantic active step ratio: {e1_cov['SEMANTIC_ACTIVE_STEP_RATIO']:.4f}.")
    report.append(f"- BTAD E1-E0: G {e1_gate['BTAD_E1_minus_E0_G_pixel']:+.4f}, P-AUROC {e1_gate['BTAD_E1_minus_E0_P-AUROC']:+.4f}, P-AP {e1_gate['BTAD_E1_minus_E0_P-AP']:+.4f}.")
    report.append(f"- ClinicDB E1-E0 G: {e1_gate['ClinicDB_E1_minus_E0_G_pixel']:+.4f}.")
    report.extend(["", "## Metrics", ""])
    for r in raw_rows:
        status = r.get("status", "")
        if status:
            report.append(f"- {r['target']} {r['variant']}: {status}.")
        else:
            report.append(f"- {r['target']} {r['variant']}: P-AUROC {float(r['P-AUROC']):.4f}, P-AP {float(r['P-AP']):.4f}, G {float(r['G_pixel']):.4f}, I-AUROC {float(r['I-AUROC']):.4f}.")
    report.extend(["", "## Source Geometry", ""])
    for r in geom_rows:
        report.append(f"- {r['variant']}: Spearman visual/text relation {float(r.get('spearman_visual_text', float('nan'))):.4f}.")
    report.extend(["", "## Inference Audit", "", "Text, defect taxonomy, target vocabulary, memory, retrieval, and target adaptation are all disabled at inference. Only the rank-4 adapter and frozen visual states are needed.", "", f"Runtime: {time.time() - start:.1f}s.", "", "Stop: Phase XLVI-A2 ended here."])
    (OUT / "XLVI_A2_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {"phase": "XLVI-A2", "outcome": outcome, "controls_run": controls_run, "E1_gate": e1_gate, "causal_comparison": comparisons, "runtime_sec": time.time() - start})
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n## Phase XLVI-A2 Source-Only Fine-Grained Semantic Relation Distillation\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- E1 gate: {json.dumps(e1_gate, default=json_default)}.\n")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
