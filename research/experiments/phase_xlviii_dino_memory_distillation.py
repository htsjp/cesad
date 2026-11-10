import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, EPOCHS, LR, RANK, SEED, WEIGHT_DECAY, balanced_loss
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.phase_xliii_anchor_specificity import low_rank_rows
from experiments.phase_xlv_cross_domain_transfer import ExternalDataset, evaluate_external, gp, load_full_visa_source, score_v1
from experiments.phase_xlv_medical_pixel_transfer import MedicalDataset, eval_scores
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY, norm_np, patch_labels


OUT = Path("results/phase_xlviii_dino_memory_distillation")
XLV_A = Path("results/phase_xlv_cross_domain_transfer")
XLV_B = Path("results/phase_xlv_medical_pixel_transfer")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
K = 8
TAU = 0.07
LAMBDA_STRUCT = 1.0


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


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_source_arrays():
    dataset_name, dataset, indices, _cls, clip_patch, _y_img, _cat_img, _rows = load_full_visa_source()
    dino = np.load(FEATURE_CACHE / "D2_fold0_dino.npz", allow_pickle=True)
    dino_patch = np.concatenate([dino["train_patch"], dino["held_patch"]], axis=0).astype(np.float16)
    if dino_patch.shape[:2] != clip_patch.shape[:2]:
        raise SystemExit("XLVIII_SOURCE_FEATURE_ALIGNMENT_FAILURE")
    return dataset_name, dataset, indices, np.asarray(clip_patch, dtype=np.float16), dino_patch


def build_source_training_sample():
    sample_path = OUT / "cache" / "source_patch_sample.npz"
    if sample_path.exists():
        return np.load(sample_path, allow_pickle=True)
    dataset_name, dataset, indices, clip_patch, dino_patch = load_source_arrays()
    rng = np.random.default_rng(SEED)
    by_key = {}
    for pos, idx in enumerate(indices):
        item = dataset[idx]
        cat = str(item["cls_name"])
        labels = patch_labels(item)
        for cls in [0, 1]:
            key = (cat, cls)
            if key not in by_key:
                by_key[key] = []
            room = PATCH_TRAIN_SAMPLES_PER_CLASS_PER_CATEGORY - len(by_key[key])
            if room <= 0:
                continue
            local = np.flatnonzero(labels == cls)
            if local.size == 0:
                continue
            if local.size > room:
                local = rng.choice(local, size=room, replace=False)
            by_key[key].extend((pos, int(pid)) for pid in local.tolist())
    coords, labels, cats = [], [], []
    for (cat, cls), vals in sorted(by_key.items()):
        coords.extend(vals)
        labels.extend([cls] * len(vals))
        cats.extend([cat] * len(vals))
    image_pos = np.asarray([c[0] for c in coords], dtype=np.int32)
    patch_id = np.asarray([c[1] for c in coords], dtype=np.int16)
    x_clip = norm_np(np.stack([clip_patch[p, q].astype(np.float32) for p, q in coords], axis=0))
    x_dino = norm_np(np.stack([dino_patch[p, q].astype(np.float32) for p, q in coords], axis=0))
    y = np.asarray(labels, dtype=np.int64)
    cats = np.asarray(cats)
    OUT.joinpath("cache").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        sample_path,
        x_clip=x_clip.astype(np.float32),
        x_dino=x_dino.astype(np.float32),
        y=y,
        image_pos=image_pos,
        patch_id=patch_id,
        cats=cats,
        dataset_name=dataset_name,
    )
    return np.load(sample_path, allow_pickle=True)


def build_teacher_neighbors(x_dino, image_pos, mode):
    path = OUT / "cache" / f"teacher_neighbors_{mode}.npz"
    if path.exists():
        return np.load(path)
    x = norm_np(x_dino.astype(np.float32))
    n = x.shape[0]
    idx = np.zeros((n, K), dtype=np.int64)
    probs = np.zeros((n, K), dtype=np.float32)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    all_t = torch.tensor(x, dtype=torch.float32, device=device)
    img = np.asarray(image_pos)
    if mode == "E1_TRUE_DINO_NEIGHBOUR":
        with torch.no_grad():
            for start in range(0, n, 512):
                end = min(n, start + 512)
                sim = (all_t[start:end] @ all_t.t()).detach().cpu().numpy()
                same = img[None, :] == img[start:end, None]
                sim[same] = -np.inf
                part = np.argpartition(-sim, kth=K - 1, axis=1)[:, :K]
                part_sim = np.take_along_axis(sim, part, axis=1)
                order = np.argsort(-part_sim, axis=1)
                part = np.take_along_axis(part, order, axis=1)
                part_sim = np.take_along_axis(part_sim, order, axis=1)
                q = np.exp((part_sim / TAU) - np.max(part_sim / TAU, axis=1, keepdims=True))
                q = q / np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
                idx[start:end] = part
                probs[start:end] = q.astype(np.float32)
                print(f"[XLVIII] teacher E1 neighbours {end}/{n}", flush=True)
    else:
        e1 = np.load(OUT / "cache" / "teacher_neighbors_E1_TRUE_DINO_NEIGHBOUR.npz")
        probs = e1["probs"].astype(np.float32)
        rng = np.random.default_rng(SEED)
        for i in range(n):
            pool = np.flatnonzero(img != img[i])
            idx[i] = rng.choice(pool, size=K, replace=False)
        print("[XLVIII] E2 random neighbour identities built", flush=True)
    np.savez_compressed(path, idx=idx, probs=probs, mode=mode, K=K, tau=TAU)
    return np.load(path)


def train_variant(variant, x_np, y_np, neigh_idx, neigh_probs, states, device):
    ckpt = OUT / "checkpoints" / f"{variant}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        model = AdapterToState(x_np.shape[1], states).to(device).eval()
        model.load_state_dict(payload["model"], strict=True)
        return model, payload["curves"]
    torch.manual_seed(SEED)
    model = AdapterToState(x_np.shape[1], states).to(device)
    x_all_np = norm_np(x_np.astype(np.float32))
    y_all_np = y_np.astype(np.float32)
    rng = np.random.default_rng(SEED)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    curves = []
    x_all_cpu = torch.tensor(x_all_np, dtype=torch.float32)
    for epoch in range(1, EPOCHS + 1):
        order = rng.permutation(len(y_all_np))
        vals = []
        for start in range(0, len(order), BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            x = x_all_cpu[ids].to(device)
            y = torch.tensor(y_all_np[ids], dtype=torch.float32, device=device)
            nb = neigh_idx[ids]
            q = torch.tensor(neigh_probs[ids], dtype=torch.float32, device=device)
            nb_flat = torch.tensor(nb.reshape(-1), dtype=torch.long)
            x_nb = x_all_cpu[nb_flat].to(device)
            opt.zero_grad()
            logits, z = model.logits(x)
            base, bce, dice = balanced_loss(logits, y)
            z_nb = model.adapt(x_nb).reshape(len(ids), K, -1)
            sim = torch.einsum("bd,bkd->bk", z, z_nb)
            log_p = F.log_softmax(sim / TAU, dim=1)
            struct = F.kl_div(log_p, q, reduction="batchmean")
            loss = base + LAMBDA_STRUCT * struct
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(base.detach().cpu()), float(struct.detach().cpu()), float(bce.cpu()), float(dice.cpu())))
        arr = np.asarray(vals, dtype=np.float64)
        row = {
            "variant": variant,
            "epoch": epoch,
            "loss": float(arr[:, 0].mean()),
            "base_loss": float(arr[:, 1].mean()),
            "struct_loss": float(arr[:, 2].mean()),
            "bce": float(arr[:, 3].mean()),
            "dice": float(arr[:, 4].mean()),
        }
        curves.append(row)
        print(f"[XLVIII] {variant} epoch {epoch}/{EPOCHS} loss {row['loss']:.4f} struct {row['struct_loss']:.4f}", flush=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "curves": curves}, ckpt)
    return model.eval(), curves


def metric_row(target, variant, mean):
    row = {"target": target, "variant": variant}
    for key in ["I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "mask_images"]:
        row[key] = safe_float(mean.get(key, float("nan")))
    row["G_pixel"] = gp(row)
    return row


def evaluate_btad(adapter, device):
    dataset = ExternalDataset(DATA_ROOT / "btad" / "BTech_Dataset_transformed", "BTAD", mode="test")
    cache = np.load(XLV_A / "feature_cache" / "BTAD_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1_cached(cache["patch"], adapter, device)
    mean, per = evaluate_external(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)
    return mean, per


def evaluate_clinicdb(adapter, device):
    dataset = MedicalDataset(DATA_ROOT / "CVC-ClinicDB", mode="test")
    cache = np.load(XLV_B / "feature_cache" / "CVC-ClinicDB_clip.npz", allow_pickle=True)
    image_scores, maps = score_v1_cached(cache["patch"], adapter, device)
    metric, _per_img, _pixel = eval_scores(dataset, [dict(r) for r in cache["rows"]], image_scores, maps)
    return metric, {}


def top1_image_score(map37):
    flat = np.asarray(map37, dtype=np.float32).reshape(-1)
    k = max(1, int(flat.size * 0.01))
    return float(np.sort(flat)[-k:].mean())


def score_v1_cached(patch, adapter, device):
    maps = np.zeros((patch.shape[0], 37, 37), dtype=np.float16)
    image_scores = np.zeros((patch.shape[0],), dtype=np.float32)
    with torch.no_grad():
        for i in range(patch.shape[0]):
            x = torch.tensor(norm_np(np.asarray(patch[i], dtype=np.float32)), dtype=torch.float32, device=device)
            logits, _ = adapter.logits(x)
            m = torch.sigmoid(logits).detach().cpu().numpy().reshape(37, 37).astype(np.float32)
            maps[i] = m.astype(np.float16)
            image_scores[i] = top1_image_score(m)
    return image_scores, maps


def e0_rows():
    rows = []
    for path in [XLV_A / "PILOT_METRICS.csv", XLV_B / "RAW_METRICS.csv"]:
        with open(path, newline="", encoding="utf-8") as fp:
            for r in csv.DictReader(fp):
                if r["target"] not in {"BTAD", "CVC-ClinicDB"} or r["variant"] != "V1_STATE_ALIGNED":
                    continue
                rows.append({
                    "target": r["target"],
                    "variant": "E0_STATE_ALIGNED",
                    "I-AUROC": safe_float(r["I-AUROC"]),
                    "I-AP": safe_float(r["I-AP"]),
                    "P-AUROC": safe_float(r["P-AUROC"]),
                    "P-AP": safe_float(r["P-AP"]),
                    "PRO": safe_float(r["PRO"]),
                    "G_pixel": 0.5 * (safe_float(r["P-AUROC"]) + safe_float(r["P-AP"])),
                })
    return rows


def comparison_rows(rows):
    by = {(r["target"], r["variant"]): r for r in rows if "status" not in r}
    out = []
    for target in ["BTAD", "CVC-ClinicDB"]:
        e0 = by[(target, "E0_STATE_ALIGNED")]
        for variant in ["E1_DINO_STRUCTURAL_DISTILLATION", "E2_RANDOM_NEIGHBOUR_CONTROL"]:
            if (target, variant) not in by:
                continue
            e = by[(target, variant)]
            out.append({
                "target": target,
                "comparison": f"{variant}_minus_E0",
                "delta_P-AUROC": safe_float(e["P-AUROC"]) - safe_float(e0["P-AUROC"]),
                "delta_P-AP": safe_float(e["P-AP"]) - safe_float(e0["P-AP"]),
                "delta_G_pixel": gp(e) - gp(e0),
            })
        if (target, "E2_RANDOM_NEIGHBOUR_CONTROL") in by:
            e1 = by[(target, "E1_DINO_STRUCTURAL_DISTILLATION")]
            e2 = by[(target, "E2_RANDOM_NEIGHBOUR_CONTROL")]
            out.append({
                "target": target,
                "comparison": "E1_minus_E2",
                "delta_P-AUROC": safe_float(e1["P-AUROC"]) - safe_float(e2["P-AUROC"]),
                "delta_P-AP": safe_float(e1["P-AP"]) - safe_float(e2["P-AP"]),
                "delta_G_pixel": gp(e1) - gp(e2),
            })
    return out


def decide(rows, controls_run):
    by = {(r["target"], r["variant"]): r for r in rows}
    e0b = by[("BTAD", "E0_STATE_ALIGNED")]
    e1b = by[("BTAD", "E1_DINO_STRUCTURAL_DISTILLATION")]
    e0c = by[("CVC-ClinicDB", "E0_STATE_ALIGNED")]
    e1c = by[("CVC-ClinicDB", "E1_DINO_STRUCTURAL_DISTILLATION")]
    btad_g = gp(e1b) - gp(e0b)
    btad_pap = safe_float(e1b["P-AP"]) - safe_float(e0b["P-AP"])
    clinic_g = gp(e1c) - gp(e0c)
    if btad_g < 0.5:
        return "XLVIII_DINO_STRUCTURAL_DISTILLATION_NO_SIGNAL"
    if btad_g >= 0.5 and clinic_g <= -1.0:
        return "XLVIII_DINO_STRUCTURAL_KNOWLEDGE_DOMAIN_SPECIFIC"
    if not (btad_g >= 1.0 and btad_pap > 0 and clinic_g >= -0.5):
        return "XLVIII_DINO_STRUCTURAL_DISTILLATION_NO_CONTINUATION"
    if not controls_run:
        return "XLVIII_IMPLEMENTATION_FAILURE"
    e2b = by[("BTAD", "E2_RANDOM_NEIGHBOUR_CONTROL")]
    e2c = by[("CVC-ClinicDB", "E2_RANDOM_NEIGHBOUR_CONTROL")]
    e1_e2_b = gp(e1b) - gp(e2b)
    e1_e2_c = gp(e1c) - gp(e2c)
    if e1_e2_b >= 0.5 and e1_e2_c >= -0.5:
        return "XLVIII_DINO_STRUCTURAL_NEIGHBOURHOOD_SUPPORTED"
    return "XLVIII_DINO_STRUCTURAL_IDENTITY_NOT_SPECIFIC"


def main():
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sample = build_source_training_sample()
    x_clip = sample["x_clip"].astype(np.float32)
    x_dino = sample["x_dino"].astype(np.float32)
    y = sample["y"].astype(np.int64)
    image_pos = sample["image_pos"].astype(np.int32)
    states = np.load(XLV_A / "source_model" / "full_visa_states.npz")["patch_states"]
    write_json(OUT / "TRAINING_PROTOCOL.json", {
        "source": "Full VisA",
        "clip_feature_cache": str(FEATURE_CACHE / "D2_fold0_clip.npz"),
        "dino_feature_cache": str(FEATURE_CACHE / "D2_fold0_dino.npz"),
        "K": K,
        "tau": TAU,
        "lambda_struct": LAMBDA_STRUCT,
        "rank": RANK,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "checkpoint_selection": "final_epoch_10_only",
        "target_data_used_for_training_or_tuning": False,
        "dino_at_inference": False,
        "memory_at_inference": False,
    })
    write_csv(OUT / "SOURCE_SAMPLE_AUDIT.csv", [{
        "source_patches": int(x_clip.shape[0]),
        "normal_patches": int((y == 0).sum()),
        "anomaly_patches": int((y == 1).sum()),
        "clip_dim": int(x_clip.shape[1]),
        "dino_dim": int(x_dino.shape[1]),
        "unique_source_images": int(np.unique(image_pos).size),
    }])

    e1_teacher = build_teacher_neighbors(x_dino, image_pos, "E1_TRUE_DINO_NEIGHBOUR")
    e1_model, e1_curves = train_variant(
        "E1_DINO_STRUCTURAL_DISTILLATION",
        x_clip,
        y,
        e1_teacher["idx"],
        e1_teacher["probs"],
        states,
        device,
    )
    raw = e0_rows()
    for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
        mean, _per = fn(e1_model, device)
        raw.append(metric_row(target, "E1_DINO_STRUCTURAL_DISTILLATION", mean))
    by = {(r["target"], r["variant"]): r for r in raw}
    btad_gate = {
        "BTAD_E1_minus_E0_G_pixel": gp(by[("BTAD", "E1_DINO_STRUCTURAL_DISTILLATION")]) - gp(by[("BTAD", "E0_STATE_ALIGNED")]),
        "BTAD_E1_minus_E0_P-AP": safe_float(by[("BTAD", "E1_DINO_STRUCTURAL_DISTILLATION")]["P-AP"]) - safe_float(by[("BTAD", "E0_STATE_ALIGNED")]["P-AP"]),
        "ClinicDB_E1_minus_E0_G_pixel": gp(by[("CVC-ClinicDB", "E1_DINO_STRUCTURAL_DISTILLATION")]) - gp(by[("CVC-ClinicDB", "E0_STATE_ALIGNED")]),
    }
    btad_gate["E1_continuation_pass"] = bool(
        btad_gate["BTAD_E1_minus_E0_G_pixel"] >= 1.0
        and btad_gate["BTAD_E1_minus_E0_P-AP"] > 0
        and btad_gate["ClinicDB_E1_minus_E0_G_pixel"] >= -0.5
    )
    btad_gate["E1_no_signal_stop"] = bool(btad_gate["BTAD_E1_minus_E0_G_pixel"] < 0.5)
    write_json(OUT / "E1_GATE.json", btad_gate)

    controls_run = False
    curves = list(e1_curves)
    if btad_gate["E1_continuation_pass"]:
        e2_teacher = build_teacher_neighbors(x_dino, image_pos, "E2_RANDOM_NEIGHBOUR")
        e2_model, e2_curves = train_variant(
            "E2_RANDOM_NEIGHBOUR_CONTROL",
            x_clip,
            y,
            e2_teacher["idx"],
            e2_teacher["probs"],
            states,
            device,
        )
        curves.extend(e2_curves)
        for target, fn in [("BTAD", evaluate_btad), ("CVC-ClinicDB", evaluate_clinicdb)]:
            mean, _per = fn(e2_model, device)
            raw.append(metric_row(target, "E2_RANDOM_NEIGHBOUR_CONTROL", mean))
        controls_run = True
    else:
        for target in ["BTAD", "CVC-ClinicDB"]:
            raw.append({"target": target, "variant": "E2_RANDOM_NEIGHBOUR_CONTROL", "status": "NOT_RUN_E1_GATE_FAILED"})

    comparisons = comparison_rows(raw)
    outcome = decide([r for r in raw if "status" not in r], controls_run)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "TRAINING_CURVES.csv", curves)
    write_csv(OUT / "CAUSAL_COMPARISON.csv", comparisons)
    write_csv(OUT / "LOW_RANK_UPDATE_GEOMETRY.csv", low_rank_rows({"E1_DINO_STRUCTURAL_DISTILLATION": e1_model}, "XLVIII", 0))
    write_json(OUT / "INFERENCE_AUDIT.json", {
        "DINO_AT_INFERENCE": False,
        "MEMORY_AT_INFERENCE": False,
        "TEXT_AT_INFERENCE": False,
        "DYNAMIC_MEMORY": False,
        "TARGET_ADAPTATION": False,
        "inference_trainable_parameters": 8192,
        "same_as_StateAligned": True,
    })
    report = ["# Phase XLVIII DINO Memory Structural Distillation", "", f"Outcome: **{outcome}**", "", "## E1 Gate", ""]
    report.append(f"- BTAD E1-E0 G: {btad_gate['BTAD_E1_minus_E0_G_pixel']:+.4f}; P-AP: {btad_gate['BTAD_E1_minus_E0_P-AP']:+.4f}.")
    report.append(f"- ClinicDB E1-E0 G: {btad_gate['ClinicDB_E1_minus_E0_G_pixel']:+.4f}.")
    report.extend(["", "## Metrics", ""])
    for r in raw:
        if "status" in r:
            report.append(f"- {r['target']} {r['variant']}: {r['status']}.")
        else:
            report.append(f"- {r['target']} {r['variant']}: P-AUROC {safe_float(r['P-AUROC']):.4f}, P-AP {safe_float(r['P-AP']):.4f}, G {safe_float(r['G_pixel']):.4f}.")
    report.extend(["", "DINO and memory are training-only teachers; final inference remains the rank-4 StateAligned adapter with frozen s_N/s_A.", "", f"Runtime: {time.time() - start:.1f}s.", "", "Stop: no K/lambda/layer search and no fusion experiment was started."])
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(OUT / "OUTCOME.json", {
        "phase": "XLVIII",
        "outcome": outcome,
        "controls_run": controls_run,
        "E1_gate": btad_gate,
        "runtime_sec": time.time() - start,
    })
    with (ROOT / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as fp:
        fp.write("\n\n## Phase XLVIII DINO Memory Structural Distillation\n")
        fp.write(f"- Outcome: {outcome}.\n")
        fp.write(f"- E1 gate: {json.dumps(btad_gate, default=json_default)}.\n")
    print(json.dumps(read_json(OUT / "OUTCOME.json"), indent=2), flush=True)


if __name__ == "__main__":
    main()
