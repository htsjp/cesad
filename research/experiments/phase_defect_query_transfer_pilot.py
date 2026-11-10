import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_liv_memory_support_modes import ArrayConcat, npz_array_mmap
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xa_checkpoint_source_readout import direction_cfg
from experiments.phase_xa_smoke import load_dataset
from experiments.phase_xxxviii_visual_state_representation import FEATURE_CACHE, norm_np
from models.prompt_learner import tokenize
from phase_xa.source_folds import deterministic_source_folds


OUT = Path("results/phase_defect_query_transfer_pilot")
DATA_ROOT = Path(r"E:\WorkSpace\dataset")
MVTEC_ROOT = DATA_ROOT / "mvtec"
TAXONOMY = Path("results/phase_xlvi_semantic_relation_distillation/SOURCE_TAXONOMY_FINAL.csv")
CHECKPOINT = Path("checkpoints/test_on_visa.pth")
SIDE = 37
IMAGE_SIZE = 518
SEED = 0
EPOCHS = 10
LR = 1e-3


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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader(f)]


def safe_auc(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(roc_auc_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def safe_ap(y, s):
    y = np.asarray(y, dtype=np.uint8)
    s = np.asarray(s, dtype=np.float32)
    return float(average_precision_score(y, s) * 100.0) if np.unique(y).size == 2 else float("nan")


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1.0 - (2.0 * inter + 1.0) / (p.sum() + target.sum() + 1.0)


def defect_name_from_mvtec(specie):
    return str(specie).replace("_", " ").replace("-", " ").strip()


def light_fold(direction, fold):
    dataset_name, data_path, _ckpt = direction_cfg(direction)
    dataset = load_dataset(dataset_name, data_path)
    rotation = deterministic_source_folds(dataset_name)["rotations"][fold]
    train_cats = set(rotation["train_categories"])
    held_cats = set(rotation["val_categories"])
    train_indices = [i for i, r in enumerate(dataset.data_all) if r["cls_name"] in train_cats]
    held_indices = [i for i, r in enumerate(dataset.data_all) if r["cls_name"] in held_cats]
    return dataset_name, dataset, train_indices, held_indices


def load_taxonomy():
    rows = read_csv(TAXONOMY)
    by_obj_id = {}
    names = []
    for r in rows:
        obj = r["object"]
        mid = int(r["mask_id"])
        name = r["defect_name"].strip()
        by_obj_id[(obj, mid)] = name
        names.append(name)
    return by_obj_id, sorted(set(names))


def resize_mask_ids(path):
    arr = np.asarray(Image.open(path).convert("L"))
    im = Image.fromarray(arr.astype(np.uint8), mode="L").resize((SIDE, SIDE), resample=Image.Resampling.NEAREST)
    return np.asarray(im, dtype=np.int64).reshape(-1)


def resize_binary_mask(item):
    mask = (item["img_mask"].detach().cpu().numpy() > 0.5).astype(np.float32)
    x = torch.tensor(mask, dtype=torch.float32).view(1, 1, *mask.shape[-2:])
    return F.interpolate(x, size=(SIDE, SIDE), mode="nearest").view(-1).numpy().astype(np.float32)


def text_embeddings(model, names, device):
    prompts = [f"a photo of an object with a {n} defect" for n in names]
    outs = []
    with torch.no_grad():
        for start in range(0, len(prompts), 16):
            tok = tokenize(prompts[start:start + 16])
            if tok.ndim == 2 and tok.shape[0] == 77 and tok.shape[1] != 77:
                tok = tok.T.contiguous()
            tok = tok.long().to(device)
            x = model.token_embedding(tok).type(model.dtype)
            x = x + model.positional_embedding.type(model.dtype)
            x = x.permute(1, 0, 2)
            state = [x, [], 0]
            for block in model.transformer.resblocks:
                state = block(state)
            x = state[0].permute(1, 0, 2)
            x = model.ln_final(x).type(model.dtype)
            emb = (x[torch.arange(x.shape[0], device=device), tok.argmax(dim=-1)] @ model.text_projection).float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            outs.append(emb)
    return torch.cat(outs, dim=0)


def project_patch_image(patch_proj, patch_img, device):
    z = torch.tensor(np.asarray(patch_img, dtype=np.float32), device=device)
    with torch.no_grad():
        out = patch_proj(z, 0).float()
        out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return out


def make_source_records(smoke):
    mapping, source_names = load_taxonomy()
    dataset_name, dataset, train_indices, held_indices = light_fold("D2", 0)
    cache = FEATURE_CACHE / "D2_fold0_clip.npz"
    patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
    indices = list(train_indices) + list(held_indices)
    records = []
    rng = random.Random(SEED)
    smoke_anom = 0
    smoke_norm = 0
    for pos, idx in enumerate(indices):
        item_meta = dataset.data_all[idx]
        obj = item_meta["cls_name"]
        if int(item_meta["anomaly"]) == 0:
            if not smoke or smoke_norm < 24:
                records.append({"pos": pos, "idx": idx, "defect_name": rng.choice(source_names), "mask": np.zeros((SIDE * SIDE,), dtype=np.float32), "normal": True})
                smoke_norm += 1
            if smoke and smoke_anom >= 24 and smoke_norm >= 24:
                break
            continue
        mask_path = Path(dataset.root) / item_meta["mask_path"]
        ids = resize_mask_ids(mask_path)
        for mid in sorted(int(x) for x in np.unique(ids) if int(x) > 0):
            name = mapping.get((obj, mid))
            if name and (not smoke or smoke_anom < 24):
                records.append({"pos": pos, "idx": idx, "defect_name": name, "mask": (ids == mid).astype(np.float32), "normal": False})
                smoke_anom += 1
        if smoke and smoke_anom >= 24 and smoke_norm >= 24:
            break
    return dataset, indices, patch, records, source_names


def make_mvec_target(smoke):
    dataset_name, dataset, train_indices, held_indices = light_fold("D1", 0)
    cache = FEATURE_CACHE / "D1_fold0_clip.npz"
    patch = ArrayConcat([npz_array_mmap(cache, "train_patch"), npz_array_mmap(cache, "held_patch")])
    indices = list(train_indices) + list(held_indices)
    if smoke:
        chosen = []
        for pos, idx in enumerate(indices):
            item_meta = dataset.data_all[idx]
            if int(item_meta["anomaly"]) == 1:
                chosen.append((pos, idx))
            if len(chosen) >= 32:
                break
        indices = [idx for _pos, idx in chosen]
        patch = np.stack([patch[pos] for pos, _idx in chosen], axis=0)
    vocab = defaultdict(set)
    for item in dataset.data_all:
        if int(item["anomaly"]) == 1:
            vocab[item["cls_name"]].add(defect_name_from_mvtec(item["specie_name"]))
    return dataset, indices, patch, {k: sorted(v) for k, v in vocab.items()}


class QueryProjector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.eye_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, t):
        q = self.linear(t)
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def train_w(records, patch, source_names, text_by_name, patch_proj, device):
    torch.manual_seed(SEED)
    random.seed(SEED)
    w = QueryProjector(text_by_name[source_names[0]].shape[-1]).to(device)
    opt = torch.optim.AdamW(w.parameters(), lr=LR)
    rng = random.Random(SEED)
    curves = []
    for epoch in range(EPOCHS):
        order = list(range(len(records)))
        rng.shuffle(order)
        total = 0.0
        for rid in order:
            rec = records[rid]
            z = project_patch_image(patch_proj, patch[rec["pos"]], device).detach()
            q = w(text_by_name[rec["defect_name"]].to(device).unsqueeze(0))[0]
            logits = z @ q
            y = torch.tensor(rec["mask"], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            neg = rec["defect_name"]
            while neg == rec["defect_name"]:
                neg = rng.choice(source_names)
            qn = w(text_by_name[neg].to(device).unsqueeze(0))[0]
            logit_neg = z @ qn
            loss = loss + F.binary_cross_entropy_with_logits(logit_neg, torch.zeros_like(y)) + dice_loss(logit_neg, torch.zeros_like(y))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        curves.append({"epoch": epoch + 1, "loss": total / max(len(order), 1)})
        print(f"[DefectQuery] epoch {epoch+1}/{EPOCHS} loss={curves[-1]['loss']:.4f}", flush=True)
    return w.eval(), curves


def query_bank(names, text_by_name, w, device, trained):
    t = torch.stack([text_by_name[n] for n in names], dim=0).to(device)
    with torch.no_grad():
        q = w(t) if trained else t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return q, {n: i for i, n in enumerate(names)}


def eval_variant(dataset, indices, patch, vocab, text_by_name, w, patch_proj, device, variant, trained):
    raw_y_oracle, raw_s_oracle, raw_y_e2e, raw_s_e2e = [], [], [], []
    recog_true, recog_pred = [], []
    per_def = defaultdict(lambda: {"true": [], "pred": [], "y_oracle": [], "s_oracle": [], "y_e2e": [], "s_e2e": []})
    conf_names = sorted({n for vals in vocab.values() for n in vals})
    for pos, idx in enumerate(indices):
        meta = dataset.data_all[idx]
        if int(meta["anomaly"]) == 0:
            continue
        obj = meta["cls_name"]
        gt = defect_name_from_mvtec(meta["specie_name"])
        candidates = vocab.get(obj, [])
        if not candidates:
            continue
        item = dataset[idx]
        mask = resize_binary_mask(item)
        z = project_patch_image(patch_proj, patch[pos], device)
        q, name_to_i = query_bank(candidates, text_by_name, w, device, trained)
        with torch.no_grad():
            maps = (z @ q.T).detach().cpu().numpy().T
        scores = np.asarray([top1_image_score(m.reshape(SIDE, SIDE)) for m in maps], dtype=np.float32)
        pred = candidates[int(np.argmax(scores))]
        if gt not in name_to_i:
            continue
        oracle = maps[name_to_i[gt]]
        e2e = maps[name_to_i[pred]]
        raw_y_oracle.append(mask); raw_s_oracle.append(oracle)
        raw_y_e2e.append(mask); raw_s_e2e.append(e2e)
        recog_true.append(gt); recog_pred.append(pred)
        d = per_def[gt]
        d["true"].append(gt); d["pred"].append(pred)
        d["y_oracle"].append(mask); d["s_oracle"].append(oracle)
        d["y_e2e"].append(mask); d["s_e2e"].append(e2e)
    y_o = np.concatenate(raw_y_oracle) if raw_y_oracle else np.asarray([], dtype=np.uint8)
    s_o = np.concatenate(raw_s_oracle) if raw_s_oracle else np.asarray([], dtype=np.float32)
    y_e = np.concatenate(raw_y_e2e) if raw_y_e2e else np.asarray([], dtype=np.uint8)
    s_e = np.concatenate(raw_s_e2e) if raw_s_e2e else np.asarray([], dtype=np.float32)
    metrics = {
        "variant": variant,
        "Top1_Accuracy": float(accuracy_score(recog_true, recog_pred) * 100.0) if recog_true else float("nan"),
        "Macro_F1": float(f1_score(recog_true, recog_pred, average="macro", zero_division=0) * 100.0) if recog_true else float("nan"),
        "Oracle_P-AUROC": safe_auc(y_o, s_o),
        "Oracle_P-AP": safe_ap(y_o, s_o),
        "E2E_P-AUROC": safe_auc(y_e, s_e),
        "E2E_P-AP": safe_ap(y_e, s_e),
        "evaluated_anomaly_images": len(recog_true),
    }
    per_rows = []
    for name, d in sorted(per_def.items()):
        yo = np.concatenate(d["y_oracle"])
        so = np.concatenate(d["s_oracle"])
        ye = np.concatenate(d["y_e2e"])
        se = np.concatenate(d["s_e2e"])
        per_rows.append({
            "variant": variant,
            "defect_name": name,
            "n": len(d["true"]),
            "Top1_Accuracy": float(accuracy_score(d["true"], d["pred"]) * 100.0),
            "Oracle_P-AUROC": safe_auc(yo, so),
            "Oracle_P-AP": safe_ap(yo, so),
            "E2E_P-AUROC": safe_auc(ye, se),
            "E2E_P-AP": safe_ap(ye, se),
        })
    cm = confusion_matrix(recog_true, recog_pred, labels=conf_names) if recog_true else np.zeros((len(conf_names), len(conf_names)), dtype=int)
    cm_rows = []
    for i, a in enumerate(conf_names):
        for j, b in enumerate(conf_names):
            if cm[i, j]:
                cm_rows.append({"variant": variant, "true": a, "pred": b, "count": int(cm[i, j])})
    return metrics, per_rows, cm_rows


def decide(c0, c1):
    d_top1 = c1["Top1_Accuracy"] - c0["Top1_Accuracy"]
    d_f1 = c1["Macro_F1"] - c0["Macro_F1"]
    d_oap = c1["Oracle_P-AP"] - c0["Oracle_P-AP"]
    signal = (d_top1 >= 3.0 or d_f1 >= 3.0) or d_oap >= 3.0
    return ("DEFECT_QUERY_TRANSFER_SIGNAL" if signal else "DEFECT_QUERY_TRANSFER_NO_SIGNAL", {"dTop1": d_top1, "dMacroF1": d_f1, "dOraclePAP": d_oap})


def write_report(out, outcome, deltas, raw):
    lines = ["# Source-Trained Defect Query Transfer Pilot", "", f"Outcome: `{outcome}`", "", "| Variant | Top1 | Macro-F1 | Oracle P-AUROC | Oracle P-AP | E2E P-AUROC | E2E P-AP |", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in raw:
        lines.append(f"| {r['variant']} | {r['Top1_Accuracy']:.4f} | {r['Macro_F1']:.4f} | {r['Oracle_P-AUROC']:.4f} | {r['Oracle_P-AP']:.4f} | {r['E2E_P-AUROC']:.4f} | {r['E2E_P-AP']:.4f} |")
    lines.extend(["", "## Answers", "", f"1. Source-trained query vs direct CLIP: deltas `{json.dumps(deltas, default=json_default)}`.", "2. Unseen MVTec semantics are assessed by Top1/Macro-F1 and oracle localization above.", "3. Recognition vs localization follows the delta split.", "4. Continue Query Memory/Operator only if outcome is signal.", "", "STOP: no template search, prompt learning, multi-query, cross-attention, memory, DINO, or target adaptation was started."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[DefectQuery] start smoke={args.smoke} device={device}", flush=True)
    print("[DefectQuery] prepare source records", flush=True)
    _source_dataset, _source_indices, source_patch, records, source_names = make_source_records(args.smoke)
    print(f"[DefectQuery] source records={len(records)} defect_types={len(source_names)}", flush=True)
    print("[DefectQuery] prepare target records", flush=True)
    target_dataset, target_indices, target_patch, target_vocab = make_mvec_target(args.smoke)
    print(f"[DefectQuery] target images={len(target_indices)} candidate_types={len(set(n for vals in target_vocab.values() for n in vals))}", flush=True)
    print("[DefectQuery] load frozen official components", flush=True)
    model, _prompt, _image_proj, patch_proj, _prompt_proj = load_components(str(CHECKPOINT), device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in patch_proj.parameters():
        p.requires_grad_(False)
    all_names = sorted(set(source_names) | {n for vals in target_vocab.values() for n in vals})
    print(f"[DefectQuery] encode text prompts={len(all_names)}", flush=True)
    txt = text_embeddings(model, all_names, device).detach().cpu()
    text_by_name = {n: txt[i] for i, n in enumerate(all_names)}
    w, curves = train_w(records, source_patch, source_names, text_by_name, patch_proj, device)
    raw, per, cm = [], [], []
    for variant, trained in [("C0_DIRECT_CLIP_TEXT", False), ("C1_SOURCE_TRAINED_W", True)]:
        print(f"[DefectQuery] evaluate {variant}", flush=True)
        m, pr, cr = eval_variant(target_dataset, target_indices, target_patch, target_vocab, text_by_name, w, patch_proj, device, variant, trained)
        raw.append(m); per.extend(pr); cm.extend(cr)
    c0, c1 = raw[0], raw[1]
    outcome, deltas = decide(c0, c1)
    loc = []
    for r in raw:
        loc.append({"variant": r["variant"], "mode": "oracle_type", "P-AUROC": r["Oracle_P-AUROC"], "P-AP": r["Oracle_P-AP"]})
        loc.append({"variant": r["variant"], "mode": "end_to_end", "P-AUROC": r["E2E_P-AUROC"], "P-AP": r["E2E_P-AP"]})
    write_csv(out / "RAW_METRICS.csv", raw)
    write_csv(out / "PER_DEFECT.csv", per)
    write_csv(out / "CONFUSION_MATRIX.csv", cm)
    write_csv(out / "LOCALIZATION_METRICS.csv", loc)
    write_csv(out / "TRAIN_CURVE.csv", curves)
    write_json(out / "OUTCOME.json", {"outcome": outcome, "deltas": deltas, "smoke": args.smoke, "runtime_sec": time.time() - start, "target_used_for_training": False})
    write_report(out, outcome, deltas, raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
