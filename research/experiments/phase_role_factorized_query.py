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
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_defect_query_transfer_pilot import (
    CHECKPOINT,
    EPOCHS,
    LR,
    OUT as PREV_OUT,
    SEED,
    SIDE,
    defect_name_from_mvtec,
    dice_loss,
    light_fold,
    make_mvec_target,
    make_source_records,
    project_patch_image,
    resize_binary_mask,
    safe_ap,
    safe_auc,
    text_embeddings,
)
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv


OUT = Path("results/phase_role_factorized_query")


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


class SharedAnomalyQuery(nn.Module):
    def __init__(self, dim):
        super().__init__()
        torch.manual_seed(SEED)
        self.q = nn.Parameter(torch.randn(dim) * 0.02)

    def forward(self):
        return self.q / self.q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def train_shared_query(records, patch, patch_proj, device):
    torch.manual_seed(SEED)
    random.seed(SEED)
    qmod = SharedAnomalyQuery(768).to(device)
    opt = torch.optim.AdamW(qmod.parameters(), lr=LR)
    rng = random.Random(SEED)
    curves = []
    for epoch in range(EPOCHS):
        order = list(range(len(records)))
        rng.shuffle(order)
        total = 0.0
        for rid in order:
            rec = records[rid]
            z = project_patch_image(patch_proj, patch[rec["pos"]], device).detach()
            logits = z @ qmod()
            y = torch.tensor(rec["mask"], dtype=torch.float32, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        curves.append({"epoch": epoch + 1, "loss": total / max(1, len(order))})
        print(f"[RoleQuery] epoch {epoch+1}/{EPOCHS} loss={curves[-1]['loss']:.4f}", flush=True)
    return qmod.eval(), curves


def eval_role_factorized(dataset, indices, patch, vocab, text_by_name, qmod, patch_proj, device):
    raw_y, raw_s = [], []
    recog_true, recog_pred = [], []
    per_def = defaultdict(lambda: {"true": [], "pred": [], "y": [], "s": []})
    conf_names = sorted({n for vals in vocab.values() for n in vals})
    with torch.no_grad():
        q_a = qmod()
    for pos, idx in enumerate(indices):
        meta = dataset.data_all[idx]
        if int(meta["anomaly"]) == 0:
            continue
        obj = meta["cls_name"]
        gt = defect_name_from_mvtec(meta["specie_name"])
        candidates = vocab.get(obj, [])
        if gt not in candidates or not candidates:
            continue
        item = dataset[idx]
        mask = resize_binary_mask(item)
        z = project_patch_image(patch_proj, patch[pos], device)
        with torch.no_grad():
            logits = z @ q_a
            weights = torch.softmax(logits.float(), dim=0)
            r = (weights[:, None] * z.float()).sum(dim=0)
            r = r / r.norm().clamp_min(1e-12)
            t = torch.stack([text_by_name[n] for n in candidates], dim=0).to(device)
            t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            scores = (t @ r).detach().cpu().numpy()
            pred = candidates[int(np.argmax(scores))]
            amap = logits.detach().cpu().numpy()
        raw_y.append(mask)
        raw_s.append(amap)
        recog_true.append(gt)
        recog_pred.append(pred)
        d = per_def[gt]
        d["true"].append(gt)
        d["pred"].append(pred)
        d["y"].append(mask)
        d["s"].append(amap)
    y = np.concatenate(raw_y) if raw_y else np.asarray([], dtype=np.uint8)
    s = np.concatenate(raw_s) if raw_s else np.asarray([], dtype=np.float32)
    metric = {
        "variant": "E2_ROLE_FACTORIZED_QUERY",
        "Top1_Accuracy": float(accuracy_score(recog_true, recog_pred) * 100.0) if recog_true else float("nan"),
        "Macro_F1": float(f1_score(recog_true, recog_pred, average="macro", zero_division=0) * 100.0) if recog_true else float("nan"),
        "P-AUROC": safe_auc(y, s),
        "P-AP": safe_ap(y, s),
        "evaluated_anomaly_images": len(recog_true),
    }
    per_rows = []
    for name, d in sorted(per_def.items()):
        yy = np.concatenate(d["y"])
        ss = np.concatenate(d["s"])
        per_rows.append({
            "variant": "E2_ROLE_FACTORIZED_QUERY",
            "defect_name": name,
            "n": len(d["true"]),
            "Top1_Accuracy": float(accuracy_score(d["true"], d["pred"]) * 100.0),
            "P-AUROC": safe_auc(yy, ss),
            "P-AP": safe_ap(yy, ss),
        })
    cm = confusion_matrix(recog_true, recog_pred, labels=conf_names) if recog_true else np.zeros((len(conf_names), len(conf_names)), dtype=int)
    cm_rows = []
    for i, a in enumerate(conf_names):
        for j, b in enumerate(conf_names):
            if cm[i, j]:
                cm_rows.append({"variant": "E2_ROLE_FACTORIZED_QUERY", "true": a, "pred": b, "count": int(cm[i, j])})
    return metric, per_rows, cm_rows


def previous_rows(smoke):
    prev = PREV_OUT / ("smoke" if smoke else "formal") / "RAW_METRICS.csv"
    rows = read_csv(prev)
    out = []
    for r in rows:
        if r["variant"] == "C0_DIRECT_CLIP_TEXT":
            out.append({
                "variant": "E0_DIRECT_CLIP_TEXT",
                "Top1_Accuracy": float(r["Top1_Accuracy"]),
                "Macro_F1": float(r["Macro_F1"]),
                "P-AUROC": float(r["E2E_P-AUROC"]),
                "P-AP": float(r["E2E_P-AP"]),
                "Oracle_P-AP_reference": float(r["Oracle_P-AP"]),
                "evaluated_anomaly_images": int(r["evaluated_anomaly_images"]),
            })
        elif r["variant"] == "C1_SOURCE_TRAINED_W":
            out.append({
                "variant": "E1_SOURCE_TRAINED_W",
                "Top1_Accuracy": float(r["Top1_Accuracy"]),
                "Macro_F1": float(r["Macro_F1"]),
                "P-AUROC": float(r["E2E_P-AUROC"]),
                "P-AP": float(r["E2E_P-AP"]),
                "Oracle_P-AP_reference": float(r["Oracle_P-AP"]),
                "evaluated_anomaly_images": int(r["evaluated_anomaly_images"]),
            })
    return out


def decide(rows):
    by = {r["variant"]: r for r in rows}
    e1, e2 = by["E1_SOURCE_TRAINED_W"], by["E2_ROLE_FACTORIZED_QUERY"]
    loc_kept = float(e2["P-AP"]) >= float(e1["P-AP"]) - 3.0
    rec_gain = (float(e2["Top1_Accuracy"]) >= float(e1["Top1_Accuracy"]) + 3.0) or (float(e2["Macro_F1"]) >= float(e1["Macro_F1"]) + 3.0)
    if loc_kept and rec_gain:
        outcome = "ROLE_FACTORIZED_QUERY_SUPPORTED"
    elif loc_kept and not rec_gain:
        outcome = "LOCALIZATION_OPERATOR_ONLY"
    else:
        outcome = "DEFECT_CONDITIONING_REQUIRED_FOR_LOCALIZATION"
    return outcome, {
        "E2_minus_E1_Top1": float(e2["Top1_Accuracy"]) - float(e1["Top1_Accuracy"]),
        "E2_minus_E1_MacroF1": float(e2["Macro_F1"]) - float(e1["Macro_F1"]),
        "E2_minus_E1_PAP": float(e2["P-AP"]) - float(e1["P-AP"]),
        "localization_kept": loc_kept,
        "recognition_gain": rec_gain,
    }


def write_report(out, outcome, gate, rows):
    lines = ["# Role-Factorized Defect Query", "", f"Outcome: `{outcome}`", "", "| Variant | Top1 | Macro-F1 | P-AUROC | P-AP |", "|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['variant']} | {float(r['Top1_Accuracy']):.4f} | {float(r['Macro_F1']):.4f} | {float(r['P-AUROC']):.4f} | {float(r['P-AP']):.4f} |")
    lines.extend(["", "## Gate", "", json.dumps(gate, indent=2, default=json_default), "", "STOP: no prompt search, cross-attention, multi-query, object/relation query, Memory, DINO, or target tuning was started."])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    out = OUT / ("smoke" if args.smoke else "formal")
    out.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[RoleQuery] start smoke={args.smoke} device={device}", flush=True)
    _source_dataset, _source_indices, source_patch, records, source_names = make_source_records(args.smoke)
    target_dataset, target_indices, target_patch, target_vocab = make_mvec_target(args.smoke)
    model, _prompt, _image_proj, patch_proj, _prompt_proj = load_components(str(CHECKPOINT), device)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in patch_proj.parameters():
        p.requires_grad_(False)
    all_target_names = sorted({n for vals in target_vocab.values() for n in vals})
    txt = text_embeddings(model, all_target_names, device).detach().cpu()
    text_by_name = {n: txt[i] for i, n in enumerate(all_target_names)}
    qmod, curves = train_shared_query(records, source_patch, patch_proj, device)
    e2, per, cm = eval_role_factorized(target_dataset, target_indices, target_patch, target_vocab, text_by_name, qmod, patch_proj, device)
    rows = previous_rows(args.smoke) + [e2]
    outcome, gate = decide(rows)
    write_csv(out / "RAW_METRICS.csv", rows)
    write_csv(out / "PER_DEFECT.csv", per)
    write_csv(out / "CONFUSION_MATRIX.csv", cm)
    write_csv(out / "TRAIN_CURVE.csv", curves)
    write_json(out / "OUTCOME.json", {"outcome": outcome, "gate": gate, "smoke": args.smoke, "runtime_sec": time.time() - start, "E0_E1_reused_from_previous": True})
    write_report(out, outcome, gate, rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
