import argparse
import csv
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_spatial_supervision_transfer_audit import fast_patch_grid_eval, target_cells
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import remove_pc1
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_transferable_contrast_memory_pilot import target_cells_with_cls
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlv_cross_domain_transfer import top1_image_score
from experiments.phase_xxxviii_visual_state_representation import SIDE, norm_np

OUT = Path("results/phase_keyless_contrast_memory/formal")
TCM_OUT = Path("results/phase_transferable_contrast_memory_pilot/formal_corrected")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")


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


def load_payload(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def load_slot_vectors(prefix):
    data = np.load(TCM_OUT / "contrast_memory_vectors.npz")
    out = {}
    for key in sorted(data.files):
        if key.startswith(prefix):
            name = key[len(prefix) :]
            out[name] = unit(data[key])
    return out


def build_state_pairs(vectors, m_shared, alpha):
    names = list(vectors)
    states = []
    for name in names:
        states.append(hybrid_states(m_shared, alpha * vectors[name]))
    arr = np.stack(states, axis=0).astype(np.float32)
    return names, arr


@torch.no_grad()
def score_keyless_memory(patch, align_model, state_pairs, q_pc1, device, smoke, batch_images=8):
    names, states_np = state_pairs
    s = torch.tensor(states_np, dtype=torch.float32, device=device)
    s = F.normalize(s, dim=-1)
    n = min(24, patch.shape[0]) if smoke else patch.shape[0]
    maps = np.zeros((n, SIDE, SIDE), dtype=np.float16)
    image = np.zeros((n,), dtype=np.float32)
    slot_counts = Counter()
    for start in range(0, n, batch_images):
        end = min(n, start + batch_images)
        arr = remove_pc1(np.asarray(patch[start:end], dtype=np.float32), q_pc1)
        flat = norm_np(arr.reshape(-1, arr.shape[-1]))
        x = torch.tensor(flat, dtype=torch.float32, device=device)
        z = align_model.adapt(x)
        logits = z @ s[:, 1, :].T - z @ s[:, 0, :].T
        scores = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)
        winners = np.argmax(scores, axis=1)
        max_scores = scores[np.arange(scores.shape[0]), winners].reshape(end - start, SIDE, SIDE)
        for sid, count in Counter(winners.tolist()).items():
            slot_counts[names[int(sid)]] += int(count)
        for bi in range(max_scores.shape[0]):
            maps[start + bi] = max_scores[bi].astype(np.float16)
            image[start + bi] = top1_image_score(max_scores[bi])
    return image, maps, slot_counts


def usage_rows(dataset, variant, slot_counts, names):
    total = max(sum(slot_counts.values()), 1)
    probs = np.asarray([slot_counts.get(n, 0) / total for n in names], dtype=np.float64)
    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
    rows = [{"dataset": dataset, "variant": variant, "slot": "__entropy__", "patch_count": total, "fraction": 1.0, "entropy": entropy, "effective_slots": float(math.exp(entropy))}]
    for name in names:
        count = int(slot_counts.get(name, 0))
        rows.append({"dataset": dataset, "variant": variant, "slot": name, "patch_count": count, "fraction": count / total, "entropy": entropy, "effective_slots": float(math.exp(entropy))})
    return rows


def evaluate_keyless(variants, align_model, q_pc1, device, smoke):
    raw, percat, usage = [], [], []
    for cell_id, target, dataset, idx_or_rows, patch, kind, _cls in target_cells_with_cls(smoke):
        n = min(24, patch.shape[0]) if smoke else patch.shape[0]
        idx_use = idx_or_rows[:n]
        for variant, state_pairs in variants.items():
            print(f"[KeylessTCM] eval {variant} on {target}", flush=True)
            image, maps, counts = score_keyless_memory(patch, align_model, state_pairs, q_pc1, device, smoke)
            metric, per = fast_patch_grid_eval(cell_id, target, dataset, idx_use, kind, image, maps)
            metric.update({"dataset": target, "variant": variant, "G_pixel": 0.5 * (float(metric["P-AUROC"]) + float(metric["P-AP"]))})
            raw.append(metric)
            usage.extend(usage_rows(target, variant, counts, state_pairs[0]))
            for cat, vals in per.items():
                row = {"dataset": target, "category": cat, "variant": variant, **vals}
                row["G_pixel"] = 0.5 * (float(row["P-AUROC"]) + float(row["P-AP"])) if np.isfinite(float(row["P-AUROC"])) and np.isfinite(float(row["P-AP"])) else float("nan")
                percat.append(row)
    return raw, percat, usage


def load_baseline_rows():
    raw, percat = [], []
    wanted = {
        "B0_GlobalContrast_ReconstructedAlpha": "B0_GlobalContrast",
        "E1_TransferableContrastMemory": "B1_HardRoutedTCM",
    }
    with (TCM_OUT / "RAW_METRICS.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] not in wanted:
                continue
            out = dict(row)
            out["variant"] = wanted[row["variant"]]
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            raw.append(out)
    with (TCM_OUT / "PER_CATEGORY.csv").open("r", encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            if row["variant"] not in wanted:
                continue
            out = dict(row)
            out["variant"] = wanted[row["variant"]]
            for key in ("I-AUROC", "I-AP", "P-AUROC", "P-AP", "PRO", "G_pixel"):
                try:
                    out[key] = float(out[key])
                except ValueError:
                    out[key] = float("nan")
            out["mask_images"] = int(float(out["mask_images"]))
            percat.append(out)
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_GlobalContrast")]
        b1 = by[(ds, "B1_HardRoutedTCM")]
        e1 = by[(ds, "E1_TrueKeylessContrastMemory")]
        e2 = by[(ds, "E2_RandomKeylessContrastMemory")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "B1_G": float(b1["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_B1_G": float(e1["G_pixel"]) - float(b1["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
            "E1_minus_E2_P-AUROC": float(e1["P-AUROC"]) - float(e2["P-AUROC"]),
            "E1_minus_E2_P-AP": float(e1["P-AP"]) - float(e2["P-AP"]),
        })
    return rows


def decide(summary_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_b1 = np.asarray([r["E1_minus_B1_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    mean_b0 = float(e1_b0.mean())
    mean_b1 = float(e1_b1.mean())
    mean_e2 = float(e1_e2.mean())
    if mean_b0 >= 1.0 and mean_e2 >= 1.0:
        outcome = "CONTRAST_MEMORY_VALUE_SUPPORTED_ACCESS_WAS_BOTTLENECK"
    elif mean_b1 >= 3.0 and mean_b0 <= 0.0:
        outcome = "ACCESS_ERROR_CONFIRMED_BUT_GLOBAL_CONTRAST_STILL_BEST"
    elif abs(mean_b1) < 1.0:
        outcome = "HARD_ROUTING_NOT_MAIN_FAILURE"
    elif abs(mean_e2) < 1.0:
        outcome = "CONTRAST_MEMORY_SEMANTIC_VALUE_NOT_ROBUST"
    else:
        outcome = "KEYLESS_CONTRAST_MEMORY_INCONCLUSIVE"
    return outcome, {"mean_E1_minus_B0_G": mean_b0, "mean_E1_minus_B1_G": mean_b1, "mean_E1_minus_E2_G": mean_e2}


def write_report(summary_rows, outcome, stats, runtime):
    lines = [
        "# Keyless Transferable Contrast Memory Audit",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: reuses corrected TCM contrast slots. Key/category routing is removed; each target patch takes the max StateAligned score over all source contrast directions. No retraining, target statistics, target memory, category names, or defect names.",
        "",
        "| Dataset | B0 Global G | B1 HardRouted G | E1 True Keyless G | E2 Random Keyless G | E1-B0 | E1-B1 | E1-E2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(f"| {r['dataset']} | {r['B0_G']:.4f} | {r['B1_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_B1_G']:+.4f} | {r['E1_minus_E2_G']:+.4f} |")
    lines.extend([
        "",
        f"Mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f} pp.",
        f"Mean E1-B1 G: {stats['mean_E1_minus_B1_G']:+.4f} pp.",
        f"Mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f} pp.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[KeylessTCM] device={device}", flush=True)
    corrupt_payload = load_payload(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", device)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    m_c, _d_c = decompose(states_c)
    true_payload = load_payload(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", device)
    _m_t, d_t = decompose(np.asarray(true_payload["patch_states"], dtype=np.float32))
    alpha = float(np.linalg.norm(d_t))
    q_pc1 = np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32)
    q_pc1 = q_pc1 / max(float(np.linalg.norm(q_pc1)), 1e-12)
    align_model = make_model(states_c, corrupt_payload, device)
    true_vectors = load_slot_vectors("E1_")
    random_vectors = load_slot_vectors("E2_")
    variants = {
        "E1_TrueKeylessContrastMemory": build_state_pairs(true_vectors, m_c, alpha),
        "E2_RandomKeylessContrastMemory": build_state_pairs(random_vectors, m_c, alpha),
    }
    raw, percat, usage = evaluate_keyless(variants, align_model, q_pc1, device, args.smoke)
    base_raw, base_percat = load_baseline_rows()
    if args.smoke:
        base_raw = base_raw[:6]
        base_percat = base_percat[:6]
    raw = base_raw + raw
    percat = base_percat + percat
    summary_rows = summarize(raw)
    outcome, decision_stats = decide(summary_rows)
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "GROUP_METRICS.csv", summary_rows)
    write_csv(OUT / "WINNING_SLOT_USAGE.csv", usage)
    runtime = time.time() - start
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": decision_stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_memory": False,
        "alpha": alpha,
        "source_slots": len(true_vectors),
        "random_slots": len(random_vectors),
        "tcm_source": str(TCM_OUT),
    })
    write_report(summary_rows, outcome, decision_stats, runtime)
    print(f"[KeylessTCM] OUTCOME={outcome}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
