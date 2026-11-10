import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase_frozen_semantic_anchor_fusion import load_b0_rows
from experiments.phase_hard_normal_contrast import balanced_category_batch, collect_source_patch_pool
from experiments.phase_learnable_text_semantic_compatibility import load_clip_text_model
from experiments.phase_pc1_orthogonal_anomaly_sufficiency import source_label_maps
from experiments.phase_private_residual_contrast_memory import contrast_logits, unit_np
from experiments.phase_state_pair_decomposition import decompose, hybrid_states
from experiments.phase_statealigned_pc1_debias import evaluate_pc1_removed
from experiments.phase_statealigned_state_alignment_swap import make_model
from experiments.phase_statealigned_supervision_necessity import load_full_visa_source_lazy
from experiments.phase_transferable_contrast_memory_pilot import source_slots_by_category
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xxxviii_visual_state_representation import LR


OUT = Path("results/phase_semantic_warmstart_state_contrast/formal")
HN_OUT = Path("results/phase_hard_normal_contrast/formal")
TEXT_OUT = Path("results/phase_learnable_text_semantic_compatibility/formal")
PC1_DEBIAS_OUT = Path("results/phase_statealigned_pc1_debias/formal")
EPOCHS = 10
TOP_NEG_FRAC = 0.10
EPS = 1e-12


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


def read_csv_rows(path):
    with Path(path).open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def cos_np(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(a @ b / max(float(np.linalg.norm(a) * np.linalg.norm(b)), EPS))


def angle_deg(a, b):
    c = max(-1.0, min(1.0, cos_np(a, b)))
    return float(math.degrees(math.acos(c)))


def load_payloads(device):
    q_pc1 = unit_np(np.load(PC1_DEBIAS_OUT / "q_pc1_statealigned1024.npy").astype(np.float32))
    true_payload = torch.load(PC1_DEBIAS_OUT / "E1_PC1Removed-TrueMask_formal.pt", map_location=device, weights_only=False)
    corrupt_payload = torch.load(PC1_DEBIAS_OUT / "E2_PC1Removed-CorruptedMask_formal.pt", map_location=device, weights_only=False)
    states_t = np.asarray(true_payload["patch_states"], dtype=np.float32)
    states_c = np.asarray(corrupt_payload["patch_states"], dtype=np.float32)
    _m_t, d_t = decompose(states_t)
    m_c, _d_c = decompose(states_c)
    alpha = float(np.linalg.norm(d_t))
    return q_pc1, corrupt_payload, states_c, m_c.astype(np.float32), alpha


def text_direction_to_state_space(anchor_pair, visual_proj):
    d_text = unit_np(np.asarray(anchor_pair[1] - anchor_pair[0], dtype=np.float32))
    # Prior experiments compare state directions in text space via v @ visual_proj.
    # For initialization, use the corresponding least-extra-parameter backprojection.
    return unit_np(d_text @ np.asarray(visual_proj, dtype=np.float32).T)


def load_initial_directions(device):
    anchors = np.load(TEXT_OUT / "prompt_anchors.npz")
    clip_model, visual_proj, _scale = load_clip_text_model(device)
    del clip_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dirs = {
        "E1_LearnableSemanticWarmStart": text_direction_to_state_space(anchors["learnable"], visual_proj),
        "E2_FrozenSemanticWarmStart": text_direction_to_state_space(anchors["frozen"], visual_proj),
        "E3_RandomSemanticWarmStart": text_direction_to_state_space(anchors["random"], visual_proj),
    }
    rng = np.random.default_rng(SEED)
    dirs["E4_RandomDirectionWarmStart"] = unit_np(rng.standard_normal(dirs["E1_LearnableSemanticWarmStart"].shape).astype(np.float32))
    return dirs


def prepare_groups(q_pc1, states_c, corrupt_payload, device, smoke):
    align_model = make_model(states_c, corrupt_payload, device)
    _dataset_name, dataset, indices, _cls, patch, _y_img, _cat_img = load_full_visa_source_lazy(False)
    labels_by_pos, _corrupt, _audit = source_label_maps(False)
    slots = source_slots_by_category(dataset, indices, smoke)
    return collect_source_patch_pool(slots, labels_by_pos, patch, align_model, q_pc1, device)


def train_warmstart_variant(variant, groups, m0, r0, init_v, b0_final_v, device, smoke):
    suffix = "smoke" if smoke else "formal"
    ckpt = OUT / f"{variant}_{suffix}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        return payload["v"].astype(np.float32), payload["trajectory"], True

    print(f"[SemanticWarmStart] train {variant}", flush=True)
    total_samples = sum(len(groups[n]["y"]) for n in groups)
    steps_per_epoch = max(1, int(np.ceil(total_samples / BATCH_SIZE)))
    if smoke:
        steps_per_epoch = min(steps_per_epoch, 3)
    model = torch.nn.Parameter(torch.tensor(init_v, dtype=torch.float32, device=device))
    m = torch.tensor(m0, dtype=torch.float32, device=device)
    opt = torch.optim.AdamW([model], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED)
    trajectory = []
    prev_v = unit_np(init_v)

    for epoch in range(1, EPOCHS + 1):
        vals = []
        for _step in range(steps_per_epoch):
            xb, yb, _origin, _cats = balanced_category_batch(groups, rng, BATCH_SIZE)
            z = torch.tensor(xb, dtype=torch.float32, device=device)
            y = torch.tensor(yb, dtype=torch.float32, device=device)
            v = F.normalize(model, dim=-1)
            logits = contrast_logits(z, m, r0, v)
            task_loss, bce, dice = balanced_loss(logits, y)
            neg_idx = torch.nonzero(y < 0.5, as_tuple=False).flatten()
            k = max(1, int(np.ceil(float(len(neg_idx)) * TOP_NEG_FRAC)))
            selected_idx = neg_idx[torch.topk(logits.detach()[neg_idx], k=k, largest=True).indices]
            hard_loss = F.softplus(logits[selected_idx]).mean()
            loss = task_loss + hard_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            vals.append((float(loss.detach().cpu()), float(task_loss.detach().cpu()), float(hard_loss.detach().cpu()), float(bce.detach().cpu()), float(dice.detach().cpu())))
        arr = np.asarray(vals, dtype=np.float32)
        with torch.no_grad():
            epoch_v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
        row = {
            "variant": variant,
            "epoch": epoch,
            "source_total_loss": float(arr[:, 0].mean()),
            "source_statealigned_loss": float(arr[:, 1].mean()),
            "source_hardnormal_loss": float(arr[:, 2].mean()),
            "source_bce": float(arr[:, 3].mean()),
            "source_dice": float(arr[:, 4].mean()),
            "cos_d_epoch_d_init": cos_np(epoch_v, init_v),
            "cos_d_epoch_d_B0_final": cos_np(epoch_v, b0_final_v),
            "angle_d_epoch_prev_deg": angle_deg(epoch_v, prev_v),
        }
        trajectory.append(row)
        prev_v = epoch_v
        print(
            f"[SemanticWarmStart] {variant} epoch {epoch}/{EPOCHS} "
            f"loss={row['source_total_loss']:.4f} hard={row['source_hardnormal_loss']:.4f} "
            f"cosB0={row['cos_d_epoch_d_B0_final']:.4f}",
            flush=True,
        )

    with torch.no_grad():
        v = F.normalize(model, dim=-1).detach().cpu().numpy().astype(np.float32)
    torch.save({"v": v, "trajectory": trajectory, "init_v": np.asarray(init_v, dtype=np.float32)}, ckpt)
    return v, trajectory, False


def load_b0_as_original_init():
    raw, percat = load_b0_rows()
    for row in raw:
        row["variant"] = "B0_HardNormalOriginalInit"
    for row in percat:
        row["variant"] = "B0_HardNormalOriginalInit"
    return raw, percat


def summarize(raw):
    by = {(r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for ds in ("MVTec", "BTAD", "CVC-ClinicDB"):
        b0 = by[(ds, "B0_HardNormalOriginalInit")]
        e1 = by[(ds, "E1_LearnableSemanticWarmStart")]
        e2 = by[(ds, "E2_FrozenSemanticWarmStart")]
        e3 = by[(ds, "E3_RandomSemanticWarmStart")]
        e4 = by[(ds, "E4_RandomDirectionWarmStart")]
        rows.append({
            "dataset": ds,
            "B0_G": float(b0["G_pixel"]),
            "E1_G": float(e1["G_pixel"]),
            "E2_G": float(e2["G_pixel"]),
            "E3_G": float(e3["G_pixel"]),
            "E4_G": float(e4["G_pixel"]),
            "E1_minus_B0_G": float(e1["G_pixel"]) - float(b0["G_pixel"]),
            "E1_minus_E2_G": float(e1["G_pixel"]) - float(e2["G_pixel"]),
            "E1_minus_E3_G": float(e1["G_pixel"]) - float(e3["G_pixel"]),
            "E1_minus_E4_G": float(e1["G_pixel"]) - float(e4["G_pixel"]),
            "E1_minus_B0_P-AUROC": float(e1["P-AUROC"]) - float(b0["P-AUROC"]),
            "E1_minus_B0_P-AP": float(e1["P-AP"]) - float(b0["P-AP"]),
        })
    return rows


def initialization_geometry(init_dirs, final_dirs, b0_init, b0_final):
    rows = []
    for variant in ("B0_HardNormalOriginalInit", "E1_LearnableSemanticWarmStart", "E2_FrozenSemanticWarmStart", "E3_RandomSemanticWarmStart", "E4_RandomDirectionWarmStart"):
        init_v = b0_init if variant == "B0_HardNormalOriginalInit" else init_dirs[variant]
        final_v = b0_final if variant == "B0_HardNormalOriginalInit" else final_dirs[variant]
        rows.append({
            "variant": variant,
            "cos_init_B0_init": cos_np(init_v, b0_init),
            "cos_init_B0_final": cos_np(init_v, b0_final),
            "cos_final_B0_final": cos_np(final_v, b0_final),
            "semantic_retention_cos_final_init": cos_np(final_v, init_v),
            "B0_convergence_cos_final_B0_final": cos_np(final_v, b0_final),
            "init_norm": float(np.linalg.norm(init_v)),
            "final_norm": float(np.linalg.norm(final_v)),
        })
    names = list(init_dirs)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            rows.append({
                "variant": f"pair:{a}|{b}",
                "cos_init_B0_init": float("nan"),
                "cos_init_B0_final": float("nan"),
                "cos_final_B0_final": cos_np(final_dirs[a], final_dirs[b]),
                "semantic_retention_cos_final_init": cos_np(init_dirs[a], init_dirs[b]),
                "B0_convergence_cos_final_B0_final": float("nan"),
                "init_norm": 1.0,
                "final_norm": 1.0,
            })
    return rows


def decide(summary_rows, geometry_rows):
    e1_b0 = np.asarray([r["E1_minus_B0_G"] for r in summary_rows], dtype=np.float32)
    e1_e2 = np.asarray([r["E1_minus_E2_G"] for r in summary_rows], dtype=np.float32)
    e1_e3 = np.asarray([r["E1_minus_E3_G"] for r in summary_rows], dtype=np.float32)
    e1_e4 = np.asarray([r["E1_minus_E4_G"] for r in summary_rows], dtype=np.float32)
    by = {r["dataset"]: r for r in summary_rows}
    geom = {r["variant"]: r for r in geometry_rows if not r["variant"].startswith("pair:")}
    e1_final_b0 = float(geom["E1_LearnableSemanticWarmStart"]["cos_final_B0_final"])
    init_variants = ["E1_LearnableSemanticWarmStart", "E2_FrozenSemanticWarmStart", "E3_RandomSemanticWarmStart", "E4_RandomDirectionWarmStart"]
    init_scores = np.asarray([[r[f"{v[:2]}_G"] for v in ("E1", "E2", "E3", "E4")] for r in summary_rows], dtype=np.float32)
    mean_init_std = float(np.mean(np.std(init_scores, axis=1)))

    success = (
        int(np.sum(e1_b0 > 0)) >= 2
        and float(e1_b0.mean()) >= 1.0
        and max(float(by["MVTec"]["E1_minus_B0_G"]), float(by["CVC-ClinicDB"]["E1_minus_B0_G"])) >= 1.5
        and float(e1_e3.mean()) >= 0.5
        and float(e1_e4.mean()) >= 0.5
        and float(e1_e2.mean()) >= 0.0
        and float(by["BTAD"]["E1_minus_B0_G"]) >= -1.0
    )
    if success:
        outcome = "LEARNED_SEMANTIC_INITIALIZATION_IMPROVES_VISUAL_TRANSFER"
    elif e1_final_b0 >= 0.995 and abs(float(e1_b0.mean())) < 0.5:
        outcome = "SEMANTIC_INITIALIZATION_FORGOTTEN"
    elif abs(float(e1_e3.mean())) < 0.5 and abs(float(e1_e4.mean())) < 0.5:
        outcome = "STATEALIGNED_INITIALIZATION_SENSITIVE_NOT_SEMANTIC_SPECIFIC"
    elif float(e1_e3.mean()) >= 0.5 and float(e1_e4.mean()) >= 0.5 and float(e1_b0.mean()) < 1.0:
        outcome = "SEMANTIC_INITIALIZATION_SIGNAL_NO_TARGET_GAIN"
    elif mean_init_std >= 3.0:
        outcome = "STATE_OPTIMIZATION_BASIN_UNSTABLE"
    else:
        outcome = "SEMANTIC_WARMSTART_NO_SIGNAL"
    return outcome, {
        "targets_E1_improved_over_B0": int(np.sum(e1_b0 > 0)),
        "mean_E1_minus_B0_G": float(e1_b0.mean()),
        "mean_E1_minus_E2_G": float(e1_e2.mean()),
        "mean_E1_minus_E3_G": float(e1_e3.mean()),
        "mean_E1_minus_E4_G": float(e1_e4.mean()),
        "MVTec_E1_minus_B0_G": float(by["MVTec"]["E1_minus_B0_G"]),
        "BTAD_E1_minus_B0_G": float(by["BTAD"]["E1_minus_B0_G"]),
        "ClinicDB_E1_minus_B0_G": float(by["CVC-ClinicDB"]["E1_minus_B0_G"]),
        "E1_final_cos_B0_final": e1_final_b0,
        "mean_target_std_across_warmstarts": mean_init_std,
    }


def write_sota_gap(raw):
    refs = {
        "MoECLIP OpenCLIP ViT-L/14-336 518x518": {
            "MVTec": {"P-AUROC": 92.5, "P-AP": 45.7, "G_pixel": 69.10},
            "BTAD": {"P-AUROC": 96.8, "P-AP": 50.4, "G_pixel": 73.60},
            "CVC-ClinicDB": {"P-AUROC": 89.7, "P-AP": 49.9, "G_pixel": 69.80},
        },
        "CVPR 2026 VisualAD(CLIP) pixel": {
            "MVTec": {"P-AUROC": 90.8, "P-AP": 41.2, "G_pixel": 66.00},
            "BTAD": {"P-AUROC": 91.1, "P-AP": 43.1, "G_pixel": 67.10},
            "CVC-ClinicDB": {"P-AUROC": 85.2, "P-AP": 37.8, "G_pixel": 61.50},
        },
    }
    rows = []
    for row in raw:
        if row["variant"] != "E1_LearnableSemanticWarmStart" or row["dataset"] not in refs["MoECLIP OpenCLIP ViT-L/14-336 518x518"]:
            continue
        for ref_name, table in refs.items():
            ref = table[row["dataset"]]
            rows.append({
                "dataset": row["dataset"],
                "variant": row["variant"],
                "comparison_status": "REFERENCE_ONLY unless exact backbone/preprocessing/evaluator parity is confirmed",
                "reference": ref_name,
                "current_P-AUROC": float(row["P-AUROC"]),
                "current_P-AP": float(row["P-AP"]),
                "current_G": float(row["G_pixel"]),
                "reference_P-AUROC": ref["P-AUROC"],
                "reference_P-AP": ref["P-AP"],
                "reference_G": ref["G_pixel"],
                "Delta_P_AUROC": float(row["P-AUROC"]) - ref["P-AUROC"],
                "Delta_P_AP": float(row["P-AP"]) - ref["P-AP"],
                "Delta_G": float(row["G_pixel"]) - ref["G_pixel"],
            })
    return rows


def write_report(summary_rows, stats, outcome, runtime):
    lines = [
        "# Semantic Warm-Start State Contrast",
        "",
        f"Outcome: `{outcome}`",
        "",
        "Protocol: Full VisA -> MVTec / BTAD / CVC-ClinicDB. Text is used only to initialize the state contrast direction; after initialization all variants train with the unchanged StateAligned + Symmetric HardNormal objective and visual-only inference.",
        "",
        "| Dataset | B0 G | E1 Learnable Init | E2 Frozen Init | E3 Random-Sem Init | E4 Random Dir | E1-B0 | E1-E3 | E1-E4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        lines.append(
            f"| {r['dataset']} | {r['B0_G']:.4f} | {r['E1_G']:.4f} | {r['E2_G']:.4f} | {r['E3_G']:.4f} | {r['E4_G']:.4f} | {r['E1_minus_B0_G']:+.4f} | {r['E1_minus_E3_G']:+.4f} | {r['E1_minus_E4_G']:+.4f} |"
        )
    lines.extend([
        "",
        "## Gate",
        f"- targets improved over B0: {stats['targets_E1_improved_over_B0']}/3",
        f"- mean E1-B0 G: {stats['mean_E1_minus_B0_G']:+.4f}",
        f"- mean E1-E2 G: {stats['mean_E1_minus_E2_G']:+.4f}",
        f"- mean E1-E3 G: {stats['mean_E1_minus_E3_G']:+.4f}",
        f"- mean E1-E4 G: {stats['mean_E1_minus_E4_G']:+.4f}",
        f"- E1 final cos to B0 final: {stats['E1_final_cos_B0_final']:.6f}",
        "",
        "## External SOTA Position",
        "`SOTA_GAP.csv` includes MoECLIP and VisualAD references. These are treated as reference-only unless exact backbone, preprocessing, scorer, and evaluator parity are confirmed; the causal decision here is against the internal B0/E2/E3/E4 controls.",
        "",
        f"Runtime: {runtime:.1f}s.",
    ])
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SemanticWarmStart] device={device}", flush=True)

    q_pc1, corrupt_payload, states_c, m0, r0 = load_payloads(device)
    hn_vectors = np.load(HN_OUT / "contrast_vectors.npz")
    b0_init_v = unit_np(hn_vectors["B0"].astype(np.float32))
    b0_final_v = unit_np(hn_vectors["E1"].astype(np.float32))
    init_dirs = load_initial_directions(device)
    groups, pool_rows = prepare_groups(q_pc1, states_c, corrupt_payload, device, args.smoke)

    final_dirs = {}
    trajectory = []
    reused = {}
    for variant in ("E1_LearnableSemanticWarmStart", "E2_FrozenSemanticWarmStart", "E3_RandomSemanticWarmStart", "E4_RandomDirectionWarmStart"):
        v, rows, was_reused = train_warmstart_variant(variant, groups, m0, r0, init_dirs[variant], b0_final_v, device, args.smoke)
        final_dirs[variant] = unit_np(v)
        trajectory.extend(rows)
        reused[variant] = bool(was_reused)

    models = {name: make_model(hybrid_states(m0, r0 * v), corrupt_payload, device) for name, v in final_dirs.items()}
    raw_new, per_new = evaluate_pc1_removed(models, q_pc1, device, args.smoke)
    raw_b0, per_b0 = load_b0_as_original_init()
    if args.smoke:
        raw_b0 = raw_b0[:3]
        per_b0 = per_b0[:3]
    raw = raw_b0 + raw_new
    percat = per_b0 + per_new
    summary_rows = summarize(raw)
    geom_rows = initialization_geometry(init_dirs, final_dirs, b0_init_v, b0_final_v)
    outcome, stats = decide(summary_rows, geom_rows)
    runtime = time.time() - start

    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY.csv", percat)
    write_csv(OUT / "STATE_TRAJECTORY.csv", trajectory)
    write_csv(OUT / "INITIALIZATION_GEOMETRY.csv", geom_rows)
    write_csv(OUT / "SOURCE_POOL_AUDIT.csv", pool_rows)
    write_csv(OUT / "CONTROL_COMPARISON.csv", summary_rows)
    write_csv(OUT / "SOTA_GAP.csv", write_sota_gap(raw))
    np.savez_compressed(
        OUT / "contrast_vectors.npz",
        B0_init=b0_init_v.astype(np.float32),
        B0_final=b0_final_v.astype(np.float32),
        E1=final_dirs["E1_LearnableSemanticWarmStart"].astype(np.float32),
        E2=final_dirs["E2_FrozenSemanticWarmStart"].astype(np.float32),
        E3=final_dirs["E3_RandomSemanticWarmStart"].astype(np.float32),
        E4=final_dirs["E4_RandomDirectionWarmStart"].astype(np.float32),
    )
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "device": device,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "text_used_after_initialization": False,
        "text_inference": False,
        "new_loss": False,
        "epochs": EPOCHS,
        "lr": LR,
        "seed": SEED,
        "top_negative_fraction": TOP_NEG_FRAC,
        "state_midpoint_source": "corrupted checkpoint midpoint used by formal contrast experiments",
        "state_magnitude": r0,
        "text_to_state_projection": "d_state = normalize(d_text @ visual.proj.T)",
        "checkpoints_reused": reused,
    })
    write_report(summary_rows, stats, outcome, runtime)
    print(json.dumps({"outcome": outcome, **stats, "runtime_sec": runtime}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
