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

from experiments.phase_cmab_compact_multi_anomaly_basis import load_mvtec_groups, load_visa_groups
from experiments.phase_final_cross_dataset_benchmark import load_mvtec_source, load_visa_source
from experiments.phase_hard_normal_contrast import TOP_NEG_FRAC
from experiments.phase_snir_source_nuisance_invariant_state_regularization import (
    contrast_direction_np,
    evaluate_pairs,
    fnum,
    finite_mean,
    source_state_pair,
)
from experiments.phase_ssec_state_space_experience_consolidation import two_state_logits
from experiments.phase_trsl_tail_ranked_state_learning import hardnormal_loss
from experiments.phase_x0a_mandatory_gates import load_components
from experiments.phase_xa_smoke import write_csv
from experiments.phase_xlii_patch_specialized_adaptation import BATCH_SIZE, SEED, balanced_loss
from experiments.phase_xlv_cross_domain_transfer import CHECKPOINT
from experiments.phase_xxxviii_visual_state_representation import IMAGE_SIZE, LR
from utils.transforms import get_transform


OUT = Path("results/phase_cross_environment_state_consensus/formal")
VARIANTS = ("E0_Baseline", "E1_SimpleEnvironmentAveraging", "E2_AgreementFilteredConsensus", "E3_AnchoredConsensus")
PILOT_PAIRS = (("Full MVTec", "VisA"), ("Full VisA", "MPDD"))
STABILITY_PAIR = ("Full MVTec", "BTAD")
NUM_ENV = 4
PROPOSAL_STEPS = 20
ANCHOR_DEGREES = 5.0
AGREEMENT_THRESHOLD = 0.75


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


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_source_groups(source, model_info, device, smoke):
    if source == "Full VisA":
        return load_visa_groups(model_info, device, smoke)
    return load_mvtec_groups(smoke)


def subset_groups(groups, cats):
    return {c: groups[c] for c in cats if c in groups}


def split_environments(groups):
    cats = np.asarray(sorted(groups), dtype=object)
    rng = np.random.default_rng(SEED)
    shuffled = cats.copy()
    rng.shuffle(shuffled)
    envs = [[] for _ in range(NUM_ENV)]
    for i, cat in enumerate(shuffled.tolist()):
        envs[i % NUM_ENV].append(cat)
    return envs


def category_balanced_batch(groups, rng, total_batch):
    names = sorted(groups)
    per_cat = max(2, total_batch // max(1, len(names)))
    xs, ys = [], []
    for cat in names:
        data = groups[cat]
        per_cls = max(1, per_cat // 2)
        for cls in (0, 1):
            ids = np.flatnonzero(np.asarray(data["y"]) == cls)
            if ids.size == 0:
                continue
            choice = rng.choice(ids, size=per_cls, replace=ids.size < per_cls)
            xs.append(np.asarray(data["x"], dtype=np.float32)[choice])
            ys.append(np.asarray(data["y"], dtype=np.float32)[choice])
    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.float32)
    order = rng.permutation(len(y))
    return x[order], y[order]


def state_loss(z, y, s_n, s_a):
    logits = two_state_logits(z, s_n, s_a)
    task, bce, dice = balanced_loss(logits, y)
    ntr, hard_count = hardnormal_loss(logits, y)
    return task + ntr, task.detach(), ntr.detach(), bce.detach(), dice.detach(), hard_count


def mean_source_loss(groups, states, device, max_batches=None):
    s_n = torch.tensor(states[0], dtype=torch.float32, device=device)
    s_a = torch.tensor(states[1], dtype=torch.float32, device=device)
    vals = []
    with torch.no_grad():
        for cat in sorted(groups):
            x = np.asarray(groups[cat]["x"], dtype=np.float32)
            y = np.asarray(groups[cat]["y"], dtype=np.float32)
            n = len(y)
            for start in range(0, n, BATCH_SIZE):
                xb = torch.tensor(x[start:start + BATCH_SIZE], dtype=torch.float32, device=device)
                yb = torch.tensor(y[start:start + BATCH_SIZE], dtype=torch.float32, device=device)
                loss, task, ntr, bce, dice, hard = state_loss(xb, yb, s_n, s_a)
                vals.append((float(loss.cpu()), float(task.cpu()), float(ntr.cpu()), float(bce.cpu()), float(dice.cpu()), hard))
                if max_batches and len(vals) >= max_batches:
                    break
            if max_batches and len(vals) >= max_batches:
                break
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "loss": float(arr[:, 0].mean()),
        "state_loss": float(arr[:, 1].mean()),
        "ntr_loss": float(arr[:, 2].mean()),
        "bce": float(arr[:, 3].mean()),
        "dice": float(arr[:, 4].mean()),
        "hard_patch_count": float(arr[:, 5].mean()),
    }


def train_environment_proposal(source, env_id, env_groups, init_states, device, smoke):
    steps = 2 if smoke else PROPOSAL_STEPS
    s_n = torch.nn.Parameter(torch.tensor(init_states[0], dtype=torch.float32, device=device))
    s_a = torch.nn.Parameter(torch.tensor(init_states[1], dtype=torch.float32, device=device))
    opt = torch.optim.AdamW([s_n, s_a], lr=LR, weight_decay=0.0)
    rng = np.random.default_rng(SEED + env_id)
    before = mean_source_loss(env_groups, init_states, device, max_batches=2 if smoke else None)
    vals = []
    for step in range(1, steps + 1):
        xb, yb = category_balanced_batch(env_groups, rng, BATCH_SIZE)
        z = torch.tensor(xb, dtype=torch.float32, device=device)
        y = torch.tensor(yb, dtype=torch.float32, device=device)
        loss, task, ntr, bce, dice, hard = state_loss(z, y, s_n, s_a)
        opt.zero_grad()
        loss.backward()
        opt.step()
        vals.append((float(loss.detach().cpu()), float(task.cpu()), float(ntr.cpu()), float(bce.cpu()), float(dice.cpu()), hard))
    after_states = (s_n.detach().cpu().numpy().astype(np.float32), s_a.detach().cpu().numpy().astype(np.float32))
    after = mean_source_loss(env_groups, after_states, device, max_batches=2 if smoke else None)
    arr = np.asarray(vals, dtype=np.float64)
    row = {
        "source": source,
        "environment": env_id,
        "steps": steps,
        "train_loss_mean": float(arr[:, 0].mean()),
        "train_state_loss_mean": float(arr[:, 1].mean()),
        "train_ntr_loss_mean": float(arr[:, 2].mean()),
        "source_loss_before": before["loss"],
        "source_loss_after": after["loss"],
        "source_loss_delta": after["loss"] - before["loss"],
        "source_ntr_before": before["ntr_loss"],
        "source_ntr_after": after["ntr_loss"],
    }
    return after_states, row


def safe_cos(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def consensus_gate(source, init_states, proposals):
    d0 = contrast_direction_np(*init_states)
    ds, dns, das = [], [], []
    env_rows = []
    for env_id, states in proposals.items():
        sn, sa = states
        d = contrast_direction_np(sn, sa)
        dd = d - d0
        dn = sn - init_states[0]
        da = sa - init_states[1]
        ds.append(dd)
        dns.append(dn)
        das.append(da)
        cos0 = safe_cos(d, d0)
        env_rows.append({
            "source": source,
            "environment": env_id,
            "delta_d_norm": float(np.linalg.norm(dd)),
            "delta_sN_norm": float(np.linalg.norm(dn)),
            "delta_sA_norm": float(np.linalg.norm(da)),
            "state_rotation_angle_deg": float(np.degrees(np.arccos(np.clip(cos0, -1.0, 1.0)))) if np.isfinite(cos0) else float("nan"),
            "cos_d_env_E0": cos0,
        })
    pair_rows = []
    cos_d, cos_n, cos_a = [], [], []
    for i in range(len(ds)):
        for j in range(i + 1, len(ds)):
            cd = safe_cos(ds[i], ds[j])
            cn = safe_cos(dns[i], dns[j])
            ca = safe_cos(das[i], das[j])
            cos_d.append(cd)
            cos_n.append(cn)
            cos_a.append(ca)
            pair_rows.append({"source": source, "env_i": i, "env_j": j, "cos_delta_d": cd, "cos_delta_sN": cn, "cos_delta_sA": ca})
    finite = np.asarray([v for v in cos_d if np.isfinite(v)], dtype=np.float64)
    summary = {
        "source": source,
        "mean_pairwise_cos_delta_d": float(finite.mean()) if finite.size else float("nan"),
        "median_pairwise_cos_delta_d": float(np.median(finite)) if finite.size else float("nan"),
        "positive_pair_fraction": float(np.mean(finite > 0)) if finite.size else float("nan"),
        "mean_pairwise_cos_delta_sN": finite_mean(cos_n),
        "mean_pairwise_cos_delta_sA": finite_mean(cos_a),
        "mean_update_norm": finite_mean([r["delta_d_norm"] for r in env_rows]),
        "mean_state_rotation_angle": finite_mean([r["state_rotation_angle_deg"] for r in env_rows]),
    }
    return summary, env_rows, pair_rows


def normalize_state(x):
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def angle_between_d(states, init_states):
    d0 = contrast_direction_np(*init_states)
    d = contrast_direction_np(*states)
    return float(np.degrees(np.arccos(np.clip(np.dot(d0, d), -1.0, 1.0))))


def build_consensus_states(init_states, proposals):
    sn0, sa0 = init_states
    dns = np.stack([states[0] - sn0 for states in proposals.values()], axis=0)
    das = np.stack([states[1] - sa0 for states in proposals.values()], axis=0)
    dn_mean = dns.mean(axis=0)
    da_mean = das.mean(axis=0)
    e1 = (normalize_state(sn0 + dn_mean), normalize_state(sa0 + da_mean))

    def filter_update(arr):
        mean = arr.mean(axis=0)
        sign_mean = np.sign(mean)
        sign_arr = np.sign(arr)
        agree = np.mean(sign_arr == sign_mean[None, :], axis=0)
        keep = agree >= AGREEMENT_THRESHOLD
        return np.where(keep, mean, 0.0).astype(np.float32), keep

    dn_cons, keep_n = filter_update(dns)
    da_cons, keep_a = filter_update(das)
    e2 = (normalize_state(sn0 + dn_cons), normalize_state(sa0 + da_cons))
    scale = 1.0
    e3 = e2
    if angle_between_d(e2, init_states) > ANCHOR_DEGREES:
        lo, hi = 0.0, 1.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            cand = (normalize_state(sn0 + mid * dn_cons), normalize_state(sa0 + mid * da_cons))
            if angle_between_d(cand, init_states) <= ANCHOR_DEGREES:
                lo = mid
            else:
                hi = mid
        scale = lo
        e3 = (normalize_state(sn0 + scale * dn_cons), normalize_state(sa0 + scale * da_cons))
    return {
        "E1_SimpleEnvironmentAveraging": e1,
        "E2_AgreementFilteredConsensus": e2,
        "E3_AnchoredConsensus": e3,
    }, {
        "retained_sN_fraction": float(np.mean(keep_n)),
        "retained_sA_fraction": float(np.mean(keep_a)),
        "delta_sN_mean_norm": float(np.linalg.norm(dn_mean)),
        "delta_sA_mean_norm": float(np.linalg.norm(da_mean)),
        "delta_sN_cons_norm": float(np.linalg.norm(dn_cons)),
        "delta_sA_cons_norm": float(np.linalg.norm(da_cons)),
        "anchor_scale": float(scale),
    }


def state_geometry(states_by_source_variant, consensus_meta, groups_by_source, device, smoke):
    rows = []
    for source, variants in states_by_source_variant.items():
        init = variants["E0_Baseline"]
        d0 = contrast_direction_np(*init)
        base_loss = mean_source_loss(groups_by_source[source], init, device, max_batches=3 if smoke else None)
        for variant, states in variants.items():
            d = contrast_direction_np(*states)
            cos = float(np.dot(d0, d))
            cur_loss = mean_source_loss(groups_by_source[source], states, device, max_batches=3 if smoke else None)
            meta = consensus_meta.get(source, {})
            row = {
                "source": source,
                "variant": variant,
                "cos_d_E0": cos,
                "state_angle_deg": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))),
                "delta_sN_norm": float(np.linalg.norm(states[0] - init[0])),
                "delta_sA_norm": float(np.linalg.norm(states[1] - init[1])),
                "sN_norm": float(np.linalg.norm(states[0])),
                "sA_norm": float(np.linalg.norm(states[1])),
                "source_loss": cur_loss["loss"],
                "source_loss_delta_vs_E0": cur_loss["loss"] - base_loss["loss"],
                "source_ntr_loss": cur_loss["ntr_loss"],
                "inference_state_count": 2,
                "retained_sN_fraction": meta.get("retained_sN_fraction", float("nan")),
                "retained_sA_fraction": meta.get("retained_sA_fraction", float("nan")),
                "consensus_update_norm": float(np.linalg.norm(states[0] - init[0]) + np.linalg.norm(states[1] - init[1])),
                "anchor_scale": meta.get("anchor_scale", float("nan")),
            }
            rows.append(row)
    return rows


def delta_summary(raw, pairs):
    by = {(r["source"], r["dataset"], r["variant"]): r for r in raw}
    rows = []
    for variant in VARIANTS[1:]:
        vals = []
        for source, target in pairs:
            b = by.get((source, target, "E0_Baseline"))
            e = by.get((source, target, variant))
            if not b or not e:
                continue
            dap = fnum(e, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
            vals.append(dap)
            rows.append({
                "source": source,
                "dataset": target,
                "variant": variant,
                "Delta_M_FR_P_AUROC": fnum(e, "M_FR_P_AUROC") - fnum(b, "M_FR_P_AUROC"),
                "Delta_M_FR_P_AP": dap,
                "Delta_M_FR_P_F1max": fnum(e, "M_FR_P_F1max") - fnum(b, "M_FR_P_F1max"),
                "Delta_M_FR_PRO": fnum(e, "M_FR_PRO") - fnum(b, "M_FR_PRO"),
            })
        rows.append({
            "source": "SUMMARY",
            "dataset": "PILOT",
            "variant": variant,
            "mean_delta_M_FR_P_AP": finite_mean(vals),
            "improved": int(sum(v > 0 for v in vals if np.isfinite(v))),
            "total": int(sum(np.isfinite(v) for v in vals)),
        })
    return rows


def gate0_pass(rows):
    for r in rows:
        if fnum(r, "mean_pairwise_cos_delta_d") <= 0 or fnum(r, "positive_pair_fraction") < 0.5:
            return False
    return True


def decide(delta, geometry, stability_delta=None):
    summaries = [r for r in delta if r.get("source") == "SUMMARY"]
    if not summaries:
        return "CROSS_ENVIRONMENT_STATE_CONSENSUS_HAS_NO_COMPETITIVE_VALUE", {}
    best = max(summaries, key=lambda r: fnum(r, "mean_delta_M_FR_P_AP"))
    vals = [r for r in delta if r.get("variant") == best["variant"] and r.get("source") != "SUMMARY"]
    mean_delta = fnum(best, "mean_delta_M_FR_P_AP")
    improved = int(fnum(best, "improved", 0))
    per = [fnum(r, "Delta_M_FR_P_AP") for r in vals]
    focus = max(per) if per else float("nan")
    worst = min(per) if per else float("nan")
    by_geom = {(r["source"], r["variant"]): r for r in geometry}
    rotations = [fnum(r, "state_angle_deg") for r in geometry if r["variant"] == best["variant"]]
    max_rot = max(rotations) if rotations else float("nan")
    stats = {
        "best_variant": best["variant"],
        "mean_delta_P_AP": mean_delta,
        "improved": improved,
        "focus_best_delta_P_AP": focus,
        "worst_delta_P_AP": worst,
        "max_state_rotation_deg": max_rot,
    }
    pilot_gate = mean_delta >= 1.0 and focus >= 1.0 and worst >= -0.2 and max_rot < 20.0
    stats["pilot_gate_pass"] = pilot_gate
    if pilot_gate and stability_delta:
        stats["stability_delta_P_AP"] = stability_delta
        if stability_delta >= -0.2:
            return "CROSS_ENVIRONMENT_CONSENSUS_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    if pilot_gate:
        return "CROSS_ENVIRONMENT_CONSENSUS_SIGNIFICANTLY_IMPROVES_TRANSFER", stats
    e3 = next((r for r in summaries if r["variant"] == "E3_AnchoredConsensus"), None)
    if e3 and fnum(e3, "mean_delta_M_FR_P_AP") > 0:
        return "ANCHORING_PRESERVES_DIRECTION_BUT_GAIN_IS_WEAK", stats
    if mean_delta > 0 or improved > 0:
        return "CONSENSUS_EXISTS_BUT_DOES_NOT_IMPROVE_TRANSFER", stats
    return "CROSS_ENVIRONMENT_STATE_CONSENSUS_HAS_NO_COMPETITIVE_VALUE", stats


def write_report(outcome, split_rows, gate_rows, env_rows, pair_rows, raw, delta, geometry, proposal_rows, stats, runtime):
    lines = [
        "# Phase CESC: Cross-Environment State Consensus",
        "",
        f"OUTCOME: `{outcome}`",
        "",
        "Baseline is Canonical Two-State + NTR + fixed 2x2 Local Re-encoding Pyramid. Category labels are used only to build source pseudo-environments; inference remains one normal state plus one anomaly state.",
        "",
        f"Environments: `{NUM_ENV}`. Proposal steps per environment: `{PROPOSAL_STEPS}`. Agreement threshold: `{AGREEMENT_THRESHOLD}`. Anchor: `{ANCHOR_DEGREES}` degrees.",
        "",
        "## Environment Split",
        "",
        "| Source | env | categories |",
        "|---|---:|---|",
    ]
    for r in split_rows:
        lines.append(f"| {r['source']} | {r['environment']} | {r['categories']} |")
    lines += ["", "## Gate 0 Consensus Audit", "", "| Source | mean cos Δd | median cos Δd | positive pairs | mean update norm | mean rotation | cos ΔsN | cos ΔsA |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in gate_rows:
        lines.append(f"| {r['source']} | {fnum(r,'mean_pairwise_cos_delta_d'):.4f} | {fnum(r,'median_pairwise_cos_delta_d'):.4f} | {fnum(r,'positive_pair_fraction'):.4f} | {fnum(r,'mean_update_norm'):.4f} | {fnum(r,'mean_state_rotation_angle'):.3f} | {fnum(r,'mean_pairwise_cos_delta_sN'):.4f} | {fnum(r,'mean_pairwise_cos_delta_sA'):.4f} |")
    if raw:
        lines += ["", "## Pilot Metrics", "", "| Source | Target | Variant | P-AUROC | P-AP | P-F1max | PRO |", "|---|---|---|---:|---:|---:|---:|"]
        for r in raw:
            lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'M_FR_P_AUROC'):.4f} | {fnum(r,'M_FR_P_AP'):.4f} | {fnum(r,'M_FR_P_F1max'):.4f} | {fnum(r,'M_FR_PRO'):.4f} |")
        lines += ["", "## Delta vs E0", "", "| Source | Target | Variant | Delta P-AP | Delta PRO |", "|---|---|---|---:|---:|"]
        for r in delta:
            if r.get("source") != "SUMMARY":
                lines.append(f"| {r['source']} | {r['dataset']} | {r['variant']} | {fnum(r,'Delta_M_FR_P_AP'):+.4f} | {fnum(r,'Delta_M_FR_PRO'):+.4f} |")
        lines += ["", "## Summary", "", "| Variant | mean Delta P-AP | improved |", "|---|---:|---:|"]
        for r in delta:
            if r.get("source") == "SUMMARY":
                lines.append(f"| {r['variant']} | {fnum(r,'mean_delta_M_FR_P_AP'):+.4f} | {int(fnum(r,'improved',0))}/{int(fnum(r,'total',0))} |")
    if geometry:
        lines += ["", "## State Geometry", "", "| Source | Variant | cos(d,E0) | angle | ΔsN | ΔsA | retained N | retained A | source loss Δ |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in geometry:
            lines.append(f"| {r['source']} | {r['variant']} | {fnum(r,'cos_d_E0'):.4f} | {fnum(r,'state_angle_deg'):.3f} | {fnum(r,'delta_sN_norm'):.4f} | {fnum(r,'delta_sA_norm'):.4f} | {fnum(r,'retained_sN_fraction'):.4f} | {fnum(r,'retained_sA_fraction'):.4f} | {fnum(r,'source_loss_delta_vs_E0'):+.4f} |")
    lines += ["", f"Decision stats: `{json.dumps(stats, default=json_default)}`", "", f"Runtime: {runtime:.1f}s."]
    write_text(OUT / "REPORT.md", "\n".join(lines) + "\n")


def run(args):
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CESC] device={device}", flush=True)
    preprocess, _ = get_transform(argparse.Namespace(image_size=IMAGE_SIZE))
    clip_model, *_ = load_components(str(CHECKPOINT), device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    model_infos = {"Full VisA": load_visa_source(device), "Full MVTec": load_mvtec_source()}
    groups_by_source, states_by_source_variant = {}, {}
    split_rows, proposal_rows, gate_rows, env_rows, pair_rows = [], [], [], [], []
    consensus_meta = {}
    for source, info in model_infos.items():
        print(f"[CESC] load source groups {source}", flush=True)
        groups, _audit = load_source_groups(source, info, device, args.smoke)
        groups_by_source[source] = groups
        init = source_state_pair(info)
        envs = split_environments(groups)
        proposals = {}
        for env_id, cats in enumerate(envs):
            split_rows.append({"source": source, "environment": env_id, "categories": ";".join(cats)})
            states, row = train_environment_proposal(source, env_id, subset_groups(groups, cats), init, device, args.smoke)
            proposals[env_id] = states
            proposal_rows.append(row)
        summary, env, pairs = consensus_gate(source, init, proposals)
        gate_rows.append(summary)
        env_rows.extend(env)
        pair_rows.extend(pairs)
        states_by_source_variant[source] = {"E0_Baseline": init}
        cons, meta = build_consensus_states(init, proposals)
        states_by_source_variant[source].update(cons)
        consensus_meta[source] = meta
    write_csv(OUT / "ENVIRONMENT_SPLIT.csv", split_rows)
    write_csv(OUT / "ENVIRONMENT_PROPOSAL_TRAINING.csv", proposal_rows)
    write_csv(OUT / "CONSENSUS_GATE_SUMMARY.csv", gate_rows)
    write_csv(OUT / "ENVIRONMENT_UPDATE_STATS.csv", env_rows)
    write_csv(OUT / "ENVIRONMENT_PAIRWISE_COSINE.csv", pair_rows)
    if not gate0_pass(gate_rows):
        outcome = "NO_STABLE_CROSS_ENVIRONMENT_STATE_CONSENSUS"
        geometry = state_geometry(states_by_source_variant, consensus_meta, groups_by_source, device, args.smoke)
        runtime = time.time() - start
        stats = {"gate0_pass": False}
        write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
        write_json(OUT / "OUTCOME.json", {"outcome": outcome, "decision_stats": stats, "runtime_sec": runtime})
        write_report(outcome, split_rows, gate_rows, env_rows, pair_rows, [], [], geometry, proposal_rows, stats, runtime)
        print(f"[CESC] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)
        return
    raw, cats = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, PILOT_PAIRS, device, args.smoke, VARIANTS)
    delta = delta_summary(raw, PILOT_PAIRS)
    geometry = state_geometry(states_by_source_variant, consensus_meta, groups_by_source, device, args.smoke)
    outcome, stats = decide(delta, geometry)
    if stats.get("pilot_gate_pass"):
        best_variant = stats["best_variant"]
        raw_s, cats_s = evaluate_pairs(model_infos, states_by_source_variant, clip_model, preprocess, (STABILITY_PAIR,), device, args.smoke, ("E0_Baseline", best_variant))
        raw.extend(raw_s)
        cats.extend(cats_s)
        d_stab = delta_summary(raw_s, (STABILITY_PAIR,))
        stab = next((r for r in d_stab if r.get("source") != "SUMMARY" and r.get("variant") == best_variant), None)
        outcome, stats = decide(delta, geometry, fnum(stab, "Delta_M_FR_P_AP") if stab else float("nan"))
    runtime = time.time() - start
    write_csv(OUT / "RAW_METRICS.csv", raw)
    write_csv(OUT / "PER_CATEGORY_FULL_RES.csv", cats)
    write_csv(OUT / "PILOT_DELTA_SUMMARY.csv", delta)
    write_csv(OUT / "STATE_GEOMETRY.csv", geometry)
    write_json(OUT / "OUTCOME.json", {
        "outcome": outcome,
        "decision_stats": stats,
        "runtime_sec": runtime,
        "num_environments": NUM_ENV,
        "proposal_steps": PROPOSAL_STEPS,
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "anchor_degrees": ANCHOR_DEGREES,
        "strict_source_only": True,
        "target_training": False,
        "target_statistics": False,
        "target_tuning": False,
        "fixed_lrp": True,
        "inference_state_count": 2,
    })
    write_report(outcome, split_rows, gate_rows, env_rows, pair_rows, raw, delta, geometry, proposal_rows, stats, runtime)
    print(f"[CESC] OUTCOME={outcome} runtime={runtime:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
