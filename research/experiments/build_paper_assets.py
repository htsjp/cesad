import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "paper_assets"
LATEX = OUT / "latex"
FIGS = OUT / "figures"

TRANSFERS = [
    ("Full MVTec", "VisA", "M→V"),
    ("Full MVTec", "BTAD", "M→B"),
    ("Full MVTec", "MPDD", "M→P"),
    ("Full MVTec", "DTD-Synthetic", "M→D"),
    ("Full VisA", "MVTec", "V→M"),
    ("Full VisA", "BTAD", "V→B"),
    ("Full VisA", "MPDD", "V→P"),
    ("Full VisA", "DTD-Synthetic", "V→D"),
]
METRICS = ["P_AUROC", "P_AP", "P_F1max", "PRO"]


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fnum(row, key):
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def src(path):
    return str(Path(path).relative_to(ROOT)).replace("\\", "/")


def add_metric(master, trace, method, row, source_file, variant=None, same_protocol="yes", official_checkpoint="", reproduction_note=""):
    source, target = row.get("source"), row.get("dataset") or row.get("target")
    if not source or not target:
        return
    for met in METRICS + ["I_AUROC", "I_AP", "I_F1max"]:
        key = f"M_FR_{met}"
        val = fnum(row, key)
        if not math.isfinite(val):
            key2 = met.replace("_", "-")
            val = fnum(row, key2)
        if not math.isfinite(val):
            continue
        out = {
            "method": method,
            "source": source,
            "target": target,
            "transfer": transfer_label(source, target),
            "metric": met.replace("_", "-"),
            "value": val,
            "variant": variant or row.get("variant", ""),
            "source_file": src(source_file),
            "same_protocol": same_protocol,
            "official_checkpoint": official_checkpoint,
            "ours_reproduction": reproduction_note,
        }
        master.append(out)
        trace.append(dict(out))


def transfer_label(source, target):
    s = "M" if source == "Full MVTec" else "V" if source == "Full VisA" else source
    t = {"MVTec": "M", "VisA": "V", "BTAD": "B", "MPDD": "P", "DTD-Synthetic": "D"}.get(target, target)
    return f"{s}→{t}"


def collect_master():
    master, trace = [], []
    # Ours final method.
    p = RESULTS / "phase_paper_ready_final_validation" / "formal" / "INDUSTRIAL_MAIN_TABLE.csv"
    for r in read_csv(p):
        add_metric(master, trace, "Ours", r, p, "Canonical Two-State + NTR + Fixed LRP", "yes", "no", "ours final")
    # Ours ablation / multiseed full-resolution.
    p_ms = RESULTS / "phase_paper_ready_final_validation" / "formal" / "MULTISEED_SUMMARY.csv"
    for r in read_csv(p_ms):
        rr = {"source": r.get("source"), "dataset": r.get("dataset")}
        for met in METRICS:
            rr[f"M_FR_{met}"] = r.get(f"M_FR_{met}_mean", "")
        add_metric(master, trace, "Ablation:" + r.get("method", ""), rr, p_ms, r.get("method", ""), "yes", "no", "ours multiseed mean")
    # AnomalyCLIP.
    p = RESULTS / "phase_anomalyclip_same_protocol_audit" / "formal" / "RAW_METRICS.csv"
    for r in read_csv(p):
        if r.get("variant") == "AnomalyCLIP_OfficialCheckpoint":
            add_metric(master, trace, "AnomalyCLIP", r, p, r.get("variant"), "yes", "yes", "official checkpoint wrapper")
    # MRAD.
    p = RESULTS / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "MRAD_SAME_PROTOCOL.csv"
    name_map = {"M1_MRAD_TF": "MRAD-TF", "M2_MRAD_FT": "MRAD-FT", "M3_MRAD_CLIP": "MRAD-CLIP"}
    for r in read_csv(p):
        method = name_map.get(r.get("variant"))
        if method:
            add_metric(master, trace, method, r, p, r.get("variant"), "yes", "yes", "official MRAD code/checkpoint")
    # VisualAD.
    p = RESULTS / "phase_visualad_exact_protocol_audit" / "formal" / "RAW_METRICS.csv"
    for r in read_csv(p):
        if r.get("variant") == "E1c_FullVisualAD":
            add_metric(master, trace, "VisualAD", r, p, r.get("variant"), "yes", "official implementation", "exact protocol audit")
    # FB-CLIP.
    p = RESULTS / "phase_fbclip_pretrained_headroom_audit" / "formal" / "RAW_METRICS.csv"
    for r in read_csv(p):
        if r.get("variant") == "E1_Official_FBCLIP_Pretrained":
            add_metric(master, trace, "FB-CLIP", r, p, r.get("variant"), "yes", "yes", "official pretrained audit")
    # MoECLIP exact audit if available, but not strict main unless full method rows exist.
    p = RESULTS / "phase_moeclip_exact_protocol_audit" / "formal" / "RAW_METRICS.csv"
    for r in read_csv(p):
        if "MoE" in r.get("variant", "") and r.get("variant") != "E0_Core_NTR_FixedLRP":
            add_metric(master, trace, "MoECLIP", r, p, r.get("variant"), "yes", "official implementation", "exact protocol audit")
    return master, trace


def values_by_method(master, metric="P-AP"):
    out = defaultdict(dict)
    for r in master:
        if r["metric"] == metric:
            out[r["method"]][(r["source"], r["target"])] = float(r["value"])
    return out


def main_pap(master):
    preferred = ["AnomalyCLIP", "FAPrompt", "MRAD-TF", "MRAD-FT", "MRAD-CLIP", "VisualAD", "FB-CLIP", "Ours"]
    vals = values_by_method(master, "P-AP")
    rows = []
    for method in preferred:
        row = {"Method": method}
        present = []
        for source, target, label in TRANSFERS:
            v = vals.get(method, {}).get((source, target), float("nan"))
            row[label] = v if math.isfinite(v) else ""
            if math.isfinite(v):
                present.append(v)
        row["Mean"] = float(np.mean(present)) if present else ""
        row["Wins"] = 0
        rows.append(row)
    for source, target, label in TRANSFERS:
        col = [(i, r[label]) for i, r in enumerate(rows) if isinstance(r[label], float) and math.isfinite(r[label])]
        if not col:
            continue
        best = max(v for _i, v in col)
        for i, v in col:
            if abs(v - best) < 1e-9:
                rows[i]["Wins"] += 1
    return rows


def metrics_table(master):
    rows = []
    for method in sorted(values_by_method(master, "P-AP")):
        rr = {"Method": method}
        for met in ["P-AUROC", "P-AP", "P-F1max", "PRO"]:
            vals = [float(r["value"]) for r in master if r["method"] == method and r["metric"] == met and r["target"] in {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}]
            rr[f"Mean {met}"] = float(np.mean(vals)) if vals else ""
        for source_name, short in (("Full MVTec", "M-source"), ("Full VisA", "V-source")):
            vals = [float(r["value"]) for r in master if r["method"] == method and r["metric"] == "P-AP" and r["source"] == source_name and r["target"] in {"MVTec", "VisA", "BTAD", "MPDD", "DTD-Synthetic"}]
            rr[f"{short} Mean P-AP"] = float(np.mean(vals)) if vals else ""
        rows.append(rr)
    return rows


def ablation_table(master):
    methods = ["Ablation:StateAligned", "Ablation:StateAligned+NTR", "Ablation:StateAligned+NTR+LRP"]
    labels = {"Ablation:StateAligned": "Base State", "Ablation:StateAligned+NTR": "+ NTR", "Ablation:StateAligned+NTR+LRP": "+ Fixed LRP"}
    rows = []
    base_ap = None
    for m in methods:
        rr = {"Variant": labels[m], "Protocol": "full-res M_FR, multiseed mean where available"}
        for met in ["P-AUROC", "P-AP", "P-F1max", "PRO"]:
            vals = [float(r["value"]) for r in master if r["method"] == m and r["metric"] == met and r["target"] in {"MVTec", "VisA", "BTAD", "MPDD"}]
            rr[f"Mean {met}"] = float(np.mean(vals)) if vals else ""
        if rr["Variant"] == "Base State":
            base_ap = rr["Mean P-AP"] if isinstance(rr["Mean P-AP"], float) else None
        rr["Delta P-AP vs Base"] = rr["Mean P-AP"] - base_ap if base_ap is not None and isinstance(rr["Mean P-AP"], float) else ""
        rows.append(rr)
    return rows


def design_table():
    return [
        {"Method": "AnomalyCLIP", "Visual Backbone": "CLIP ViT-L/14@336", "Text": "yes", "Memory": "no", "Prompt Learning": "object-agnostic learnable text prompts", "Visual Adaptation": "DPAM visual path", "Multi-layer": "official feature layer setting", "Target Reference": "no", "Main Representation": "normal/abnormal prompt similarity"},
        {"Method": "AdaCLIP", "Visual Backbone": "CLIP", "Text": "yes", "Memory": "no", "Prompt Learning": "hybrid static/dynamic prompts", "Visual Adaptation": "prompt-driven adaptation", "Multi-layer": "not used in our strict table", "Target Reference": "protocol-dependent", "Main Representation": "learnable prompts"},
        {"Method": "AA-CLIP", "Visual Backbone": "CLIP", "Text": "yes", "Memory": "no", "Prompt Learning": "anomaly-aware text anchors", "Visual Adaptation": "residual adapters", "Multi-layer": "not reproduced", "Target Reference": "not used here", "Main Representation": "text anchors + aligned patch features"},
        {"Method": "FAPrompt", "Visual Backbone": "CLIP ViT-L/14@336", "Text": "yes", "Memory": "no", "Prompt Learning": "CAP + DAP", "Visual Adaptation": "no explicit adapter", "Multi-layer": "official AnomalyCLIP-like path", "Target Reference": "query/image-dependent prior", "Main Representation": "fine-grained abnormality prompts"},
        {"Method": "MRAD", "Visual Backbone": "CLIP", "Text": "optional in MRAD-CLIP", "Memory": "yes", "Prompt Learning": "MRAD-CLIP uses prompt biases", "Visual Adaptation": "MRAD-FT metric layers", "Multi-layer": "no", "Target Reference": "no target reference in our audit", "Main Representation": "source feature-label memory retrieval"},
        {"Method": "VisualAD", "Visual Backbone": "frozen ViT/CLIP visual", "Text": "no", "Memory": "no", "Prompt Learning": "no text prompt", "Visual Adaptation": "learnable normal/anomaly tokens + SAF", "Multi-layer": "yes", "Target Reference": "no", "Main Representation": "interactive visual normal/anomaly tokens"},
        {"Method": "FB-CLIP", "Visual Backbone": "CLIP ViT-L/14@336", "Text": "yes", "Memory": "no", "Prompt Learning": "multi-strategy textual representation", "Visual Adaptation": "foreground/background separation path", "Multi-layer": "yes", "Target Reference": "no in pretrained audit", "Main Representation": "foreground-background disentangled CLIP scoring"},
        {"Method": "Ours", "Visual Backbone": "frozen CLIP", "Text": "no at inference", "Memory": "no", "Prompt Learning": "no", "Visual Adaptation": "no", "Multi-layer": "no", "Target Reference": "no", "Main Representation": "compact canonical two-state + NTR + fixed LRP"},
    ]


def protocol_table():
    return [
        {"Method": "AnomalyCLIP", "Training Source": "MVTec/VisA official checkpoints", "Target Used in Training?": "No", "Target Val?": "No", "Target Normal Ref?": "No", "Target Class Name?": "No", "Official Checkpoint?": "Yes", "Strict Comparable?": "Yes"},
        {"Method": "FAPrompt", "Training Source": "official repo supports training; local checkpoint missing", "Target Used in Training?": "Unclear for pretrained path", "Target Val?": "Unverified", "Target Normal Ref?": "No explicit normal ref", "Target Class Name?": "No object category in our protocol", "Official Checkpoint?": "No local checkpoint", "Strict Comparable?": "No"},
        {"Method": "AdaCLIP", "Training Source": "official/pretrained protocol includes mixed auxiliary data in paper", "Target Used in Training?": "Protocol-dependent", "Target Val?": "Potential validation selection per user audit note", "Target Normal Ref?": "No", "Target Class Name?": "May use prompt/category settings", "Official Checkpoint?": "Not used", "Strict Comparable?": "No"},
        {"Method": "AA-CLIP", "Training Source": "not locally reproduced", "Target Used in Training?": "Unverified", "Target Val?": "Unverified", "Target Normal Ref?": "No", "Target Class Name?": "text anchors", "Official Checkpoint?": "No local checkpoint", "Strict Comparable?": "No"},
        {"Method": "MRAD", "Training Source": "Full MVTec / Full VisA in our audit", "Target Used in Training?": "No", "Target Val?": "No", "Target Normal Ref?": "No", "Target Class Name?": "No", "Official Checkpoint?": "Yes/local official assets", "Strict Comparable?": "Yes"},
        {"Method": "VisualAD", "Training Source": "Full MVTec / Full VisA in our audit", "Target Used in Training?": "No", "Target Val?": "No", "Target Normal Ref?": "No", "Target Class Name?": "No", "Official Checkpoint?": "Official implementation", "Strict Comparable?": "Yes pilot"},
        {"Method": "FB-CLIP", "Training Source": "official pretrained MVTec/VisA checkpoints", "Target Used in Training?": "No in audit", "Target Val?": "No", "Target Normal Ref?": "No", "Target Class Name?": "No tuning", "Official Checkpoint?": "Yes", "Strict Comparable?": "Yes pilot"},
        {"Method": "Ours", "Training Source": "Full MVTec / Full VisA", "Target Used in Training?": "No", "Target Val?": "No", "Target Normal Ref?": "No", "Target Class Name?": "No", "Official Checkpoint?": "Own reproducible checkpoint", "Strict Comparable?": "Yes"},
    ]


def visa_tradeoff(master):
    pap = values_by_method(master, "P-AP")
    core = pap.get("Ours", {})
    rows = []
    methods = ["MRAD-CLIP", "VisualAD", "FB-CLIP", "Ours"]
    for m in methods:
        vals = pap.get(m, {})
        mv = vals.get(("Full MVTec", "VisA"), float("nan"))
        btad_delta = vals.get(("Full MVTec", "BTAD"), float("nan")) - core.get(("Full MVTec", "BTAD"), float("nan"))
        mpdd_delta = vals.get(("Full VisA", "MPDD"), float("nan")) - core.get(("Full VisA", "MPDD"), float("nan"))
        rows.append({"Method": m, "MVTec→VisA P-AP": mv, "Delta vs Core": mv - core.get(("Full MVTec", "VisA"), float("nan")) if math.isfinite(mv) else "", "MVTec→BTAD Delta": btad_delta if math.isfinite(btad_delta) else "", "VisA→MPDD Delta": mpdd_delta if math.isfinite(mpdd_delta) else ""})
    return rows


def visa_analysis():
    rows = []
    ours_p = RESULTS / "phase_mva_mrad_same_protocol_visa_factorization" / "formal" / "VISA_CATEGORY_AUDIT.csv"
    mva = read_csv(ours_p)
    vis_p = RESULTS / "phase_visualad_exact_protocol_audit" / "formal" / "PER_CATEGORY_STANDARD_METRICS.csv"
    fb_p = RESULTS / "phase_fbclip_pretrained_headroom_audit" / "formal" / "PER_CATEGORY_STANDARD_METRICS.csv"
    def lookup(rows_in, variant_match, cat):
        vals = [fnum(r, "M_FR_P_AP") for r in rows_in if r.get("source") == "Full MVTec" and r.get("dataset") == "VisA" and r.get("category") == cat and variant_match(r)]
        return vals[0] if vals else float("nan")
    vis_rows = read_csv(vis_p)
    fb_rows = read_csv(fb_p)
    cats = ["macaroni1","macaroni2","pcb1","pcb2","pcb3","pcb4","fryum","pipe_fryum","capsules","cashew","chewinggum","candle"]
    for cat in cats:
        mrows = [r for r in mva if r.get("category") == cat]
        ours = fnum(mrows[0], "E0_P_AP") if mrows else float("nan")
        mrad = max([fnum(r, k) for r in mrows for k in ("M1_P_AP","M2_P_AP","M3_P_AP") if k in r], default=float("nan")) if mrows else float("nan")
        visual = lookup(vis_rows, lambda r: r.get("variant") == "E1c_FullVisualAD", cat)
        fb = lookup(fb_rows, lambda r: r.get("variant") == "E1_Official_FBCLIP_Pretrained", cat)
        gains = [x - ours for x in (mrad, visual, fb) if math.isfinite(x) and math.isfinite(ours)]
        rows.append({"Category": cat, "Ours": ours, "MRAD best": mrad, "VisualAD": visual, "FB-CLIP": fb, "Best Specialist Gain": max(gains) if gains else ""})
    return rows


def negative_table():
    specs = [
        ("Experience Enrichment", "CMAB / DTS / SCEE / SSEC", ["phase_cmab_compact_multi_anomaly_basis", "phase_dts_a_symmetric_distributional_two_state", "phase_dts_b_asymmetric_normal_variation_geometry", "phase_scee_lite", "phase_ssec_state_space_experience_consolidation"], "often diagnostic, no stable transfer gain"),
        ("Feature Adaptation", "LDRO / GCRA / SAF-LSA / ECAP", ["phase_learned_defect_relation_operator", "phase_gcra_geometry_constrained_residual_adaptation", "phase_layerwise_state_alignment", "phase_ecap_experience_conditioned_attention_prior"], "adaptation tends to overfit or disrupt transfer geometry"),
        ("Auxiliary Representation", "DINO/SPSD / GSCR / LDSB / FOVA", ["phase_spsd_state_preserving_structural_diffusion", "phase_gscr_graph_spectral_contrast_residual", "phase_ldsb_local_differential_state_branch", "phase_fova_field_of_view_amplification_audit"], "residual signals mostly not useful as stable fusion"),
        ("Semantic/Text", "Text complementarity / SDP", ["phase_text_state_complementarity_audit", "phase_sdp_state_guided_dynamic_prompt_feedback"], "text signals did not stably beat frozen visual core"),
        ("Specialist/Structure", "VisualAD/FB/CSR/ISRO/SNTS", ["phase_visualad_exact_protocol_audit", "phase_fbclip_pretrained_headroom_audit", "phase_conservative_specialist_rescue", "phase_isro_instance_structure_reobservation", "phase_snts_structure_normalized_two_state"], "VisA gains trade off with BTAD/MPDD or fail to improve core"),
    ]
    rows = []
    for family, rep, dirs, failure in specs:
        best_delta = float("-inf")
        pos = 0
        total = 0
        outcome = []
        for d in dirs:
            op = RESULTS / d / "formal" / "OUTCOME.json"
            if op.exists():
                try:
                    data = json.loads(op.read_text(encoding="utf-8"))
                    outcome.append(data.get("outcome", ""))
                    for k, v in data.items():
                        if "mean" in k.lower() and "delta" in k.lower() and isinstance(v, (int, float)):
                            best_delta = max(best_delta, float(v))
                except Exception:
                    pass
            raw = RESULTS / d / "formal" / "RAW_METRICS.csv"
            rows_raw = read_csv(raw)
            if rows_raw:
                variants = sorted({r.get("variant") for r in rows_raw})
                bases = [v for v in variants if v and ("E0" in v or "B0" in v)]
                if bases:
                    base = bases[0]
                    by = {(r.get("source"), r.get("dataset"), r.get("variant")): r for r in rows_raw}
                    for s, t in sorted({(r.get("source"), r.get("dataset")) for r in rows_raw}):
                        b = by.get((s, t, base))
                        if not b:
                            continue
                        for v in variants:
                            if v == base:
                                continue
                            c = by.get((s, t, v))
                            if c:
                                delta = fnum(c, "M_FR_P_AP") - fnum(b, "M_FR_P_AP")
                                if math.isfinite(delta):
                                    best_delta = max(best_delta, delta)
                                    pos += int(delta > 0)
                                    total += 1
        rows.append({"Family": family, "Representative Variant": rep, "Best Mean/Cell Delta P-AP": best_delta if best_delta != float("-inf") else "", "Positive Transfers": f"{pos}/{total}" if total else "", "Main Failure": failure, "Representative Outcomes": "; ".join([o for o in outcome if o])})
    return rows


def fmt(x):
    if x == "" or x is None:
        return "—"
    try:
        return f"{float(x):.2f}"
    except Exception:
        return str(x)


def markdown_table(rows):
    if not rows:
        return ""
    keys = list(rows[0].keys())
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |")
    return "\n".join(lines)


def latex_table(rows, path, caption):
    if not rows:
        return
    keys = list(rows[0].keys())
    numeric_by_col = defaultdict(list)
    for k in keys[1:]:
        vals = []
        for r in rows:
            v = r.get(k, "")
            if isinstance(v, float) and math.isfinite(v):
                vals.append(v)
        if vals:
            order = sorted(set(vals), reverse=True)
            numeric_by_col[k] = order[:2]
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{l" + "r" * (len(keys) - 1) + "}", "\\toprule", " & ".join(keys) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for k in keys:
            v = r.get(k, "")
            if isinstance(v, float) and math.isfinite(v):
                s = f"{v:.2f}"
                top = numeric_by_col.get(k, [])
                if top and abs(v - top[0]) < 1e-9:
                    s = "\\textbf{" + s + "}"
                elif len(top) > 1 and abs(v - top[1]) < 1e-9:
                    s = "\\underline{" + s + "}"
                cells.append(s)
            else:
                cells.append(str(v) if v != "" else "—")
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def figures(main_rows, trade_rows, ablation_rows):
    FIGS.mkdir(parents=True, exist_ok=True)
    methods = [m for m in ["MRAD-CLIP", "VisualAD", "FB-CLIP", "AnomalyCLIP", "Ours"] if any(r["Method"] == m for r in main_rows)]
    cols = [x[2] for x in TRANSFERS]
    mat = []
    for m in methods:
        r = next(x for x in main_rows if x["Method"] == m)
        mat.append([float(r[c]) if isinstance(r[c], float) else np.nan for c in cols])
    plt.figure(figsize=(10, 3.8))
    im = plt.imshow(np.asarray(mat, dtype=float), aspect="auto", cmap="viridis")
    plt.colorbar(im, label="P-AP")
    plt.xticks(range(len(cols)), cols)
    plt.yticks(range(len(methods)), methods)
    plt.title("Strict cross-dataset P-AP")
    plt.tight_layout()
    plt.savefig(FIGS / "figure_a_pap_heatmap.png", dpi=200)
    plt.savefig(FIGS / "figure_a_pap_heatmap.pdf")
    plt.close()

    plt.figure(figsize=(5, 4))
    for r in trade_rows:
        x = r.get("Delta vs Core", "")
        ys = [v for v in (r.get("MVTec→BTAD Delta", ""), r.get("VisA→MPDD Delta", "")) if isinstance(v, float) and math.isfinite(v)]
        if isinstance(x, float) and ys:
            plt.scatter([x], [float(np.mean(ys))])
            plt.text(x, float(np.mean(ys)), " " + r["Method"])
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel("MVTec→VisA ΔP-AP")
    plt.ylabel("Mean ΔP-AP on BTAD/MPDD")
    plt.tight_layout()
    plt.savefig(FIGS / "figure_b_specialization_stability.png", dpi=200)
    plt.savefig(FIGS / "figure_b_specialization_stability.pdf")
    plt.close()

    labels = [r["Variant"] for r in ablation_rows if isinstance(r.get("Mean P-AP"), float)]
    vals = [r["Mean P-AP"] for r in ablation_rows if isinstance(r.get("Mean P-AP"), float)]
    plt.figure(figsize=(5, 3.5))
    plt.bar(labels, vals, color=["#777777", "#4c78a8", "#59a14f"][: len(vals)])
    plt.ylabel("Mean P-AP")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGS / "figure_c_ablation_bar.png", dpi=200)
    plt.savefig(FIGS / "figure_c_ablation_bar.pdf")
    plt.close()


def source_trace_md(trace):
    lines = ["# Source Trace", "", "Every numeric result below is copied from a formal result CSV; no table cell is hand-filled.", "", "| Method | Transfer | Metric | Value | Source File | Same Protocol | Official Checkpoint | Note |", "|---|---|---|---:|---|---|---|---|"]
    for r in trace:
        if r["metric"] in {"P-AP", "P-AUROC", "P-F1max", "PRO"}:
            lines.append(f"| {r['method']} | {r['transfer']} | {r['metric']} | {float(r['value']):.6f} | `{r['source_file']}` | {r['same_protocol']} | {r['official_checkpoint']} | {r['ours_reproduction']} |")
    return "\n".join(lines) + "\n"


def bibliography_md():
    return """# Bibliography And Method Sources

This file records method metadata used for the paper tables. Protocol notes indicate whether local strict same-protocol numbers are available.

## AnomalyCLIP
- Paper: AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection.
- Authors: Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen.
- Venue/Year: ICLR 2024.
- Official repository: https://github.com/zqhang/AnomalyCLIP
- Official checkpoint availability: local official-style checkpoints found and evaluated.
- Protocol note: strict same-protocol wrapper generated in `results/phase_anomalyclip_same_protocol_audit/formal/`.
- BibTeX:
```bibtex
@inproceedings{zhou2024anomalyclip,
  title={AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection},
  author={Zhou, Qihang and Pang, Guansong and Tian, Yu and He, Shibo and Chen, Jiming},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```

## AdaCLIP
- Paper: AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection.
- Authors: Yunkang Cao, Jiangning Zhang, Luca Frittoli, Yuqi Cheng, Weiming Shen, Giacomo Boracchi.
- Venue/Year: ECCV 2024.
- Official repository: https://github.com/caoyunkang/AdaCLIP
- Official checkpoint availability: not used in strict table.
- Protocol note: not strictly comparable under this source-only protocol.

## AA-CLIP
- Paper: AA-CLIP: Enhancing Zero-Shot Anomaly Detection via Anomaly-Aware CLIP.
- Authors: Wenxin Ma et al.
- Venue/Year: CVPR 2025.
- Official repository: https://github.com/Mwxinnn/AA-CLIP
- Official checkpoint availability: no local official pretrained checkpoint used.
- Protocol note: related-work/reference only.

## FAPrompt
- Paper: Fine-grained Abnormality Prompt Learning for Zero-shot Anomaly Detection.
- Authors: Jiawen Zhu, Yew-Soon Ong, Chunhua Shen, Guansong Pang.
- Venue/Year: ICCV 2025.
- Official repository: https://github.com/mala-lab/FAPrompt
- Official checkpoint availability: no local official checkpoint found.
- Protocol note: `PROTOCOL_MISMATCH_NOT_INCLUDED_IN_STRICT_MAIN_TABLE` / unavailable checkpoint, related-work table only.

## MRAD
- Paper: MRAD: Zero-Shot Anomaly Detection with Memory-Driven Retrieval.
- Authors: Chaoran Xu, Chengkan Lv, Qiyu Chen, Feng Zhang, Zhengtao Zhang.
- Venue/Year: 2026 preprint / ICLR listing found.
- Official repository: https://github.com/CROVO1026/MRAD
- Official checkpoint availability: local assets evaluated.
- Protocol note: strict same-protocol audit available in `phase_mva_mrad_same_protocol_visa_factorization`.

## VisualAD
- Paper: VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer.
- Authors: Yanning Hou, Peiyuan Li, Zirui Liu, Yitong Wang, Yanran Ruan, Jianfeng Qiu, Ke Xu.
- Venue/Year: CVPR 2026.
- Official repository: https://github.com/7HHHHH/VisualAD
- Official checkpoint availability: official implementation audit run locally.
- Protocol note: strict pilot results included where available.

## FB-CLIP
- Paper: FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement.
- Authors: Ming Hu et al.
- Venue/Year: CVPR 2026.
- Official repository: https://github.com/Xi-Mu-Yu/FB-CLIP
- Official checkpoint availability: official pretrained checkpoints evaluated locally.
- Protocol note: strict pretrained audit included where available.

## MoECLIP
- Paper: MoECLIP: Patch-Specialized Experts for Zero-shot Anomaly Detection.
- Authors: Jun Yeong Park, JunYoung Seo, Minji Kang, Yu Rang Park.
- Venue/Year: CVPR 2026.
- Official repository: https://github.com/CoCoRessa/MoECLIP
- Official checkpoint availability: official code audited locally; main table only includes strict rows if present.
- Protocol note: exact-protocol audit artifact exists, not automatically mixed with nonmatching paper numbers.

## WinCLIP
- Paper: Zero-/Few-Shot Anomaly Classification and Segmentation.
- Authors: Jongheon Jeong, Yang Zou, Taewan Kim, Dongqing Zhang, Avinash Ravichandran, Onkar Dabeer.
- Venue/Year: CVPR 2023.
- Official repository: no author official implementation used here; public repos are treated as reproduction/reference.
- Protocol note: literature/reference only.

## APRIL-GAN
- Paper: APRIL-GAN: A Zero-/Few-Shot Anomaly Classification and Segmentation Method for CVPR 2023 VAND Workshop Challenge Tracks 1&2.
- Authors: Xuhai Chen, Yue Han, Jiangning Zhang.
- Venue/Year: CVPR 2023 VAND Workshop technical report.
- Official repository/checkpoint: not used.
- Protocol note: literature/reference only.
"""


def report_md(main_rows, metrics_rows, ablation_rows, fairness_rows, negative_rows):
    ours = next(r for r in main_rows if r["Method"] == "Ours")
    method_means = [(r["Method"], r["Mean"]) for r in main_rows if isinstance(r["Mean"], float)]
    method_means.sort(key=lambda x: x[1], reverse=True)
    ours_rank = [m for m, _ in method_means].index("Ours") + 1 if any(m == "Ours" for m, _ in method_means) else ""
    wins = int(ours.get("Wins", 0))
    shortfalls = []
    for c in [x[2] for x in TRANSFERS]:
        v = ours.get(c, "")
        if isinstance(v, float):
            shortfalls.append((c, v))
    shortfalls.sort(key=lambda x: x[1])
    return f"""# Paper Assets Report

OUTCOME: `PAPER_ASSETS_READY_WITH_TRACE`

Final frozen method: `Canonical Two-State + NTR + Fixed LRP`.

## Main Result

- Ours mean P-AP over strict industrial transfers available in Table 1: `{ours['Mean']:.4f}`.
- Ours rank by mean P-AP among populated strict rows: `{ours_rank}/{len(method_means)}`.
- Ours wins: `{wins}` transfer cells.
- Main shortfall: `{shortfalls[0][0]}` with P-AP `{shortfalls[0][1]:.4f}`.

## Table 1: Strict Cross-Dataset P-AP

{markdown_table(main_rows)}

## Table 2: Mean Localization Metrics

{markdown_table(metrics_rows)}

## Table 3: Core Ablation

{markdown_table(ablation_rows)}

## Protocol Fairness

{markdown_table(fairness_rows)}

## Negative Experiment Compression

{markdown_table(negative_rows)}

## Notes

- `FAPrompt`, `AdaCLIP`, `AA-CLIP`, `WinCLIP`, and `APRIL-GAN` are not filled into strict main cells unless local official same-protocol outputs exist.
- Patch-grid metrics are excluded from the main P-AP table when full-resolution values exist.
- Each numeric value is traceable in `SOURCE_TRACE.md` and `MASTER_RESULTS.csv`.
"""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    LATEX.mkdir(parents=True, exist_ok=True)
    master, trace = collect_master()
    main_rows = main_pap(master)
    metrics_rows = metrics_table(master)
    ablation_rows = ablation_table(master)
    design_rows = design_table()
    fairness_rows = protocol_table()
    trade_rows = visa_tradeoff(master)
    visa_rows = visa_analysis()
    neg_rows = negative_table()
    eff_path = RESULTS / "phase_paper_ready_final_validation" / "formal" / "EFFICIENCY.csv"
    eff_rows = read_csv(eff_path)

    write_csv(OUT / "MASTER_RESULTS.csv", master)
    write_csv(OUT / "TABLE_MAIN_PAP.csv", main_rows)
    write_csv(OUT / "TABLE_METRICS.csv", metrics_rows)
    write_csv(OUT / "TABLE_ABLATION.csv", ablation_rows)
    write_csv(OUT / "TABLE_METHOD_DESIGN.csv", design_rows)
    write_csv(OUT / "TABLE_PROTOCOL_FAIRNESS.csv", fairness_rows)
    write_csv(OUT / "TABLE_VISA_ANALYSIS.csv", visa_rows)
    write_csv(OUT / "TABLE_NEGATIVE_EXPERIMENTS.csv", neg_rows)
    write_csv(OUT / "TABLE_EFFICIENCY.csv", eff_rows)
    write_csv(OUT / "TABLE_SPECIALIZATION_TRADEOFF.csv", trade_rows)
    write_csv(OUT / "SOURCE_TRACE.csv", trace)
    (OUT / "SOURCE_TRACE.md").write_text(source_trace_md(trace), encoding="utf-8")
    (OUT / "BIBLIOGRAPHY_METHODS.md").write_text(bibliography_md(), encoding="utf-8")
    (OUT / "REPORT.md").write_text(report_md(main_rows, metrics_rows, ablation_rows, fairness_rows, neg_rows), encoding="utf-8")
    (OUT / "TABLES.md").write_text("\n\n".join([
        "# Paper Tables",
        "## Table 1\n" + markdown_table(main_rows),
        "## Table 2\n" + markdown_table(metrics_rows),
        "## Table 3\n" + markdown_table(ablation_rows),
        "## Method Design\n" + markdown_table(design_rows),
        "## Protocol Fairness\n" + markdown_table(fairness_rows),
        "## VisA Analysis\n" + markdown_table(visa_rows),
        "## Negative Experiments\n" + markdown_table(neg_rows),
    ]) + "\n", encoding="utf-8")
    latex_table(main_rows, LATEX / "table_main_pap.tex", "Strict cross-dataset pixel AP.")
    latex_table(ablation_rows, LATEX / "table_ablation.tex", "Core ablation under the official-compatible full-resolution evaluator.")
    latex_table(design_rows, LATEX / "table_method_design.tex", "Method design comparison.")
    latex_table(fairness_rows, LATEX / "table_protocol.tex", "Protocol fairness audit.")
    figures(main_rows, trade_rows, ablation_rows)
    write_json(OUT / "OUTCOME.json", {"outcome": "PAPER_ASSETS_READY_WITH_TRACE", "num_master_values": len(master), "num_trace_values": len(trace)})
    print(f"[paper-assets] wrote {OUT}")


if __name__ == "__main__":
    main()
