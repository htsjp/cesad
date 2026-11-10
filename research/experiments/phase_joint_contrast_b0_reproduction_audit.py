import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import phase_joint_contrast_parameterization_causal_decomposition as test
from experiments import phase_spherical_state_trajectory_averaging as ref


OUT = Path("results/phase_joint_contrast_b0_reproduction_audit/formal")
STOP_REASON = "Stopped at CONFIG_DIFF: TEST B0 changes the feature/interface, trainable parameter set, scorer parameterization, and optimizer weight decay before the first batch."


def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = sorted({k for row in rows for k in row}) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def config_rows():
    rows = [
        {
            "item": "source_dataset",
            "reference": "Full VisA source via load_full_visa_source_lazy(False)",
            "test": "Full VisA source via load_full_visa_source_lazy(False)",
            "match": True,
            "severity": "ok",
        },
        {
            "item": "source_split_and_slots",
            "reference": "source_slots_by_category(dataset, indices, smoke)",
            "test": "source_slots_by_category(dataset, indices, smoke)",
            "match": True,
            "severity": "ok",
        },
        {
            "item": "patch_grid",
            "reference": f"{ref.SIDE}x{ref.SIDE}",
            "test": f"{test.SIDE}x{test.SIDE}",
            "match": ref.SIDE == test.SIDE,
            "severity": "ok" if ref.SIDE == test.SIDE else "fatal",
        },
        {
            "item": "pc1_protocol",
            "reference": "remove source-normal PC1, then apply frozen corrupted-mask alignment before B0 contrast training",
            "test": "remove source-normal PC1 only; train a rank4 alignment jointly inside TEST B0",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "source_feature_input_to_b0",
            "reference": "collect_source_patch_pool(..., align_model, q_pc1, device): fixed aligned patch features",
            "test": "collect_raw_source_patch_pool(..., q_pc1): raw PC1-removed normalized features",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "alignment_rank",
            "reference": "rank4 alignment already frozen/applied through make_model(corrupt states/payload)",
            "test": f"rank{test.RANK} alignment module is part of the trainable TEST B0 model",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "trainable_parameters",
            "reference": "one torch.nn.Parameter raw contrast vector only",
            "test": "rank4 adapter weights a/b plus two raw state vectors head",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "scorer_parameterization",
            "reference": "fixed midpoint/radius; logits=contrast_logits(z, m0, r0, normalize(raw_v))",
            "test": "adapt(x) @ (normalize(raw_sA) - normalize(raw_sN))",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "state_initialization",
            "reference": "raw contrast initialized from contrast_vectors.npz B0/init_v; m0 and r0 fixed",
            "test": "two trainable raw states initialized by hybrid_states(m_c, alpha*init_v)",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "optimizer",
            "reference": "torch.optim.AdamW([raw_contrast], lr=LR, weight_decay=0.0)",
            "test": "torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)",
            "match": False,
            "severity": "fatal",
        },
        {
            "item": "weight_decay",
            "reference": "0.0",
            "test": str(test.WEIGHT_DECAY),
            "match": test.WEIGHT_DECAY == 0.0,
            "severity": "fatal" if test.WEIGHT_DECAY != 0.0 else "ok",
        },
        {
            "item": "learning_rate",
            "reference": str(ref.LR),
            "test": str(test.LR),
            "match": ref.LR == test.LR,
            "severity": "ok" if ref.LR == test.LR else "fatal",
        },
        {
            "item": "epochs",
            "reference": str(ref.EPOCHS),
            "test": str(test.EPOCHS),
            "match": ref.EPOCHS == test.EPOCHS,
            "severity": "ok" if ref.EPOCHS == test.EPOCHS else "fatal",
        },
        {
            "item": "batch_size",
            "reference": str(ref.BATCH_SIZE),
            "test": str(test.BATCH_SIZE),
            "match": ref.BATCH_SIZE == test.BATCH_SIZE,
            "severity": "ok" if ref.BATCH_SIZE == test.BATCH_SIZE else "fatal",
        },
        {
            "item": "seed",
            "reference": str(ref.SEED),
            "test": str(test.SEED),
            "match": ref.SEED == test.SEED,
            "severity": "ok" if ref.SEED == test.SEED else "fatal",
        },
        {
            "item": "hardnormal_top_fraction",
            "reference": str(ref.TOP_NEG_FRAC),
            "test": str(test.TOP_NEG_FRAC),
            "match": ref.TOP_NEG_FRAC == test.TOP_NEG_FRAC,
            "severity": "ok" if ref.TOP_NEG_FRAC == test.TOP_NEG_FRAC else "fatal",
        },
        {
            "item": "loss_weights",
            "reference": "L_task + 1.0 * L_hard",
            "test": "L_task + 1.0 * L_hard",
            "match": True,
            "severity": "ok",
        },
        {
            "item": "scheduler_amp_precision_clipping",
            "reference": "no scheduler, no AMP, float32 tensors, no clipping",
            "test": "no scheduler, no AMP, float32 tensors, no clipping",
            "match": True,
            "severity": "ok",
        },
    ]
    return rows


def init_rows():
    return [
        {
            "component": "reference_trainable_raw_contrast",
            "reference": "shape=[1024], trainable=True",
            "test": "no equivalent single raw contrast parameter; TEST uses raw two-state head",
            "match": False,
            "status": "diverged_before_tensor_diff",
        },
        {
            "component": "rank4_alignment_a_b",
            "reference": "frozen/already applied to source features before B0 training",
            "test": "trainable parameters included in optimizer",
            "match": False,
            "status": "diverged_before_tensor_diff",
        },
        {
            "component": "optimizer_param_groups",
            "reference": "1 group: raw_contrast only; weight_decay=0.0",
            "test": "1 group: model.parameters() including adapter and head; weight_decay=1e-4",
            "match": False,
            "status": "diverged_before_tensor_diff",
        },
    ]


def not_run_rows(stage):
    return [{"stage": stage, "status": "not_run", "reason": STOP_REASON}]


def report_text(outcome):
    return f"""# B0 Exact Reproduction Divergence Audit

## Outcome

`{outcome}`

## First Divergence

The first divergence is at `CONFIG_DIFF`, before dataset-order, forward, hard mining, loss, backward, optimizer-step, or target evaluation checks.

Reference B0 is the verified path in `phase_spherical_state_trajectory_averaging`: it trains only one raw contrast vector on fixed, already-aligned source patch features. The optimizer is `AdamW([model], lr=LR, weight_decay=0.0)`.

TEST B0 in `phase_joint_contrast_parameterization_causal_decomposition` is not the same B0: it feeds raw PC1-removed features, trains a rank4 alignment module jointly, trains two raw states instead of one fixed-radius spherical contrast vector, and uses `AdamW(model.parameters(), weight_decay=1e-4)`.

Because the audit found a fatal config mismatch, later numerical stages were intentionally not run. This follows the requested first-divergence rule.

## Required Fix For Future Causal Runs

The causal-decomposition experiment must not rewrite B0. Its B0 should import/call the canonical reference training path from `phase_spherical_state_trajectory_averaging` or share the exact same source feature pool, fixed alignment, optimizer parameter group, zero weight decay, fixed midpoint/radius, and spherical contrast update. Only E1/E2/E3 should introduce the new parameterization being tested.

No target metrics were recomputed in this audit.
"""


def run(_args):
    OUT.mkdir(parents=True, exist_ok=True)
    outcome = "B0_DIVERGENCE_CONFIG"
    config = config_rows()

    write_csv(OUT / "CONFIG_DIFF.csv", config, ["item", "reference", "test", "match", "severity"])
    write_csv(OUT / "DATA_ORDER.csv", not_run_rows("DATA_ORDER"), ["stage", "status", "reason"])
    write_csv(OUT / "INITIALIZATION_DIFF.csv", init_rows(), ["component", "reference", "test", "match", "status"])
    write_csv(OUT / "FORWARD_DIFF.csv", not_run_rows("FORWARD"), ["stage", "status", "reason"])
    write_csv(OUT / "HARD_SET_DIFF.csv", not_run_rows("HARD_SET"), ["stage", "status", "reason"])
    write_csv(OUT / "LOSS_DIFF.csv", not_run_rows("LOSS"), ["stage", "status", "reason"])
    write_csv(OUT / "GRADIENT_DIFF.csv", not_run_rows("GRADIENT"), ["stage", "status", "reason"])
    write_csv(OUT / "OPTIMIZER_DIFF.csv", not_run_rows("OPTIMIZER"), ["stage", "status", "reason"])
    write_csv(OUT / "PROGRESSIVE_DIFF.csv", not_run_rows("PROGRESSIVE"), ["stage", "status", "reason"])

    write_json(
        OUT / "OUTCOME.json",
        {
            "outcome": outcome,
            "first_divergence_stage": "CONFIG_DIFF",
            "strict_source_only": True,
            "target_training": False,
            "target_statistics": False,
            "target_evaluation_run": False,
            "e1_e2_e3_run": False,
            "fatal_mismatches": [row["item"] for row in config if row["severity"] == "fatal"],
            "stop_reason": STOP_REASON,
        },
    )
    (OUT / "REPORT.md").write_text(report_text(outcome), encoding="utf-8")
    print(f"[B0Audit] outcome={outcome}")
    print(f"[B0Audit] wrote {OUT}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Accepted for consistency; this audit does not train.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
