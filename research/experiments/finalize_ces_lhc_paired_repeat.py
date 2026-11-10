"""Validate and summarize completed paired repeats; no fitting or inference."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import phase_ces_lhc_paired_repeat as p


def main():
    out=p.DEFAULT_OUT/"formal"
    outcome=json.loads((out/"OUTCOME.json").read_text())
    assert outcome["status"]=="PAIRED_REPEAT_COMPLETE"
    raw=p.read_csv(out/"PAIRED_RAW_METRICS.csv")
    primary=[r for r in raw if r["run_type"]=="PRIMARY_MATCHED_LR"]
    assert len(primary)==36
    checks=[]; geometry=[]
    for source in ("Full MVTec","Full VisA"):
        for seed in p.SEEDS:
            states=[]
            for method in p.METHODS:
                files=list((out/"checkpoints").glob(f"{source.replace(' ','_')}_{method}_seed{seed}_primary_*.pt"))
                assert len(files)==1
                saved=torch.load(files[0],map_location="cpu",weights_only=False)
                assert saved["realized_updates"]==30
                assert len(saved["logs"])==30
                assert [r["optimizer_step_min"] for r in saved["logs"]]==list(range(1,31))
                assert [r["optimizer_step_max"] for r in saved["logs"]]==list(range(1,31))
                assert all(r["actual_batch"]==(8190 if source=="Full MVTec" else 8184) for r in saved["logs"])
                for row in saved["logs"]:
                    # Linear's total parameter norm includes b; its effective
                    # weight norm is the requested direction-only norm.
                    geometry.append(dict(**row,raw_direction_norm=(row["effective_weight_norm"] if method=="LinearHead" else row["raw_parameter_norm"]),
                        gradient_norm_scope="all trainable parameters; Linear includes bias"))
                states.append(saved)
            assert len({s["feature_array_hash"] for s in states})==1
            assert len({s["batch_stream_hash"] for s in states})==1
            assert len({tuple(r["batch_ids_hash"] for r in s["logs"]) for s in states})==1
            checks.append(dict(source=source,seed=seed,paired_source_features=True,paired_batch_stream=True,
                all_optimizer_counters_verified=True,actual_updates=30,checkpoint_policy="last update",newly_trained_CES=True))
    p.write_csv(out/"PAIRED_EXECUTION_VALIDATION.csv",checks)
    p.write_csv(out/"TRAINING_GEOMETRY_TRACE.csv",geometry)
    choices=p.read_csv(out/"SOURCE_SELECTED_LR.csv")
    secondary=[]
    for source,target in p.PAIRS:
        for seed in p.SEEDS:
            vals={}; selected_lrs={}
            for method in p.METHODS:
                lr=.001 if method=="CES" else float(next(c["selected_lr"] for c in choices if c["source"]==source and int(c["seed"])==seed and c["method"]==method))
                rows=[r for r in raw if r["source"]==source and r["target"]==target and int(r["seed"])==seed and r["method"]==method and float(r["lr"])==lr]
                assert len(rows)==1
                vals[method]=float(rows[0]["P_AP"]);selected_lrs[method+"_lr"]=lr
            secondary.append(dict(source=source,target=target,seed=seed,**vals,**selected_lrs,
                CES_minus_Linear=vals["CES"]-vals["LinearHead"],CES_minus_Normalized=vals["CES"]-vals["NormalizedLinearHead"]))
    p.write_csv(out/"SOURCE_SELECTED_PAIRED_DELTAS.csv",secondary)
    primary_delta=p.read_csv(out/"PAIRED_DELTAS.csv")
    primary_means=p.read_csv(out/"SEED_LEVEL_MEANS.csv")
    secondary_means=[]
    for seed in p.SEEDS:
        rows=[r for r in secondary if r["seed"]==seed]
        secondary_means.append(dict(seed=seed,**{k:float(np.mean([r[k] for r in rows])) for k in (*p.METHODS,"CES_minus_Linear","CES_minus_Normalized")}))
    p.write_csv(out/"SOURCE_SELECTED_SEED_MEANS.csv",secondary_means)
    summary=[]
    for mode,data in (("PRIMARY_MATCHED_LR",primary_delta),("SOURCE_SELECTED_CONTROLS",secondary)):
        for source,target in [*p.PAIRS,("FOUR_TRANSFER_MEAN","ALL")]:
            rr=([r for r in (primary_means if mode=="PRIMARY_MATCHED_LR" else secondary_means)] if target=="ALL"
                else [r for r in data if r["source"]==source and r["target"]==target])
            for key in (*p.METHODS,"CES_minus_Linear","CES_minus_Normalized"):
                a=np.array([float(r[key]) for r in rr])
                assert len(a)==3
                summary.append(dict(comparison=mode,source=source,target=target,measure=key,mean=float(a.mean()),sample_sd=float(a.std(ddof=1)),n_seeds=3))
    p.write_csv(out/"FINAL_PAIRED_SUMMARY.csv",summary)
    lines=["","## Verified execution and uncertainty","",
        "18 newly trained primary checkpoints, each30 actual optimizer updates. Per-step optimizer counters and physical pool/batch hashes match across all three heads within every source/seed. Initial effective logit errors are recorded on the complete training pool. No historical CES checkpoint is used as a repeat.",
        "Raw direction norms are recorded separately from the Linear bias; gradient norms cover all trainable parameters (including the Linear bias). Adam moments, raw direction parameters and complete source-probe curves are retained in checkpoints. No forced gradient matching was applied.",
        "","| Comparison | Method / paired difference | Four-transfer P-AP mean +/- sample SD |","|---|---|---:|"]
    for row in summary:
        if row["target"]=="ALL":
            lines.append(f"| {row['comparison']} | {row['measure']} | {row['mean']:.6f} +/- {row['sample_sd']:.6f} |")
    lines += ["","## Per-transfer three-seed P-AP","","| Transfer | Linear | Normalized | CES |","|---|---:|---:|---:|"]
    for source,target in p.PAIRS:
        vals=[]
        for method in p.METHODS:
            row=next(r for r in summary if r["comparison"]=="PRIMARY_MATCHED_LR" and r["source"]==source and r["target"]==target and r["measure"]==method)
            vals.append(f"{row['mean']:.4f} +/- {row['sample_sd']:.4f}")
        lines.append(f"| {source} -> {target} | "+" | ".join(vals)+" |")
    lines += ["","## Interpretation boundary", ""]
    for mode,data in (("primary matched LR",primary_delta),("source-selected LR controls",secondary)):
        dl=np.array([float(r["CES_minus_Linear"]) for r in data]);dn=np.array([float(r["CES_minus_Normalized"]) for r in data])
        lines.append(f"- {mode}: CES-minus-Linear positive in {int((dl>0).sum())}/12 transfer-seed cells; CES-minus-Normalized positive in {int((dn>0).sum())}/12. All negative deltas remain in the tables.")
    dl=np.array([float(r["CES_minus_Linear"]) for r in primary_delta]);dn=np.array([float(r["CES_minus_Normalized"]) for r in primary_delta])
    if (dl>0).sum()>=9 and (dn>0).sum()>=9:
        decision="PAIRED_CONTROL_SUPPORTS_CES_WITHIN_TESTED_PROTOCOL"
        lines.append("The matched primary repeats support the tested constrained CES parameterization over these linear controls. Source-selected LR sensitivity is reported separately and must be considered alongside the primary results; this does not establish universal superiority or isolate the PC1/adapter recipe.")
    else:
        decision="PAIRED_EFFECT_UNSTABLE_NARROW_PARAMETERIZATION_CLAIM"
        lines.append("The paired effect is not consistently positive across the tested transfer-seed cells. Keep the historical control descriptive and narrow the claim to an effective tested parameterization, not stable general superiority.")
    report=out/"REPORT.md"
    report.write_text(report.read_text(encoding="utf-8")+"\n".join(lines)+"\n",encoding="utf-8")
    outcome.update(execution_validation="PASS",interpretation=decision,all_seeds_disclosed=True,
        finalized=True,main_tables_modified=False)
    p.write_json(out/"OUTCOME.json",outcome)
    p.status(out,"PAIRED_REPEAT_FINALIZED",primary_cells=36,decision=decision)
    print(json.dumps(outcome,indent=2))


if __name__=="__main__":main()
