# Code map

All paths below are relative to the extracted `sjp_ces_source/` directory.

| Role | Source |
|---|---|
| Visual outputs and feature normalization | phase_xa/official_task_forward.py |
| Attention-aware visual forward | phase_x/gradient_visual_forward.py |
| Fixed1/3/5 neighborhood aggregation | models/mlp.py: average_neighbor |
| Frozen modified CLIP implementation | AnomalyCLIP_lib/ |
| Physical source preflight/certification | experiments/rebuild_traceable_visa_source.py |
| Fresh FP32 source extraction | experiments/ (see manifest names containing traceable/source/certification) |
| Certified source build and30-update CES | experiments/source_build_traceable_visa.py |
| Certified paired endpoint target evaluation | experiments/evaluate_traceable_visa_targets.py |
| CES/linear heads and NTR helper | experiments/phase_ces_linear_head_control.py |
| Exact BCE/Dice helper | experiments/phase_xlii_patch_specialized_adaptation.py |
| Residual adapter definition | experiments/phase_xliii_adapter_head_decomposition.py |
| Circular mask shift definition | experiments/phase_spatial_supervision_transfer_audit.py |
| Frozen final evaluator | experiments/final_unified_evaluation.py |
| Repaired component source-pool comparison | experiments/repair_component_ablation_original_cache.py |
| Aligned head/capacity controls | experiments/repair_linear_capacity_aligned.py |
| Six-metric ablation completion | experiments/evaluate_aligned_ablation_six_metrics.py |
| MVTec exact-VisA-style source control | experiments/audit_repair_mvtec_exact_visa_recipe.py |
| Cache-only recipe metric completion | experiments/complete_mvtec_recipe_six_metrics.py |
| Figure scripts | visualization/ces_initialization_3d/ |

Keep dependencies in their original locations. Files named `phase_*` include
historical/exploratory models, not a recommendation to run every experiment.
Old loader/carrier paths cannot retrospectively certify historical pairing.
Current certified preparation is identified by explicit manifests, asset hashes
and source/target approval gates, not only by filenames.

`cesad_core.py` at repository root is a mechanically extracted reading/testing
surface. Its exact symbol provenance is in CORE_EXTRACTION_TRACE.json. The
historical modules inside the archive remain the authoritative experiment code.
