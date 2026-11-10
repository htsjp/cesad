"""Verify saved outputs and export precision variants; no pixel recomputation."""
from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import final_unified_evaluation as ev

OUT=ROOT/'results/mvtec_recipe_six_metrics_20261109'
KEYS=('I_AUROC','I_AP','P_AUROC','P_F1max','P_AP','AUPRO')


def main():
    config=ev.readjson(OUT/'LOCKED_CONFIG.json')
    status=ev.readjson(OUT/'STATUS.json')
    assert status['phase']=='COMPLETE'
    for item in config['inputs']:
        path=Path(item['path'])
        # The source used in the completed run is preserved exactly. The only
        # subsequent source edit restores log streams during interpreter exit.
        if path.name=='complete_mvtec_recipe_six_metrics.py':path=OUT/'EXECUTED_SCRIPT.py'
        assert ev.sha(path)==item['sha256'],path
    assert ev.evaluator_hash()==config['evaluator_id']
    frame=pd.read_csv(OUT/'MVTec_TO_VisA_SIX_METRICS.csv')
    per=pd.read_csv(OUT/'PER_CATEGORY_METRICS.csv')
    assert len(frame)==2 and len(per)==24
    assert not per.duplicated(['variant','category']).any()
    assert np.isfinite(frame[list(KEYS)].to_numpy()).all()
    assert np.isfinite(per[list(KEYS)].to_numpy()).all()
    assert per.P_AP_parity_error.abs().max()<1e-6
    assert (frame.valid_categories==12).all() and (frame.n_images==2162).all()
    for _,row in frame.iterrows():
        group=per[per.variant==row.variant]
        assert len(group)==12
        for k in KEYS:assert abs(row[k]-group[k].mean())<1e-10
        assert abs(row.P_AP-row.previous_P_AP)<1e-10
        image=pd.read_csv(OUT/f'image_scores/{row.variant}.csv')
        assert len(image)==2162 and image.sample_id.nunique()==2162
        assert (image.top_k==55).all() and (image.image_grid=='74x74').all()
    frame.to_csv(OUT/'MVTec_TO_VisA_SIX_METRICS_6DP.csv',index=False,float_format='%.6f')
    display=frame[['variant',*KEYS]].copy()
    text=display.to_markdown(index=False,floatfmt='.2f')
    (OUT/'PAPER_TABLE.md').write_text('# MVTec to VisA, seed0, category-macro (%)\n\n'+text+'\n',encoding='utf-8')
    checks=dict(status='PASS',models=2,unique_model_categories=24,images_per_model=2162,
        valid_categories_per_metric=12,max_category_P_AP_parity_error=float(per.P_AP_parity_error.abs().max()),
        frozen_inputs_unchanged=True,training=False,model_inference=False,
        original_execution_code=ev.sha(OUT/'EXECUTED_SCRIPT.py'),
        cleanup_note='Metrics completed and STATUS was COMPLETE; first process exited nonzero only during final log-stream flushing. Source now restores streams before log closure. No metric or scientific setting changed.')
    ev.savejson(OUT/'FINAL_VERIFICATION.json',checks)
    report=OUT/'REPORT.md'
    original=report.read_text(encoding='utf-8')
    if '## Final verification' not in original:
        report.write_text(original+'\n## Final verification\n\n'+json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    print(text)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
