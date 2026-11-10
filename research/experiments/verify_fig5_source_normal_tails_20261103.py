"""Independent saved-data checks; no training, encoder or target evaluation."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments import plot_fig5_source_normal_tails_20261103 as job
from experiments import final_unified_evaluation as ev


def main():
    torch.set_num_threads(4)
    seal=ev.readjson(job.OUT/'DATA_SEALED.json')
    for rec in seal['data_files']:job.check(rec)
    cert=ev.readjson(job.CERT);job.check(cert['source_labels'])
    labels=np.load(cert['source_labels']['path'],mmap_mode='r')
    df=pd.read_csv(job.OUT/'FIG5_PATCH_SCORES.csv',float_precision='round_trip')
    means=pd.read_csv(job.OUT/'FIG5_CATEGORY_PAIRED.csv',float_precision='round_trip')
    curve=pd.read_csv(job.OUT/'FIG5_SURVIVAL_CURVES.csv',float_precision='round_trip')
    sample=ev.readcsv(job.OUT/'SAMPLE_MANIFEST_LOCKED_BEFORE_SCORING.csv')
    with np.load(job.OUT/'SAMPLED_TRANSFORMED_FEATURES.npz') as cache:
        z=cache['z'];pairs=cache['physical_pairs'];q=cache['scores']
    assert len(df)==12000 and len(set(df.patch_id))==12000
    assert df.patch_id.tolist()==[r['patch_id'] for r in sample]
    assert np.array_equal(pairs,np.asarray(df[['source_position','patch_index']]))
    assert np.all(labels[pairs[:,0],pairs[:,1]]==0)
    assert np.max(np.abs(df[['q_ces','q_ntr']].to_numpy()-q))<1e-9
    assert np.all(df.groupby('category').size()==1000)
    assert np.all(df.groupby('source_position').size()==20)
    assert np.all(df.groupby('category').source_position.nunique()==50)
    assert means.category.tolist()==list(job.CATEGORIES)
    for category in job.CATEGORIES:
        idx=np.flatnonzero(df.category.to_numpy()==category)
        order=np.lexsort((pairs[idx,1],pairs[idx,0],-q[idx,0]))
        expected=np.sort(idx[order[:100]])
        actual=np.flatnonzero((df.category.to_numpy()==category)&df.hard_normal.to_numpy())
        assert np.array_equal(expected,actual)
        row=means[means.category==category].iloc[0]
        assert abs(q[actual,0].mean(dtype=np.float64)-row.hard_mean_ces)<1e-12
        assert abs(q[actual,1].mean(dtype=np.float64)-row.hard_mean_ntr)<1e-12
    for j,column in enumerate(('survival_ces','survival_ntr')):
        expected=(12000-np.searchsorted(np.sort(q[:,j].astype(np.float64)),curve.threshold.to_numpy(),side='right'))/12000
        assert np.max(np.abs(expected-curve[column].to_numpy()))<1e-15
    audit=ev.readjson(job.OUT/'DATA_AUDIT.json')
    score_error=[]
    for j,key in enumerate(('trained_CES','trained_CES_NTR')):
        cp=torch.load(audit[key],map_location='cpu',weights_only=False)
        assert cp['step']==30 and cp['rho']==(None if j==0 else .1)
        state=cp['raw_model'];v=F.normalize(state['v'],dim=0)
        delta=F.normalize(state['m']+state['alpha']*v,dim=0)-F.normalize(state['m']-state['alpha']*v,dim=0)
        result=(torch.from_numpy(z)@delta).numpy()
        error=float(np.max(np.abs(result-q[:,j])));assert error<1e-7
        score_error.append(error)
    delivery=ev.readjson(job.OUT/'FIG5_DELIVERY.json')
    for rec in delivery['figure_files']:job.check(rec)
    qa=ev.readjson(job.OUT/'FIG5_PDF_QA.json')
    assert qa['collisions']['verdict']=='PASS' and qa['text']['below_minimum_count']==0
    result=dict(status='PASS',categories=12,patches=12000,source_images=600,hard_patches=1200,
        actual_category_quota=1000,actual_image_quota=20,normal_labels_verified=True,
        sample_identity_equal=True,hard_subsets_fixed_by_CES_only=True,
        exact_strict_survival_verified=True,shared_original_raw_scoring_verified=True,
        maximum_scoring_difference=score_error,figure_hashes_verified=True,
        manuscript_or_tables_changed=False,training=False,encoder_forward=False,
        source_score_matrix_products_only=True)
    ev.savejson(job.OUT/'INDEPENDENT_DATA_VERIFICATION.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
