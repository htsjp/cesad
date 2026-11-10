"""Paired VisA source geometry and full-dimensional score distributions.

Contract: a descriptive2x3 grid, not a causal ablation. Raw and prepared source
features share one full-SVD PCA; update0/update30 share frozen prepared features.
No training, resampling, image encoding, target access or formal-table edits.
"""
from pathlib import Path
import sys
import os
import json
import hashlib
import inspect
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
OUT=BASE/'visa_ces_three_stage_final'
WORK=Path('E:/MRAD_visa_source_traceable_rebuild/source_build/seed0')
FORMAL=ROOT/'results/visa_source_traceable_rebuild/formal'
QA=Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(QA))
from experiments.phase_xxxviii_visual_state_representation import norm_np
from experiments.phase_ces_linear_head_control import CESHead, unit_np
from experiments.phase_xliii_adapter_head_decomposition import AdapterToState
from experiments.source_build_traceable_visa import pool_features, run_ces
from experiments.phase_ces_lhc_paired_repeat import prepare_source
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE,ORANGE,TEAL,RED='#4E79A7','#CE8055','#238C91','#963F40'
NAMES=('Direct Initialization','Prepared Initialization','After 30 Updates')
TITLES=tuple(f'({letter}) VisA - {name}' for letter,name in zip('abc',NAMES))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def readjson(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def trace(fn):
    return dict(file=inspect.getsourcefile(fn),function=fn.__qualname__,line=inspect.getsourcelines(fn)[1])


def main():
    OUT.mkdir(exist_ok=True)
    # Hash all pre-existing visualization evidence, including older projections.
    protected={str(p):sha(p) for p in BASE.rglob('*')
               if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
    torch.set_num_threads(4)
    inputs=[]

    def identity(path):
        p=Path(path);r=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p));inputs.append(r);return r

    certpath=FORMAL/'source_feature_certification/SOURCE_FEATURE_CERTIFICATE.json'
    cert=readjson(certpath);identity(certpath)
    planpath=FORMAL/'source_build_review/plans/CES/POOL_PLAN.npz'
    identity(planpath);plan=np.load(planpath);pairs=plan['physical_pairs'];pool_y=plan['labels']
    samplepath=BASE/'sample_manifest.csv'
    identity(samplepath)
    sample=pd.read_csv(samplepath);sample=sample[sample.source=='VisA'].copy().reset_index(drop=True)
    selection=sample.pool_index.to_numpy(dtype=int)
    assert np.array_equal(pairs[selection],sample[['source_position','patch_index']].to_numpy())
    assert np.array_equal(pool_y[selection],sample.label.to_numpy())
    assert len(sample)==5000 and not sample.patch_id.duplicated().any()
    y=sample.label.to_numpy(dtype=int)
    assert sum(y==0)==sum(y==1)==2500
    for key in ('source_labels','image_feature_manifest'):
        assert identity(cert[key]['path'])['sha256']==cert[key]['sha256']
    rawpath=Path(cert['patch_features']['path'])
    assert rawpath.stat().st_size==cert['patch_features']['bytes']
    raw=np.load(rawpath,mmap_mode='r')
    assert raw.dtype==np.float32 and raw.shape==(2162,1369,1024)
    gt=np.load(cert['source_labels']['path'],mmap_mode='r')
    assert np.array_equal(gt[pairs[:,0],pairs[:,1]],pool_y)
    images=pd.read_csv(cert['image_feature_manifest']['path'])
    positions=np.unique(pairs[:,0])
    print('Verify certified physical feature rows:',len(positions),flush=True)
    for pos in positions:
        record=images.iloc[int(pos)]
        assert int(record.source_position)==int(record.patch_feature_row)==pos
        assert hashlib.sha256(np.asarray(raw[int(pos)]).tobytes()).hexdigest()==record.patch_sha256
    raw_pool=norm_np(np.stack([raw[int(pos),int(pid)] for pos,pid in pairs]))
    raw_sample=raw_pool[selection]
    # MVTec-style direct geometry uses the full locked source pool, not the
    # balanced display sample; class means are normal/anomaly pooled means.
    sn_direct=unit_np(raw_pool[pool_y==0].mean(0))
    sa_direct=unit_np(raw_pool[pool_y==1].mean(0))
    direct_m=.5*(sn_direct+sa_direct);direct_d=.5*(sa_direct-sn_direct)
    direct_alpha=float(np.linalg.norm(direct_d));direct_v=unit_np(direct_d)

    cps=[]
    for step in (0,30):
        path=WORK/f'checkpoints/CES/update_{step:03d}.pt'
        identity(path);cp=torch.load(path,map_location='cpu',weights_only=False)
        assert cp['stage']=='CES' and cp['step']==step
        for key in ('pc1','E1','E2','pool_plan'):
            assert identity(cp['identities'][key]['path'])['sha256']==cp['identities'][key]['sha256']
        cps.append(cp)
    assert cps[0]['identities']==cps[1]['identities']
    state0,state30=[c['params']['raw_model'] for c in cps]
    assert all(torch.equal(state0[k],state30[k]) for k in ('m','alpha'))
    assert cps[0]['params']['frozen']==cps[1]['params']['frozen']
    assert not torch.equal(state0['v'],state30['v'])
    q=np.load(WORK/'pc1.npy')
    e1=torch.load(WORK/'assets/E1_formal.pt',map_location='cpu',weights_only=False)
    e2=torch.load(WORK/'assets/E2_formal.pt',map_location='cpu',weights_only=False)
    assert e1['supervision']=='true' and e2['supervision']=='shifted'
    expected_d=.5*(e1['patch_states'][1]-e1['patch_states'][0])
    expected_m=.5*(e2['patch_states'][0]+e2['patch_states'][1])
    assert np.array_equal(state0['m'].numpy(),expected_m)
    assert abs(float(state0['alpha'])-float(np.linalg.norm(expected_d)))<1e-7
    assert np.array_equal(state0['v'].numpy(),unit_np(expected_d))
    adapter=AdapterToState(1024,e2['patch_states'])
    adapter.load_state_dict(e2['model'],strict=True);adapter.eval()
    for p in adapter.parameters():p.requires_grad_(False)
    assert tuple(adapter.a.weight.shape)==(4,1024) and tuple(adapter.b.weight.shape)==(1024,4)
    residual=pool_features(raw,dict(physical_pairs=pairs[selection]),q,norm_np)
    with torch.inference_mode():
        checked_prepared=adapter.adapt(torch.from_numpy(residual)).numpy()
    prepfile=WORK/'CES_pool/features.npy';identity(prepfile)
    prepared=np.asarray(np.load(prepfile,mmap_mode='r')[selection],np.float32)
    transform_error=float(np.max(abs(checked_prepared-prepared)))
    assert transform_error<1e-6,transform_error
    # Use the saved runtime feature array; CPU/GPU verification roundoff is not
    # allowed to silently replace it.
    lockedfile=BASE/'sampled_features_geometry_pca.npz';identity(lockedfile)
    locked=np.load(lockedfile)
    assert np.array_equal(prepared.astype(np.float64),locked['panel_0_x'])
    assert np.array_equal(locked['panel_0_x'],locked['panel_1_x'])
    assert np.max(abs(np.linalg.norm(raw_sample,axis=1)-1))<1e-5
    assert np.max(abs(np.linalg.norm(prepared,axis=1)-1))<1e-5

    source_sample=sample.merge(images[['source_position','image','image_sha256','patch_sha256']],
                               on='source_position',validate='many_to_one',sort=False)
    assert np.array_equal(source_sample.patch_id.to_numpy(),sample.patch_id.to_numpy())
    source_sample.to_csv(OUT/'sample_manifest.csv',index=False)
    direct=CESHead(direct_m,direct_alpha,direct_v)
    heads=[direct]
    for state in (state0,state30):
        h=CESHead(state['m'].numpy(),float(state['alpha']),state['v'].numpy())
        h.load_state_dict(state,strict=True);heads.append(h)
    configs=[]
    for i,(name,head,x) in enumerate(zip(NAMES,heads,(raw_sample,prepared,prepared))):
        head.eval()
        for parameter in head.parameters():parameter.requires_grad_(False)
        with torch.inference_mode():
            v=F.normalize(head.v,dim=0)
            sn=F.normalize(head.m-head.alpha*v,dim=0).numpy()
            sa=F.normalize(head.m+head.alpha*v,dim=0).numpy()
            scores=head(torch.from_numpy(x)).numpy()
        w=sa-sn
        score_error=float(np.max(abs(x.astype(np.float64)@w.astype(np.float64)-scores)))
        assert score_error<1e-6
        if i>0:
            assert np.max(abs(scores-locked[f'panel_{i-1}_scores']))<1e-6
        configs.append(dict(name=name,x=x,m=head.m.numpy().copy(),alpha=float(head.alpha),
            v=v.numpy(),sn=sn,sa=sa,w=w,scores=scores,score_formula_error=score_error))
    assert np.array_equal(configs[1]['x'],configs[2]['x'])
    assert np.array_equal(configs[1]['m'],configs[2]['m'])
    assert configs[1]['alpha']==configs[2]['alpha']

    print('Fit one joint full-SVD PCA on raw and prepared paired features.',flush=True)
    pca=PCA(n_components=3,svd_solver='full')
    pca.fit(np.concatenate([raw_sample.astype(np.float64),prepared.astype(np.float64)]))
    metrics=[];coords=[];saved=dict(F_raw=raw_sample,F_prep=prepared,labels=y,
        pool_physical_pairs=pairs,pool_labels=pool_y,display_pool_indices=selection,
        pca_components=pca.components_,pca_mean=pca.mean_,
        pca_variance_ratio=pca.explained_variance_ratio_,pca_singular_values=pca.singular_values_)
    for i,c in enumerate(configs):
        c['xyz']=pca.transform(c['x'].astype(np.float64))
        c['markers']=pca.transform(np.stack([c['m'],c['sn'],c['sa']]).astype(np.float64))
        # Directions are never centered. Float32 sA-sN subtraction introduces
        # tiny roundoff, tested against independently projected endpoints.
        c['wp']=pca.components_@c['w'].astype(np.float64)
        arrow_error=float(np.max(abs(c['markers'][2]-c['markers'][1]-c['wp'])))
        assert arrow_error<1e-7
        metrics.append(dict(panel='abc'[i],configuration=c['name'],source='Full VisA',seed=0,
            normal_patches=int(sum(y==0)),anomaly_patches=int(sum(y==1)),
            m_norm=float(np.linalg.norm(c['m'])),alpha=c['alpha'],
            normal_state_norm=float(np.linalg.norm(c['sn'])),anomaly_state_norm=float(np.linalg.norm(c['sa'])),
            effective_weight_norm=float(np.linalg.norm(c['w'])),
            mean_normal_score=float(c['scores'][y==0].mean()),mean_anomaly_score=float(c['scores'][y==1].mean()),
            anomaly_normal_mean_gap=float(c['scores'][y==1].mean()-c['scores'][y==0].mean()),
            source_sample_AP=100*average_precision_score(y,c['scores']),
            source_sample_AUROC=100*roc_auc_score(y,c['scores']),
            joint_pca_variance=float(pca.explained_variance_ratio_.sum()),
            projected_weight_energy_ratio=float(np.dot(c['wp'],c['wp'])/np.dot(c['w'].astype(np.float64),c['w'].astype(np.float64))),
            projected_arrow_endpoint_error=arrow_error,score_formula_error=c['score_formula_error']))
        for key in ('x','m','alpha','v','sn','sa','w','scores','xyz','markers','wp'):
            saved[f'panel_{i}_{key}']=c[key]
        coords.append(pd.DataFrame(dict(panel='abc'[i],patch_id=sample.patch_id,label=y,
            PC1=c['xyz'][:,0],PC2=c['xyz'][:,1],PC3=c['xyz'][:,2],score=c['scores'])))
    assert np.array_equal(configs[1]['xyz'],configs[2]['xyz'])
    displacement=np.linalg.norm(raw_sample.astype(np.float64)-prepared,axis=1)
    cosine=np.sum(raw_sample.astype(np.float64)*prepared,axis=1)/(
        np.linalg.norm(raw_sample.astype(np.float64),axis=1)*np.linalg.norm(prepared.astype(np.float64),axis=1))
    differences=dict(mean_feature_L2_difference=float(displacement.mean()),
        max_feature_L2_difference=float(displacement.max()),mean_cosine_similarity=float(cosine.mean()),
        mean_3D_displacement=float(np.linalg.norm(configs[0]['xyz']-configs[1]['xyz'],axis=1).mean()),
        prepared_features_and_coordinates_bitwise_identical=True,
        prepared_transform_verification_max_error=transform_error,
        all_consumed_raw_rows_verified=int(len(positions)))
    bins=np.linspace(min(c['scores'].min() for c in configs),max(c['scores'].max() for c in configs),51)
    densities=[[np.histogram(c['scores'][y==label],bins=bins,density=True)[0]
                for label in (0,1)] for c in configs]
    for group in densities:
        for density in group:assert abs(np.sum(density*np.diff(bins))-1)<1e-12
    saved['score_bin_edges']=bins;saved['score_densities']=np.array(densities)
    np.savez_compressed(OUT/'visa_ces_three_stage_data.npz',**saved)
    pd.DataFrame(metrics).to_csv(OUT/'visa_ces_three_stage_metrics.csv',index=False,float_format='%.10f')
    pd.concat(coords).to_csv(OUT/'projected_coordinates_and_scores.csv',index=False)

    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':7.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,
        'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    fig=plt.figure(figsize=(7.1,4.9))
    grid=fig.add_gridspec(2,3,left=.055,right=.99,bottom=.16,top=.99,
                        height_ratios=[1.6,1],wspace=.23,hspace=.36)
    points=np.concatenate([np.concatenate([c['xyz'],c['markers']]) for c in configs])
    lo=points.min(0);hi=points.max(0);span=hi-lo;lo-=span*.08;hi+=span*.08
    ymax=max(float(d.max()) for ds in densities for d in ds)*1.10
    for i,c in enumerate(configs):
        ax=fig.add_subplot(grid[0,i],projection='3d',computed_zorder=False)
        for label,color in ((0,BLUE),(1,ORANGE)):
            points=c['xyz'][y==label]
            ax.scatter(*points.T,s=.9,color=color,alpha=.17,edgecolors='none',depthshade=False,zorder=1)
        center,sn,sa=c['markers']
        ax.quiver(*sn,*c['wp'],length=1,normalize=False,color='#303438',linewidth=1,
                  arrow_length_ratio=.17,zorder=5)
        for point,color in ((sn,TEAL),(sa,RED)):
            ax.scatter(*point,s=24,color=color,edgecolors='white',linewidths=.4,depthshade=False,zorder=7)
        ax.scatter(*center,s=50,color='#202020',marker='*',edgecolors='white',linewidths=.4,depthshade=False,zorder=8)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=21,azim=-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False;axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.035,TITLES[i],transform=ax.transAxes,ha='center',va='top',
                  fontsize=7.5,fontweight='semibold')
        ax=fig.add_subplot(grid[1,i])
        for label,color,density in zip((0,1),(BLUE,ORANGE),densities[i]):
            ax.stairs(density,bins,fill=True,color=color,alpha=.075,linewidth=0)
            ax.stairs(density,bins,color=color,linewidth=1.1)
        ax.set(xlim=(bins[0],bins[-1]),ylim=(0,ymax),xlabel='Anomaly score',ylabel='Density')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#707070')
        ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(width=.55,length=2.5)
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=3.5,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=3.5,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=6,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=3.5,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),ncol=6,
               frameon=False,columnspacing=.85,handletextpad=.35,handlelength=1)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',strict=True)
    stem='visa_ces_three_stage_final'
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)
    assert all(sha(path)==value for path,value in protected.items()),'A prior artifact was modified'
    evidence=dict(input_files=inputs,raw_feature_identity=cert['patch_features'],
        raw_verification='certificate byte size + SHA256 of every consumed feature row; no full12GB rehash',
        differences=differences,metrics=metrics,source_pool_patches=len(pool_y),
        source_pool_normal=int(sum(pool_y==0)),source_pool_anomaly=int(sum(pool_y==1)),
        camera=dict(elev=21,azim=-58,limits=np.stack([lo,hi],axis=1).tolist(),box_aspect=[1,1,1]),
        sampling='exact existing5000 IDs,2500 per label; no resampling',
        pca=dict(solver='full',fit='concatenate(F_raw,F_prep),10000 rows',
                 cumulative_variance=float(pca.explained_variance_ratio_.sum())),
        source_trace=[trace(f) for f in (norm_np,unit_np,prepare_source,pool_features,run_ces,AdapterToState.adapt,CESHead.forward)],
        script=dict(path=str(Path(__file__)),sha256=sha(__file__)),original_files_unchanged=True)
    (OUT/'validation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    report=['# VisA three-stage paired source geometry',
        '## Panel definitions',
        '(a) Analytical direct initialization using the MVTec-style rule on the same '
        'VisA source pool: frozen extracted/aggregated raw patches -> L2 -> unit pooled '
        'normal/anomaly means -> m,alpha,v0 -> normalized states. No PC1, auxiliary '
        'state optimization or adapter is used in(a). This is not a trained direct '
        'checkpoint or a new target benchmark.',
        '(b) The certified prepared VisA update0 checkpoint. Normal-source PC1 removal '
        'and L2 precede the frozen E2 rank4 residual adapter and final L2. True-label '
        'E1 Stage-I states supply direction/scale; shifted-label E2 Stage-I states '
        'supply the pre-normalization center. Auxiliary states construct the scorer; '
        'they do not replace physical patch labels or independently move patch features.',
        '(c) The same source preparation/features, with the certified update30 direction '
        'after BCE+Dice+NTR. m,alpha and all feature-transform assets remain frozen.',
        '## A. Physical pairing',
        f'PASS: the existing5000 IDs are reused ({sum(y==0)} normal,{sum(y==1)} anomaly). '
        'Each ID maps to one physical source image row and37x37 patch index, checked '
        'against the locked pool plan and certified nearest-ID GT array. Full direct '
        f'initialization uses all{len(pool_y)} locked pool patches, not the balanced '
        'display sample. Raw feature SHA256 was rechecked for every consumed image '
        f'row ({len(positions)} rows). See sample_manifest.csv for image paths/hashes '
        'and complete display ordering. No new sampling or encoder inference.',
        '## B. Feature differences',json.dumps(differences,indent=2),
        'F_raw is the actual raw frozen-encoder cache after L2. F_prep is the saved '
        'final source-build CES feature pool, verified by replaying the exact '
        'PC1-removal/normalization/adapter chain on the same physical patches. '
        'The saved runtime features are retained, not replaced with CPU-roundoff copies.',
        '## C. Frozen-feature check',
        'PASS: panels(b,c) share bitwise-identical FP32 features and PCA coordinates. '
        'All panels share sample IDs, images and labels.',
        '## D. States and scoring',
        'PASS: each1024-dimensional state pair is used with its corresponding feature '
        'stage, before projection. Prepared m/alpha/v0 were checked against E1/E2 '
        'payloads, and both checkpoints against their bound PC1/E1/E2/plan hashes. '
        'The original CESHead.forward computes all displayed score distributions. '
        'Checks verify q=F@(sA-sN), unit state norms, fixed m/alpha and changed v. '
        'The arrow starts at sN and ends at sA; m is separately projected and is '
        'not forced onto the normalized-state midpoint. No arrow magnification.',
        '## E. Shared PCA and projection loss',
        'One new PCA(n_components=3,svd_solver=full) fits the concatenation of '
        '5000 raw-L2 and5000 prepared features, once each. The identical prepared '
        'cloud for update30 is not duplicated in PCA fitting. Each point uses '
        'PCA.transform; each difference vector uses components@w without centering. '
        'The projected difference agrees with the difference of state coordinates '
        'within FP32 subtraction tolerance. All panels share camera/limits/aspect. '
        'This replaces neither previous projection nor any old image. Expect the '
        'prepared clouds to rotate relative to the previous fixed-basis version.',
        f'Cumulative explained variance: {pca.explained_variance_ratio_.sum():.10f}. '
        'Per-direction retained energy is reported below, not printed in the image.',
        pd.DataFrame(metrics).to_markdown(index=False),
        '## Score distributions',
        'Full1024-dimensional official CES scores, never PCA scores. All six '
        'class/configuration histograms share the same50 bins, x/y limits and '
        'class-wise density normalization; each integrates to1. No KDE, calibration, '
        'zero/mean reference lines or performance annotations.',
        '## Reproducibility',
        'All input paths/hashes and source function/line traces are in validation.json. '
        'The data NPZ preserves both feature matrices, labels, pool/selection indices, '
        'states, scores, PCA mean/components/variance/singular values, bin edges and '
        'densities. CSVs preserve physical IDs and projected coordinates. Original '
        'visualization evidence is unchanged by SHA256 checks.',
        '## Interpretation boundaries',
        '(a)->(b) changes the full preparation chain, not PC1 alone. '
        '(b)->(c) is BCE+Dice+NTR optimization, not an NTR-only ablation. '
        'All1024-dimensional coordinates retain their corresponding feature/state '
        'stage; a common PCA is a descriptive visualization, not a learned alignment '
        'between spaces. Source training-pool AP/AUROC and3D state distances do not '
        'establish cross-dataset generalization or exact high-dimensional scoring '
        'capacity. Only seed0 is depicted; no uncertainty is inferred.',
        '## Source traces',json.dumps(evidence['source_trace'],indent=2)]
    (OUT/'visa_ces_three_stage_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(differences,indent=2),flush=True)
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print('Output:',OUT,flush=True)


if __name__=='__main__':
    main()
