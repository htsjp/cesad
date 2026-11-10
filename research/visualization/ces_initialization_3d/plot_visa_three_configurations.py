"""VisA direct initialization versus saved prepared update0/update30.

Direct initialization uses the entire SAME physical CES source pool, not the
display sample. The existing5000 display IDs are reused; no sampling/training.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

from refine_figures_final import OUT as BASE, BLUE, ORANGE, TEAL, RED, sha
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT=BASE.parents[1]
OUT=BASE/'visa_three_configurations_fixed_basis'
sys.path.insert(0,str(ROOT))
from experiments.phase_xxxviii_visual_state_representation import norm_np
from experiments.phase_ces_linear_head_control import unit_np


def unit(x):
    x=np.asarray(x,np.float64)
    return x/max(np.linalg.norm(x),1e-12)


def main():
    OUT.mkdir(exist_ok=True)
    previous={p.name:sha(p) for p in BASE.iterdir() if p.is_file()}
    certpath=ROOT/'results/visa_source_traceable_rebuild/formal/source_feature_certification/SOURCE_FEATURE_CERTIFICATE.json'
    cert=json.loads(certpath.read_text(encoding='utf-8'))
    planpath=ROOT/'results/visa_source_traceable_rebuild/formal/source_build_review/plans/CES/POOL_PLAN.npz'
    plan=np.load(planpath);pairs=plan['physical_pairs'];labels=plan['labels']
    rawpath=Path(cert['patch_features']['path'])
    assert rawpath.stat().st_size==cert['patch_features']['bytes']
    for key in ('source_labels','image_feature_manifest'):
        assert sha(cert[key]['path'])==cert[key]['sha256']
    raw=np.load(rawpath,mmap_mode='r')
    gt=np.load(cert['source_labels']['path'],mmap_mode='r')
    assert raw.dtype==np.float32 and raw.shape==(2162,1369,1024)
    assert np.array_equal(gt[pairs[:,0],pairs[:,1]],labels)
    physical=pd.read_csv(cert['image_feature_manifest']['path'])
    # Authenticate every physical feature row actually consumed from the raw pool.
    import hashlib
    for pos in np.unique(pairs[:,0]):
        row=physical.iloc[int(pos)]
        assert int(row.source_position)==pos and int(row.patch_feature_row)==pos
        assert hashlib.sha256(np.asarray(raw[int(pos)]).tobytes()).hexdigest()==row.patch_sha256
    z=norm_np(np.stack([raw[int(pos),int(pid)] for pos,pid in pairs]))
    # Exact unconditioned MVTec prepare_source rule: unit per-label pooled means.
    sn=unit_np(z[labels==0].mean(0));sa=unit_np(z[labels==1].mean(0))
    m=.5*(sn+sa);d=.5*(sa-sn);alpha=float(np.linalg.norm(d));v0=unit_np(d)
    sn0=unit(m-alpha*unit(v0));sa0=unit(m+alpha*unit(v0))

    manifest=pd.read_csv(BASE/'sample_manifest.csv')
    sample=manifest[manifest.source=='VisA'].copy().reset_index(drop=True)
    selected=sample.pool_index.to_numpy(dtype=int)
    assert np.array_equal(pairs[selected],sample[['source_position','patch_index']].to_numpy())
    assert np.array_equal(labels[selected],sample.label.to_numpy())
    locked=np.load(BASE/'sampled_features_geometry_pca.npz')
    prepared=locked['panel_0_x']
    assert np.array_equal(prepared,locked['panel_1_x'])
    assert np.array_equal(labels[selected],locked['panel_0_y'])
    configs=[dict(name='Direct Initialization',x=z[selected].astype(np.float64),m=m,sn=sn0,sa=sa0,alpha=alpha)]
    for i,name in ((0,'Prepared Initialization'),(1,'After 30 Updates')):
        configs.append(dict(name=name,x=locked[f'panel_{i}_x'],m=locked[f'panel_{i}_m'],
            sn=locked[f'panel_{i}_sn'],sa=locked[f'panel_{i}_sa']))
    # Keep the original VisA basis; adding a configuration must not rotate the
    # already displayed endpoints. Direct features are an out-of-fit projection.
    components=locked['VisA_pca_components'];mean=locked['VisA_pca_mean']
    variance=locked['VisA_pca_explained_variance_ratio']
    prior=json.loads((BASE/'final_render_identity.json').read_text(encoding='utf-8'))
    saved=dict(pca_components=components,pca_mean=mean,
               pca_variance_ratio=variance,direct_m=m,
               direct_alpha=np.array(alpha),direct_v0=v0,pool_physical_pairs=pairs,pool_labels=labels)
    metrics=[];coordinates=[]
    for i,c in enumerate(configs):
        if i==0:
            c['xyz']=(c['x']-mean)@components.T
            c['markers']=(np.stack([c['m'],c['sn'],c['sa']])-mean)@components.T
        else:
            c['xyz']=locked[f'panel_{i-1}_xyz'].copy()
            c['markers']=locked[f'panel_{i-1}_markers'].copy()
            assert np.array_equal(c['xyz'],locked[f'panel_{i-1}_xyz'])
            assert np.array_equal(c['markers'],locked[f'panel_{i-1}_markers'])
        c['w']=c['sa']-c['sn'];c['wp']=components@c['w'];c['q']=c['x']@c['w']
        assert np.allclose(c['markers'][2]-c['markers'][1],c['wp'],atol=1e-12)
        for key in ('x','xyz','markers','w','wp','q','m','sn','sa'):
            saved[f'panel_{i}_{key}']=c[key]
        metrics.append(dict(configuration=c['name'],source='Full VisA',seed=0,
            displayed_normal=int(sum(sample.label==0)),displayed_anomaly=int(sum(sample.label==1)),
            weight_norm=float(np.linalg.norm(c['w'])),normal_mean_score=float(c['q'][sample.label==0].mean()),
            anomaly_mean_score=float(c['q'][sample.label==1].mean()),
            original_prepared_pca_cumulative_variance=float(variance.sum())))
        coordinates.append(pd.DataFrame(dict(panel='abc'[i],patch_id=sample.patch_id,label=sample.label,
            x=c['xyz'][:,0],y=c['xyz'][:,1],z=c['xyz'][:,2],q=c['q'])))
    assert np.array_equal(configs[1]['xyz'],configs[2]['xyz'])
    sample.to_csv(OUT/'sample_manifest.csv',index=False)
    pd.concat(coordinates).to_csv(OUT/'projected_patch_scores.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'geometry_metrics.csv',index=False)
    np.savez_compressed(OUT/'saved_geometry_and_projection.npz',**saved)

    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,3.05))
    fig.subplots_adjust(left=.005,right=.995,bottom=.30,top=.99,wspace=.005)
    points=np.concatenate([np.concatenate([c['xyz'],c['markers']]) for c in configs])
    limits=np.array(prior['geometry'][0]['limits'])
    lo=limits[:,0];hi=limits[:,1]
    assert np.all(points>=lo) and np.all(points<=hi),'Do not hide out-of-range direct points'
    titles=('(a) VisA\nDirect Initialization','(b) VisA\nPrepared Initialization',
            '(c) VisA\nAfter 30 Updates')
    for i,c in enumerate(configs):
        ax=fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False)
        for label,color in ((0,BLUE),(1,ORANGE)):
            xyz=c['xyz'][sample.label.to_numpy()==label]
            ax.scatter(*xyz.T,s=1.1,color=color,alpha=.17,edgecolors='none',depthshade=False,zorder=1)
        center,sn,sa=c['markers']
        ax.quiver(*sn,*c['wp'],length=1,normalize=False,color='#303438',linewidth=1,
                  arrow_length_ratio=.17,zorder=5)
        for point,color in ((sn,TEAL),(sa,RED)):
            ax.scatter(*point,s=25,color=color,edgecolors='white',linewidths=.4,depthshade=False,zorder=7)
        ax.scatter(*center,s=52,color='#202020',marker='*',edgecolors='white',linewidths=.4,depthshade=False,zorder=8)
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
        ax.set_box_aspect((1,1,1));ax.view_init(elev=21,azim=-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False;axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.015,titles[i],transform=ax.transAxes,ha='center',va='top',
                  fontsize=8.5,fontweight='semibold',linespacing=1.25)
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=4,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=4,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=7,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4.5,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4.5,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=4,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,
        frameon=False,columnspacing=1.5,handletextpad=.55,handlelength=1.25,labelspacing=.55)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',strict=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'visa_three_configurations.{ext}',dpi=600)
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)

    for name,identity in previous.items():
        assert sha(BASE/name)==identity
    (OUT/'report.md').write_text(
        '# VisA three-configuration geometry\n\n'
        'Panel(a) is a newly computed analytical direct initialization, not a trained '
        'checkpoint or an existing target result. It implements the MVTec matched-control '
        'initialization rule on the certified VisA physical source pool: L2 raw patch features; '
        'unit normal/anomaly pooled means; m=(sA+sN)/2, d=(sA-sN)/2, alpha=||d||, '
        'v0=unit(d); reconstruct normalized CES states. No PC1, auxiliary states or adapter. '
        'Category quotas are inherited from the locked pool, not reselected for this figure.\n\n'
        f'Full pool: {len(labels)} patches ({sum(labels==0)} normal, {sum(labels==1)} anomaly). '
        'All consumed physical feature rows were checked against the certified row hashes; '
        'labels were checked against the certified nearest-ID mask array.\n\n'
        'Panels(b,c) reuse exactly the saved prepared VisA initialization/update30 states '
        'and frozen transformed features, from one BCE+Dice+NTR run. They are not an '
        'NTR-only ablation. No head, adapter or encoder was trained.\n\n'
        'All three panels reuse the existing5000 display patch IDs (2500/class). '
        'The ORIGINAL prepared-VisA PCA basis is retained, without any refitting. '
        'Panels(b,c) copy their coordinates and geometry markers bitwise from the original '
        'panels(a,b). The added direct initialization is projected out-of-fit into that '
        'same basis. All3 panels retain the original prepared limits, camera and box '
        'aspect. Every direct point/marker fits within those limits; no point is clipped. '
        'The previous joint-PCA version is retained separately.\n\n'
        f'Original prepared-feature PCA cumulative variance: {variance.sum():.8f}; '
        'this percentage is not a newly fitted explained-variance estimate for raw features. '
        'This low-dimensional view cannot prove full-dimensional discrimination or '
        'target improvement. Direct versus prepared changes the whole preparation chain, '
        'not just CES optimization. Arrows have their exact projected state-difference length.\n\n'
        'Initialization rule evidence: experiments/phase_ces_lhc_paired_repeat.py '
        'prepare_source, with norm_np and unit_np reused directly. The source certificate, '
        'plan and prepared endpoint provenance remain in the adjacent original artifacts.\n',encoding='utf-8')
    identity=dict(source_certificate_sha256=sha(certpath),pool_plan_sha256=sha(planpath),
        raw_feature_certificate=cert['patch_features'],raw_verification='all consumed rows individually hashed',
        verified_feature_rows=int(len(np.unique(pairs[:,0]))),
        original_display_manifest_sha256=sha(BASE/'sample_manifest.csv'),
        original_saved_geometry_sha256=sha(BASE/'sampled_features_geometry_pca.npz'),
        previous_files_unchanged=True,training=False,resampling=False,
        pca_refitted=False,prepared_coordinates_and_markers_bitwise_match_original=True,
        joint_pca='original prepared-VisA basis; direct configuration projected out-of-fit')
    (OUT/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print('Output:',OUT,flush=True)


if __name__=='__main__':
    main()
