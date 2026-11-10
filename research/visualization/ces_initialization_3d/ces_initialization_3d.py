"""Saved-source geometry only: no training, image encoding or target evaluation.

The revised request compares VisA update0/update30, not direct/prepared VisA.
Physical patch identities are matched within each row. No feature movement is
introduced for the VisA endpoints, whose feature transform is frozen.
"""
from pathlib import Path
import sys
import json
import hashlib

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
VISA = Path('E:/MRAD_visa_source_traceable_rebuild/source_build/seed0')
MVTEC = Path('E:/MRAD_mvtec_exact_visa_recipe_checked/seed0')
REVIEW = ROOT/'results/visa_source_traceable_rebuild/formal/source_build_review'
QA = Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
sys.path.insert(0, str(ROOT))
from experiments.phase_ces_linear_head_control import CESHead
sys.path.insert(0, str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, ORANGE, TEAL, RED = '#4E79A7', '#CE8055', '#238C91', '#963F40'
N_PER_CLASS = 2500
SEED = 0


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def unit(x):
    x = np.asarray(x, np.float64)
    return x/max(np.linalg.norm(x), 1e-12)


def states(m, alpha, v):
    return unit(m-alpha*unit(v)), unit(m+alpha*unit(v))


def sample_indices(labels, categories):
    # Equal category quotas where available; deterministic redistribution when
    # anomaly availability is insufficient. No scores enter sampling.
    rng = np.random.default_rng(SEED)
    result = []
    for label in (0, 1):
        groups = [np.flatnonzero((labels == label) & (categories == c))
                  for c in sorted(set(categories))]
        target = min(N_PER_CLASS, sum(map(len, groups)))
        quotas = np.zeros(len(groups), dtype=int)
        while quotas.sum() < target:
            for i, g in enumerate(groups):
                if quotas[i] < len(g) and quotas.sum() < target:
                    quotas[i] += 1
        for g, q in zip(groups, quotas):
            result.extend(np.sort(rng.choice(g, q, replace=False)).tolist())
    return np.asarray(result, np.int64)


def main():
    torch.set_num_threads(4)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
        'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
        'xtick.labelsize':6.5, 'ytick.labelsize':6.5, 'legend.fontsize':7,
        'pdf.fonttype':42, 'svg.fonttype':'none', 'axes.linewidth':.6})
    inputs = []

    def record(path):
        path = Path(path)
        inputs.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
        return path

    def loadcp(path):
        return torch.load(record(path), map_location='cpu', weights_only=False)

    # VisA is the independently certified physical pool, not the old carrier pool.
    plan = np.load(record(REVIEW/'plans/CES/POOL_PLAN.npz'))
    vy = np.load(record(VISA/'CES_pool/labels.npy')).astype(int)
    vz = np.load(record(VISA/'CES_pool/features.npy'), mmap_mode='r')
    assert np.array_equal(vy, plan['labels'])
    cats = plan['categories']; vi = sample_indices(vy, cats)
    pairs = plan['physical_pairs'][vi]
    vmanifest = pd.DataFrame(dict(source='VisA', pool_index=vi,
        category=cats[vi], source_position=pairs[:, 0], patch_index=pairs[:, 1], label=vy[vi]))
    vmanifest['patch_id'] = [f'VisA/source_row{p:04d}/patch{k:04d}' for p,k in pairs]
    vcps = [loadcp(VISA/f'checkpoints/CES/update_{s:03d}.pt') for s in (0,30)]
    assert [c['step'] for c in vcps] == [0,30]
    vs = [c['params']['raw_model'] for c in vcps]
    assert all(torch.equal(vs[0][k], vs[1][k]) for k in ('m','alpha'))
    assert vcps[0]['params']['frozen'] == vcps[1]['params']['frozen']
    vx = np.asarray(vz[vi], np.float64)

    # MVTec control and preparation share physical IDs AND nearest-mask labels.
    mx, mcps, manifests = [], [], []
    for name in ('Original_MVTec','Exact_VisA_Recipe'):
        pool = MVTEC/name/'source_pools/Full_MVTec_seed0'
        df = pd.read_csv(record(pool/'PATCH_MANIFEST.csv'))
        y = np.load(record(pool/'labels.npy')).astype(int)
        x = np.load(record(pool/'features.npy'), mmap_mode='r')
        assert np.array_equal(y, df.label.to_numpy())
        if not mx:
            mi = sample_indices(y, df.category.to_numpy())
            my = y[mi]
        manifests.append(df)
        mx.append(np.asarray(x[mi], np.float64))
        mcps.append(loadcp(MVTEC/name/f'checkpoints/Full_MVTec_CES_seed0_{name}_lr0.001.pt'))
    pd.testing.assert_frame_equal(manifests[0], manifests[1])
    assert mcps[0]['batch_stream_hash'] == mcps[1]['batch_stream_hash']
    mm = manifests[0].iloc[mi].copy().rename(columns={'patch_flat_index':'patch_index'})
    mm['source'] = 'MVTec'
    mm['patch_id'] = [f'MVTec/source_row{p:04d}/patch{k:04d}'
                      for p,k in zip(mm.source_position, mm.patch_index)]
    columns = ['source','pool_index','category','source_position','patch_index','label','patch_id']
    manifest = pd.concat([vmanifest[columns], mm[columns]], ignore_index=True)
    assert not manifest.patch_id.duplicated().any()
    manifest.to_csv(OUT/'sample_manifest.csv', index=False)
    assert np.max(abs(np.linalg.norm(vx,axis=1)-1)) < 1e-5
    assert all(np.max(abs(np.linalg.norm(x,axis=1)-1)) < 1e-5 for x in mx)

    configs = []
    score_checks = []
    for i, (title, source, endpoint, x, y, cp) in enumerate([
        ('VisA - Initialization','VisA','update0',vx,vy[vi],vcps[0]),
        ('VisA - After 30 Updates','VisA','update30',vx,vy[vi],vcps[1]),
        ('MVTec - Matched Initialization','MVTec','initialization',mx[0],my,mcps[0]),
        ('MVTec - VisA-style Preparation','MVTec','initialization',mx[1],my,mcps[1])]):
        if source == 'VisA':
            raw = cp['params']['raw_model']
            m = raw['m'].numpy().astype(np.float64)
            alpha = float(raw['alpha']); v = raw['v'].numpy()
        else:
            g = cp['initial_geometry']; m = np.asarray(g['m'], np.float64)
            alpha = float(g['alpha']); v = g['init_v']
        sn, sa = states(m, alpha, v); w = sa-sn; scores=x@w
        # Verify the analytical FP64 reconstruction against the actual FP32
        # scoring implementation; use no independent plotting normalization.
        head = CESHead(np.asarray(m,np.float32),alpha,np.asarray(v,np.float32)).eval()
        with torch.inference_mode():
            official = head(torch.from_numpy(x.astype(np.float32))).numpy()
        error = float(np.max(abs(official-scores)))
        assert error < 1e-6, (title,error)
        score_checks.append(dict(configuration=title,official_forward_max_error=error))
        configs.append(dict(title=title,source=source,endpoint=endpoint,x=x,y=y,
            m=m,alpha=alpha,sn=sn,sa=sa,w=w,scores=scores))

    arrays, metrics, projected, pca_records = {}, [], [], []
    for row in (0,1):
        a,b = configs[2*row:2*row+2]
        # A joint row basis; repeated identical VisA arrays preserve identical PCA.
        pca = PCA(n_components=3,svd_solver='full')
        pca.fit(np.concatenate([a['x'], b['x']]))
        for key,value in [('components',pca.components_),('mean',pca.mean_),
                          ('explained_variance_ratio',pca.explained_variance_ratio_)]:
            arrays[f'{a["source"]}_pca_{key}'] = value
        pca_records.append(dict(source=a['source'],cumulative_variance=float(pca.explained_variance_ratio_.sum())))
        for j,c in enumerate((a,b)):
            c['xyz']=pca.transform(c['x'])
            c['markers']=pca.transform(np.stack([c['m'],c['sn'],c['sa']]))
            c['wp']=pca.components_@c['w']
            assert np.allclose(c['wp'],c['markers'][2]-c['markers'][1],atol=1e-12)
            prefix=f'panel_{2*row+j}'
            for key in ('x','y','m','sn','sa','w','scores','xyz','markers'):
                arrays[f'{prefix}_{key}']=c[key]
            metrics.append(dict(panel='abcd'[2*row+j],source=c['source'],configuration=c['title'],
                endpoint=c['endpoint'],normal_n=int(sum(c['y']==0)),anomaly_n=int(sum(c['y']==1)),
                effective_weight_norm=np.linalg.norm(c['w']),m_norm=np.linalg.norm(c['m']),alpha=c['alpha'],
                normal_mean_score=c['scores'][c['y']==0].mean(),
                anomaly_mean_score=c['scores'][c['y']==1].mean(),
                anomaly_normal_gap=c['scores'][c['y']==1].mean()-c['scores'][c['y']==0].mean(),
                sample_patch_AUROC=100*roc_auc_score(c['y'],c['scores']),
                sample_patch_AP=100*average_precision_score(c['y'],c['scores']),
                pca_variance_cumulative=pca.explained_variance_ratio_.sum(),
                projected_weight_norm=np.linalg.norm(c['wp']),
                weight_energy_retained=np.linalg.norm(c['wp'])**2/np.linalg.norm(c['w'])**2))
            ids = vmanifest.patch_id.to_numpy() if row==0 else mm.patch_id.to_numpy()
            projected.append(pd.DataFrame(dict(panel='abcd'[2*row+j],patch_id=ids,label=c['y'],
                PC1=c['xyz'][:,0],PC2=c['xyz'][:,1],PC3=c['xyz'][:,2],q=c['scores'],
                projection_on_unit_scoring_direction=c['scores']/np.linalg.norm(c['w']))))
        if row==0:
            assert np.array_equal(a['xyz'],b['xyz'])
    pd.DataFrame(metrics).to_csv(OUT/'geometry_metrics.csv',index=False,float_format='%.10f')
    pd.concat(projected).to_csv(OUT/'projected_patches_scores.csv',index=False)
    np.savez_compressed(OUT/'sampled_features_geometry_pca.npz',**arrays)

    # Reproduce the common-raw-space NUMERATOR angle, never compare transformed v.
    pc1path=ROOT/'results/phase_mvtec_exact_visa_recipe_checked/formal/source_assets/q_pc1_statealigned1024_formal.npy'
    ep=ROOT/'results/phase_mvtec_exact_visa_recipe_checked/formal/source_assets/E2_PC1Removed-CorruptedMask_formal.pt'
    p=np.load(record(pc1path)).astype(np.float64)
    adapter=loadcp(ep)['model']
    A=adapter['a.weight'].numpy().astype(np.float64); B=adapter['b.weight'].numpy().astype(np.float64)
    w=configs[3]['w']; numerator=w+A.T@(B.T@w); numerator-=p*(p@numerator)
    angle=float(np.degrees(np.arccos(np.clip(unit(configs[2]['w'])@unit(numerator),-1,1))))
    reference=pd.read_csv(record(ROOT/'results/phase_mvtec_exact_visa_recipe_checked/diagnostics_20261029/RAW_SPACE_DIRECTION_ANGLES.csv'))
    assert abs(angle-float(reference.loc[reference.comparison=='Control_vs_Recipe_initial','angle_degrees'].iloc[0]))<1e-6
    assert abs(metrics[2]['effective_weight_norm']-.362775078068223)<1e-7
    assert abs(metrics[3]['effective_weight_norm']-1.1322978136270152)<1e-7

    fig=plt.figure(figsize=(7.1,7.0))
    fig.subplots_adjust(left=.025,right=.89,bottom=.09,top=.90,wspace=.10,hspace=.18)
    axes=[]
    for row in (0,1):
        both=configs[2*row:2*row+2]
        allxyz=np.concatenate([np.concatenate([c['xyz'],c['markers']]) for c in both])
        lo=allxyz.min(0); hi=allxyz.max(0); span=hi-lo; lo-=span*.08;hi+=span*.08
        for col,c in enumerate(both):
            ax=fig.add_subplot(2,2,2*row+col+1,projection='3d');axes.append(ax)
            for label,color in ((0,BLUE),(1,ORANGE)):
                x=c['xyz'][c['y']==label]
                ax.scatter(*x.T,s=2.1,c=color,alpha=.22,edgecolors='none',depthshade=False)
            center,sn,sa=c['markers']
            for point,marker,color,size in ((center,'*','#222222',65),(sn,'o',TEAL,35),(sa,'o',RED,35)):
                ax.scatter(*point,s=size,marker=marker,c=color,edgecolors='white',linewidths=.5,depthshade=False,zorder=10)
            # Tail at s_N: the segment is centered on (s_N+s_A)/2, NOT on m.
            ax.quiver(*sn,*c['wp'],length=1,normalize=False,color='#272727',linewidth=1.4,arrow_length_ratio=.22)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=21,azim=-58)
            for axis,name in ((ax.xaxis,'PC1'),(ax.yaxis,'PC2'),(ax.zaxis,'PC3')):
                axis.set_major_locator(MaxNLocator(3))
                axis.pane.fill=False
                axis._axinfo['grid'].update(color=(.5,.5,.5,.12),linewidth=.4)
            ax.set_xlabel('PC1',labelpad=-1);ax.set_ylabel('PC2',labelpad=-1);ax.set_zlabel('PC3',labelpad=-2)
            ax.tick_params(pad=0)
            ax.set_title(f'({"abcd"[2*row+col]}) {c["title"]}',pad=2,fontweight='semibold')
    handles=[Line2D([],[],marker='o',ls='',color=BLUE,markersize=4,label='Normal patches'),
        Line2D([],[],marker='o',ls='',color=ORANGE,markersize=4,label='Anomalous patches'),
        Line2D([],[],marker='*',ls='',color='#222222',markersize=7,label='Construction center'),
        Line2D([],[],marker='o',ls='',color=TEAL,markersize=5,label='Normal state'),
        Line2D([],[],marker='o',ls='',color=RED,markersize=5,label='Anomaly state'),
        Line2D([],[],marker='>',color='#272727',markersize=4,label='Scoring direction')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.985),ncol=3,columnspacing=1.8,frameon=False)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'ces_initialization_3d.alignment.json',strict=True)
    # Fixed canvas preserves the requested 7.1in width and avoids mplot3d's
    # incomplete tight-bbox accounting for its projected z-axis labels.
    fig.savefig(OUT/'ces_initialization_3d.png',dpi=600)
    fig.savefig(OUT/'ces_initialization_3d.pdf')
    fig.savefig(OUT/'ces_initialization_3d.svg')
    fig.savefig(OUT/'preview.png',dpi=160);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(7.1,4.7))
    fig.subplots_adjust(left=.09,right=.98,bottom=.11,top=.92,wspace=.25,hspace=.5)
    for row in (0,1):
        group=configs[2*row:2*row+2]
        bins=np.linspace(min(c['scores'].min() for c in group),max(c['scores'].max() for c in group),51)
        for col,c in enumerate(group):
            ax=axs[row,col]
            for label,color,text in ((0,BLUE,'Normal'),(1,ORANGE,'Anomaly')):
                ax.hist(c['scores'][c['y']==label],bins=bins,density=True,histtype='step',lw=1.2,color=color,label=text)
            ax.axvline(0,color='.75',ls='--',lw=.6)
            ax.set(title=f'({"abcd"[2*row+col]}) {c["title"]}',xlabel='Anomaly score',ylabel='Density')
            ax.spines[['top','right']].set_visible(False)
            ax.set_xlim(bins[0],bins[-1]);ax.legend(frameon=False)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'score_distributions.alignment.json',strict=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'score_distributions.{ext}',dpi=600,bbox_inches='tight')
    plt.close(fig)

    record(Path(__file__))
    metadata=dict(seed=SEED,quota_per_class=N_PER_CLASS,input_files=inputs,score_checks=score_checks,
        pca=pca_records,mvtec_common_raw_numerator_angle_deg=angle,
        visa_coordinates_identical=True,mvtec_physical_ids_labels_identical=True,
        training=False,image_encoding=False,target_evaluation=False)
    (OUT/'provenance.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report=['# Source geometry visualization',
        '\n## Scope and panel identities',
        'The final user revision supersedes the original four-initialization proposal. '
        '(a,b) compare certified Full VisA CES initialization and update30 of the SAME '
        'BCE+Dice+NTR run, not trained no-NTR versus NTR. '
        '(c,d) compare MVTec nearest-mask matched-control initialization and full '
        'VisA-style preparation initialization. Neither MVTec panel uses its final trained direction.',
        '\n## Sampling and pairing',
        'Seed0, up to2500 patches per label per source, selected from actual saved CES '
        'training pools with equal category quotas where available and round-robin redistribution '
        'of shortages. No target images, scores or GT influence selection. Normal pools include '
        'normal regions of anomalous images. Source labels follow saved nearest37 IDs. '
        'All available per-category/class counts, sampled physical IDs and quotas are saved. '
        'VisA uses the certified fresh FP32 source build; MVTec uses the repaired physically '
        'matched experiment. This does not retrospectively certify other historical models.',
        '\n## Projection and arrow',
        'One joint full-SVD PCA per source row, fitted on both configurations\' sampled features; '
        'no feature standardization or independent per-panel PCA. Same limits, camera and box '
        'aspect within each row. VisA coordinates are bitwise identical: only the CES direction '
        'was trained; no patch features moved. The construction center is projected with PCA '
        'centering, whereas the difference vector is projected without centering. The arrow '
        'runs from the normalized normal state to the normalized anomaly state, hence is '
        'geometrically centered at their midpoint, which need not equal construction center m. '
        'No arrow/state enlargement is applied. Cross-row coordinates are not comparable.',
        '\n## Quantitative checks',pd.DataFrame(metrics).to_markdown(index=False),
        f'\nMVTec common-raw-space score-numerator angle = {angle:.10f} degrees, '
        'verified against the saved diagnostic. For P=I-pp^T and R=I+BA, compare control w '
        'against P R^T w_recipe. Recipe scoring also divides by the patch-dependent norm '
        '||RPz||, so this angle is not the angle between complete scoring functions.',
        '\n## Interpretation limits',
        'This is source-sample geometry, not target localization or causal attribution. '
        'VisA endpoint changes combine BCE, Dice and NTR training; they do not isolate NTR. '
        'MVTec changes the whole preparation pipeline, not just center or capacity. '
        'A larger effective norm alone cannot improve rank-based AP under uniform scaling. '
        'PCA variance and projected weight-energy retention in geometry_metrics.csv quantify '
        'what is missing in 3D; neither visible separation nor visible arrow length is a '
        'complete measure of scoring quality. The separately normalized, shared-bin 1D '
        'histograms display actual full-dimensional scores, not PCA-derived scores. '
        'Source AP/AUROC are exploratory sampled training-pool statistics, not held-out estimates.',
        '\n## Observations on this locked sample',
        f'VisA sampled-source AP changes from {metrics[0]["sample_patch_AP"]:.4f}% '
        f'to {metrics[1]["sample_patch_AP"]:.4f}%, while the anomaly-normal mean gap '
        f'changes from {metrics[0]["anomaly_normal_gap"]:.6f} to '
        f'{metrics[1]["anomaly_normal_gap"]:.6f}. Both class means decrease. '
        'Thus mean separation alone does not track ranking quality. '
        f'MVTec sampled-source AP changes from {metrics[2]["sample_patch_AP"]:.4f}% '
        f'to {metrics[3]["sample_patch_AP"]:.4f}% under preparation. This does not '
        'establish target improvement, and is compatible with the separate recorded '
        'VisA-target degradation of the complete trained preparation recipe.',
        '\n## Suggested caption',
        'Source-derived CES geometry. (a,b) The same prepared VisA source patches at CES '
        'initialization and after30 direction updates. (c,d) Matched MVTec source patches '
        'under direct initialization and VisA-style preparation. Each row uses a shared '
        'PCA basis, camera and limits; the two rows use different bases. Markers show the '
        'construction center and normalized states, and arrows show the projected effective '
        'scoring vector without magnification. Patch coordinates remain unchanged between (a,b).',
        '\n## Reproduction','Run this script with D:/Anaconda/python.exe. Input hashes, '
        'sample manifest, sampled features, PCA parameters and point coordinates are preserved. '
        'No original checkpoint, CSV, Word file or prediction cache is modified.']
    report.extend(['\n## Render QA',
        'Both multi-panel alignment checks pass. Final PDF text has a6.5pt minimum. '
        'The distribution PDF has no collision findings. The3D PDF has no clipping '
        'or text-text collisions; its16 text-fill-edge warnings correspond to axis '
        'ticks intersecting the bounding boxes of transparent3D panes, visually '
        'reviewed as intentional axes geometry. The initial clipped export was '
        'replaced with a fixed7.1in canvas and adequate right margin. '
        'The source-only heuristic validator does not recognize serif fonts or '
        'loop-based vector exports and flags preview160dpi as a submission raster; '
        'these do not describe the final vector files or600dpi PNG. Random choice '
        'is restricted to saved-pool indices, not generated feature values. '
        'This is seed0 only; no multi-seed uncertainty is claimed.'])
    # Candidate counts make limited anomaly availability explicit rather than hidden.
    counts=[]
    for source,labels,categories,selected in [('VisA',vy,cats,vi),
            ('MVTec',manifests[0].label.to_numpy(),manifests[0].category.to_numpy(),mi)]:
        for category in sorted(set(categories)):
            for label in (0,1):
                available=int(sum((labels==label)&(categories==category)))
                sampled=int(sum((labels[selected]==label)&(categories[selected]==category)))
                counts.append(dict(source=source,category=category,label=label,
                    available=available,sampled=sampled,fraction=sampled/available if available else None))
    pd.DataFrame(counts).to_csv(OUT/'sampling_counts.csv',index=False)
    (OUT/'visualization_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print('Raw numerator angle:',angle,'Output:',OUT,flush=True)


if __name__=='__main__':
    main()
