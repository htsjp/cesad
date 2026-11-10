"""Observation-to-scorer progression; no artificial feature contraction.

Only visual salience changes in the prepared background; all coordinates and
states come from the fixed-basis saved artifacts. The raw panel has no states.
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

DATA=BASE/'visa_three_configurations_fixed_basis'
OUT=BASE/'visa_compression_progression'


def main():
    OUT.mkdir(exist_ok=True)
    paths=[DATA/'saved_geometry_and_projection.npz',DATA/'sample_manifest.csv',
           BASE/'final_render_identity.json']
    before={str(p):sha(p) for p in paths}
    a=np.load(paths[0]);sample=pd.read_csv(paths[1]);y=sample.label.to_numpy()
    settings=json.loads(paths[2].read_text(encoding='utf-8'))['geometry'][0]
    assert np.array_equal(a['panel_1_xyz'],a['panel_2_xyz'])
    assert np.array_equal(a['panel_1_m'],a['panel_2_m'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,3.25))
    fig.subplots_adjust(left=.005,right=.995,bottom=.34,top=.99,wspace=.005)
    titles=('(a) Source observations','(b) CES initialization','(c) CES direction update')
    for i in range(3):
        ax=fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False)
        xyz=a[f'panel_{i}_xyz']
        if i==0:
            for label,color in ((0,BLUE),(1,ORANGE)):
                points=xyz[y==label]
                ax.scatter(*points.T,s=1.5,color=color,alpha=.36,edgecolors='none',depthshade=False,zorder=1)
        else:
            # Desaturate observations; do not recolor into synthetic clusters.
            ax.scatter(*xyz.T,s=.8,color='#999999',alpha=.065,edgecolors='none',depthshade=False,zorder=1)
            center,sn,sa=a[f'panel_{i}_markers'];w=a[f'panel_{i}_wp']
            assert np.allclose(sa-sn,w,atol=1e-12)
            if i==2:
                _,old_n,old_a=a['panel_1_markers']
                ax.plot(*np.stack([old_n,old_a]).T,color='#A0A0A0',lw=.85,ls='--',zorder=3)
                ax.quiver(*old_n,*a['panel_1_wp'],length=1,normalize=False,
                          color='#A0A0A0',linewidth=.65,linestyles='--',arrow_length_ratio=.17,zorder=3)
            ax.quiver(*sn,*w,length=1,normalize=False,color='#303438',linewidth=1.25,
                      arrow_length_ratio=.17,zorder=5)
            for point,color in ((sn,TEAL),(sa,RED)):
                ax.scatter(*point,s=33,color=color,edgecolors='white',linewidths=.55,depthshade=False,zorder=7)
            ax.scatter(*center,s=62,color='#202020',marker='*',edgecolors='white',linewidths=.55,depthshade=False,zorder=8)
        ax.set(xlim=settings['limits'][0],ylim=settings['limits'][1],zlim=settings['limits'][2])
        ax.set_box_aspect((1,1,1));ax.view_init(elev=settings['elevation'],azim=settings['azimuth'])
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False;axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.10),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.02,titles[i],transform=ax.transAxes,ha='center',va='top',fontsize=8.5,fontweight='semibold')
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=4,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=4,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=7,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4.5,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4.5,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=4,label='Scoring direction'),
        Line2D([],[],color='#A0A0A0',ls='--',lw=.85,label='Initial direction (reference)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,
        frameon=False,columnspacing=1.2,handletextpad=.5,handlelength=1.2,labelspacing=.5)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',strict=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'visa_compression_progression.{ext}',dpi=600)
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)

    w0=a['panel_1_w'];wf=a['panel_2_w']
    angle=float(np.degrees(np.arccos(np.clip(w0@wf/(np.linalg.norm(w0)*np.linalg.norm(wf)),-1,1))))
    metrics=dict(source='Full VisA',normal_patches=int(sum(y==0)),anomaly_patches=int(sum(y==1)),
        initial_weight_norm=float(np.linalg.norm(w0)),updated_weight_norm=float(np.linalg.norm(wf)),
        within_prepared_space_weight_angle_deg=angle,
        initial_projected_weight_norm=float(np.linalg.norm(a['panel_1_wp'])),
        updated_projected_weight_norm=float(np.linalg.norm(a['panel_2_wp'])),
        direction_updated=True,prepared_patch_coordinates_bitwise_identical=True,
        note='Effective weight norm, not individual state norm. All normalized states have norm1; m/alpha frozen.')
    (OUT/'geometry_change.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    assert all(sha(Path(p))==h for p,h in before.items())
    identity=dict(inputs=before,prior_files_unchanged=True,training=False,resampling=False,
                  pca_refit=False,coordinates_changed=False,
                  panel_a='existing raw-L2 feature projection; no direct-init states displayed',
                  panel_b='existing prepared initialization states, desaturated feature silhouette',
                  panel_c='existing update30 states, identical prepared feature silhouette, initial vector reference')
    (OUT/'identity.json').write_text(json.dumps(identity,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(
        '# Source observations, compact CES initialization, direction update\n\n'
        'This replaces the three-configuration comparison with a presentation of retained '
        'source supervision. Panel(a) emphasizes5000 real raw-L2 CLIP patch observations. '
        'Panels(b,c) fade prepared patch observations into an unchanged grey silhouette '
        'and emphasize the real compact scorer states. Fading is a visual hierarchy, '
        'NOT deletion, contraction, clustering, movement or PCA compression by CES.\n\n'
        'Panel(b) uses the saved prepared source initialization. Panel(c) uses the '
        'same run after30 BCE+Dice+NTR updates; the grey dashed vector is the actual '
        'initial vector at its actual state endpoints. It is not a third learned '
        'state or a new parameter. Both scoring arrows have exact projected lengths.\n\n'
        'Raw versus prepared features also differ by the fixed PC1-removal/adapter '
        'pipeline. Therefore(a)->(b) is not a CES-only feature transformation. The '
        'compression claim refers to the two retained scoring states, not to the '
        'encoder, PC1/adapter parameters or geometric contraction of the cloud.\n\n'
        'All panels retain the original prepared-VisA PCA, camera and limits. The '
        'raw cloud is projected out-of-fit; prepared panels reuse exact coordinates. '
        'No samples, scores or states were adjusted for visual separation.\n\n'
        f'Full-dimensional effective scoring norm: {metrics["initial_weight_norm"]:.8f} '
        f'-> {metrics["updated_weight_norm"]:.8f}. Prepared-space weight angle: '
        f'{angle:.8f} degrees. These values are not printed inside the figure. '
        'Only the direction parameter is optimized; m and alpha are frozen. The '
        'effective norm changes because the constructed states are normalized. '
        'Each state itself remains unit norm.3D projected arrow lengths do not '
        'equal full-dimensional norms. This is not an NTR-only ablation.\n\n'
        'Suggested caption: Source-experience representation and CES optimization. '
        '(a) Frozen CLIP source-patch observations. (b) Source-derived normalized CES '
        'states, with prepared source observations shown as a faint reference. '
        '(c) The same scorer after30 direction updates; the dashed vector indicates '
        'initialization. All panels use a fixed PCA basis. Compression denotes '
        'retaining source supervision in two states, rather than moving patch features.\n',encoding='utf-8')
    print(json.dumps(metrics,indent=2));print('Output:',OUT)


if __name__=='__main__':
    main()
