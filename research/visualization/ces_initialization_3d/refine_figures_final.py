"""Render the locked source-geometry arrays; never fit PCA or score a model.

Fig.5: qualitative geometry grid. Fig.6: separately normalized score histograms.
No training, sampling, KDE, coordinate adjustment or checkpoint loading occurs.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

OUT = Path(__file__).resolve().parent
QA = Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')
sys.path.insert(0,str(QA))
from audit_panel_alignment import require_matplotlib_panel_alignment

BLUE, ORANGE, TEAL, RED = '#4E79A7', '#CE8055', '#238C91', '#963F40'
TITLES = ('(a) VisA - Initialization','(b) VisA - After 30 Updates',
          '(c) MVTec - Matched Initialization','(d) MVTec - VisA-style Preparation')
ORIGINALS = ('ces_initialization_3d.py','sampled_features_geometry_pca.npz',
    'sample_manifest.csv','projected_patches_scores.csv','geometry_metrics.csv',
    'sampling_counts.csv','provenance.json','visualization_report.md',
    'ces_initialization_3d.png','ces_initialization_3d.pdf','ces_initialization_3d.svg',
    'score_distributions.png','score_distributions.pdf','score_distributions.svg')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def export(fig,stem):
    # Preserve the exact7.1in physical canvas rather than cropping its margins.
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')


def preview_pixels(fig):
    fig.set_dpi(160)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def audit(script,filename,output,extra=()):
    env=dict(os.environ,PYTHONIOENCODING='utf-8')
    result=subprocess.run([sys.executable,str(QA/script),str(OUT/filename),*extra],
        capture_output=True,text=True,encoding='utf-8',env=env)
    try:
        payload=json.loads(result.stdout)
    except json.JSONDecodeError:
        payload=dict(stdout=result.stdout,stderr=result.stderr)
    (OUT/output).write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return result.returncode,payload


def main():
    before={name:sha(OUT/name) for name in ORIGINALS}
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'axes.titlesize':9,'axes.titleweight':'semibold',
        'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none'})
    saved=np.load(OUT/'sampled_features_geometry_pca.npz')
    table=pd.read_csv(OUT/'projected_patches_scores.csv')
    manifest=pd.read_csv(OUT/'sample_manifest.csv')
    panels=[]
    checks=[]
    for i in range(4):
        prefix=f'panel_{i}_'
        c={key:saved[prefix+key] for key in ('xyz','markers','w','sn','sa','m','y','scores')}
        c['y']=c['y'].astype(int)
        source='VisA' if i<2 else 'MVTec'
        c['wp']=saved[f'{source}_pca_components']@c['w']
        assert np.allclose(c['markers'][2]-c['markers'][1],c['wp'],atol=1e-12)
        assert np.allclose(c['w'],c['sa']-c['sn'],atol=1e-12)
        assert np.allclose(np.linalg.norm([c['sn'],c['sa']],axis=1),1,atol=1e-12)
        rows=table[table.panel=='abcd'[i]]
        source_manifest=manifest[manifest.source==source]
        assert np.array_equal(rows.patch_id.to_numpy(),source_manifest.patch_id.to_numpy())
        assert np.array_equal(rows.label.to_numpy(),c['y'])
        assert np.allclose(rows[['PC1','PC2','PC3']].to_numpy(),c['xyz'],atol=1e-14)
        assert np.allclose(rows.q.to_numpy(),c['scores'],atol=1e-14)
        checks.append(dict(panel='abcd'[i],patches=len(c['y']),
            coordinate_csv_max_error=float(np.max(abs(rows[['PC1','PC2','PC3']].to_numpy()-c['xyz']))),
            score_csv_max_error=float(np.max(abs(rows.q.to_numpy()-c['scores']))),
            projected_arrow_error=float(np.max(abs(c['markers'][2]-c['markers'][1]-c['wp'])))))
        panels.append(c)
    assert np.array_equal(panels[0]['xyz'],panels[1]['xyz'])
    assert np.array_equal(panels[0]['y'],panels[1]['y'])
    assert np.array_equal(panels[2]['y'],panels[3]['y'])

    fig=plt.figure(figsize=(7.1,6.25))
    fig.subplots_adjust(left=.015,right=.985,bottom=.17,top=.98,wspace=.025,hspace=.18)
    axes=[];geometry_settings=[]
    for row in (0,1):
        pair=panels[row*2:row*2+2]
        points=np.concatenate([np.concatenate([c['xyz'],c['markers']]) for c in pair])
        lo=points.min(0);hi=points.max(0);span=hi-lo
        lo-=span*.08;hi+=span*.08
        for col,c in enumerate(pair):
            idx=row*2+col
            ax=fig.add_subplot(2,2,idx+1,projection='3d',computed_zorder=False)
            axes.append(ax)
            for label,color in ((0,BLUE),(1,ORANGE)):
                xyz=c['xyz'][c['y']==label]
                ax.scatter(*xyz.T,s=1.8,c=color,alpha=.17,edgecolors='none',
                           depthshade=False,zorder=1)
            center,sn,sa=c['markers']
            # The arrow length remains exactly the saved projected s_A - s_N.
            ax.quiver(*sn,*c['wp'],length=1,normalize=False,color='#303438',
                      linewidth=1.15,arrow_length_ratio=.17,zorder=5)
            for point,color in ((sn,TEAL),(sa,RED)):
                ax.scatter(*point,s=32,c=color,edgecolors='white',linewidths=.55,
                           depthshade=False,zorder=7)
            ax.scatter(*center,s=75,c='#202020',marker='*',edgecolors='white',
                       linewidths=.55,depthshade=False,zorder=8)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]))
            ax.set_box_aspect((1,1,1));ax.view_init(elev=21,azim=-58)
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.set_major_locator(MaxNLocator(3))
                axis.set_major_formatter(NullFormatter())
                axis.label.set_visible(False)
                axis.pane.fill=False
                axis.pane.set_edgecolor('#E2E2E2')
                axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.45)
                axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.4)
                axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
            ax.text2D(.5,-.035,TITLES[idx],transform=ax.transAxes,
                      ha='center',va='top',fontsize=9,fontweight='semibold')
            geometry_settings.append(dict(panel='abcd'[idx],limits=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()],
                elevation=ax.elev,azimuth=ax.azim,box_aspect=ax.get_box_aspect().tolist(),
                arrow_origin=sn.tolist(),arrow_vector=c['wp'].tolist()))
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=4,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=4,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=8,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=5,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=5,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=4,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),ncol=3,
        frameon=False,columnspacing=1.35,handletextpad=.55,handlelength=1.25,labelspacing=.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'ces_initialization_3d_final.alignment.json',strict=True)
    export(fig,'ces_initialization_3d_final')
    geometry_preview=preview_pixels(fig);plt.close(fig)

    fig,axs=plt.subplots(2,2,figsize=(7.1,4.9))
    fig.subplots_adjust(left=.085,right=.98,bottom=.23,top=.96,wspace=.235,hspace=.92)
    histogram_settings=[]
    for row in (0,1):
        pair=panels[row*2:row*2+2]
        # Exactly the original50 equal-width bins spanning both score arrays.
        bins=np.linspace(min(c['scores'].min() for c in pair),max(c['scores'].max() for c in pair),51)
        densities=[[np.histogram(c['scores'][c['y']==label],bins=bins,density=True)[0]
                    for label in (0,1)] for c in pair]
        ymax=max(float(d.max()) for group in densities for d in group)*1.10
        for col,c in enumerate(pair):
            idx=row*2+col;ax=axs[row,col]
            for label,color,density in zip((0,1),(BLUE,ORANGE),densities[col]):
                ax.stairs(density,bins,fill=True,color=color,alpha=.075,linewidth=0)
                ax.stairs(density,bins,color=color,linewidth=1.15)
                assert abs(float(np.sum(density*np.diff(bins)))-1)<1e-12
            ax.set(xlim=(bins[0],bins[-1]),ylim=(0,ymax),xlabel='Anomaly score',ylabel='Density')
            ax.text(.5,-.43,TITLES[idx],transform=ax.transAxes,
                    ha='center',va='top',fontsize=9,fontweight='semibold')
            ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.spines[['top','right']].set_visible(False)
            ax.spines[['left','bottom']].set_color('#707070')
            ax.tick_params(width=.55,length=3)
            histogram_settings.append(dict(panel='abcd'[idx],bin_edges=bins.tolist(),
                xlim=ax.get_xlim(),ylim=ax.get_ylim(),densities=[d.tolist() for d in densities[col]]))
    fig.legend(handles=[Line2D([],[],color=BLUE,lw=1.5,label='Normal'),
        Line2D([],[],color=ORANGE,lw=1.5,label='Anomaly')],loc='lower center',
        bbox_to_anchor=(.5,.018),ncol=2,columnspacing=2.2,frameon=False)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'score_distributions_final.alignment.json',strict=True)
    export(fig,'score_distributions_final')
    scores_preview=preview_pixels(fig);plt.close(fig)

    # A preview only: exported panels are unchanged and each has its own7.1in canvas.
    combined=plt.figure(figsize=(7.1,11.15))
    for image,rect in ((geometry_preview,[0,4.9/11.15,1,6.25/11.15]),
                       (scores_preview,[0,0,1,4.9/11.15])):
        ax=combined.add_axes(rect);ax.imshow(image);ax.set_axis_off()
    combined.savefig(OUT/'figures_comparison_preview.png',dpi=180)
    plt.close(combined)
    after={name:sha(OUT/name) for name in ORIGINALS}
    assert before==after,'An original artifact changed; stop delivery.'
    for settings in (geometry_settings,histogram_settings):
        for left,right in ((settings[0],settings[1]),(settings[2],settings[3])):
            if 'limits' in left:
                assert left['limits']==right['limits']
                assert left['elevation']==right['elevation'] and left['azimuth']==right['azimuth']
                assert left['box_aspect']==right['box_aspect']
            else:
                assert left['bin_edges']==right['bin_edges']
                assert left['xlim']==right['xlim'] and left['ylim']==right['ylim']
    render_record=dict(original_hashes_before=before,original_hashes_after=after,
        data_checks=checks,geometry=geometry_settings,histograms=histogram_settings,
        script_sha256=sha(__file__),training=False,sampling=False,pca_fit=False,
        model_scoring=False,originals_unchanged=True)
    (OUT/'final_render_identity.json').write_text(json.dumps(render_record,indent=2),encoding='utf-8')
    qa={}
    for stem in ('ces_initialization_3d_final','score_distributions_final'):
        code,font=audit('audit_pdf_text.py',stem+'.pdf',stem+'.font_audit.json',('--json','--min-pt','8'))
        assert code==0,(stem,'font',font)
        code,collisions=audit('audit_figure_collisions.py',stem+'.pdf',stem+'.collision_audit.json',('--json',))
        assert code==0 and collisions['summary']['fail']==0,(stem,collisions)
        qa[stem]=dict(font=font,collision_summary=collisions['summary'],verdict=collisions['verdict'])
    _,static=audit('validate_figure.py',Path(__file__).name,'final_source_preflight.json',('--backend','python','--json'))
    report=['# Final figure QA',
        '## Data integrity',
        '- PASS: all14 original scripts/data/report/figure files have identical before/after SHA256. '
        'See final_render_identity.json for exact hashes.',
        '- PASS: loaded the existing saved3D coordinates, states, centers and full-dimensional scores. '
        'No training, sampling, PCA fitting, model forward pass or score calibration.',
        '- PASS: patch IDs, labels, coordinates and scores agree with the original CSV. '
        'The VisA clouds remain bitwise identical. MVTec labels and physical IDs are matched.',
        '- PASS: arrow origins are the saved normal states; vectors equal the projected '
        'anomaly-minus-normal states, with length=1 and normalize=False. No point was moved.',
        '## Rendering changes',
        '- Geometry: removed all coordinate labels/numerals, retained faint3D frames/grids; '
        'smaller/lighter clouds; foreground states and white-edged construction centers. '
        'Original per-row limits, camera21/-58 degrees and equal box aspect are retained.',
        '- Scores: the original50 shared row bins and class-wise density normalization '
        'are retained exactly. Added only7.5% fill; no KDE/smoothing. Row y limits are '
        'now shared; no reference lines or quantitative annotations remain.',
        '- Both figures: Times New Roman serif, all text at least8pt, matching panel titles, '
        'panel labels and titles beneath each subplot, one shared bottom legend, '
        'exact7.1in canvas,600dpi PNG and vector PDF/editable SVG.',
        '## Automated checks',json.dumps(qa,indent=2),
        'Both rendered plot-area alignment gates pass. Source-only heuristic preflight '
        'is retained separately; it may flag the explicitly requested serif font or '
        'the180dpi comparison preview, neither of which changes the final600dpi exports.',
        '## Scientific scope',
        '(a,b) are initialization/update30 of one VisA BCE+Dice+NTR run, not an NTR-only '
        'ablation. Features are frozen. (c,d) are MVTec matched and VisA-style '
        'initializations, not before/after CES training. Each row shares one existing '
        'PCA basis; cross-row coordinates are not comparable. No 3D visual claim '
        'substitutes for full-dimensional ranking or target evaluation.',
        '## Captions for external typesetting',
        '**Fig.5. Source-derived CES geometry.** (a,b) VisA-source initialization and '
        'optimization after30 updates. (c,d) MVTec-source initialization with matched '
        'and VisA-style preparation. Each row uses a shared PCA basis.',
        '**Fig.6. Source patch anomaly-score distributions.** Normal and anomalous '
        'source patch scores under the four configurations in Fig.5.',
        'No Word document or formal experimental result was modified.']
    (OUT/'figure_qa_report.md').write_text('\n\n'.join(report),encoding='utf-8')
    print(json.dumps(dict(outputs=str(OUT),originals_unchanged=True,qa=qa),indent=2),flush=True)


if __name__=='__main__':
    main()
