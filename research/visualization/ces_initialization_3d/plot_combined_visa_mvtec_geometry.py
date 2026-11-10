"""Display-only3+2 source geometry plate; no PCA/scoring or histogram rendering."""
from pathlib import Path
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

from refine_figures_final import OUT as BASE, BLUE, ORANGE, TEAL, RED, sha
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT=BASE/'combined_visa_mvtec_geometry'
QA=Path('C:/Users/Administrator/.codex/skills/nature-figure/scripts')


def main():
    OUT.mkdir(exist_ok=True)
    protected={str(p):sha(p) for p in BASE.rglob('*')
               if p.is_file() and OUT not in p.parents and '__pycache__' not in p.parts}
    vp=BASE/'visa_ces_three_stage_final/visa_ces_three_stage_data.npz'
    mp=BASE/'sampled_features_geometry_pca.npz'
    visa=np.load(vp);mvtec=np.load(mp)
    vi=json.loads((BASE/'visa_ces_three_stage_final/validation.json').read_text(encoding='utf-8'))
    mi=json.loads((BASE/'final_render_identity.json').read_text(encoding='utf-8'))
    configs=[]
    for i,name in enumerate(('Direct Initialization','Prepared Initialization','After 30 Updates')):
        configs.append(dict(source='VisA',name=name,xyz=visa[f'panel_{i}_xyz'],
            y=visa['labels'],markers=visa[f'panel_{i}_markers'],wp=visa[f'panel_{i}_wp'],
            limits=vi['camera']['limits']))
    for i,name in ((2,'Matched Initialization'),(3,'VisA-style Preparation')):
        configs.append(dict(source='MVTec',name=name,xyz=mvtec[f'panel_{i}_xyz'],
            y=mvtec[f'panel_{i}_y'],markers=mvtec[f'panel_{i}_markers'],
            wp=np.array(mi['geometry'][i]['arrow_vector']),limits=mi['geometry'][i]['limits']))
    assert np.array_equal(configs[1]['xyz'],configs[2]['xyz'])
    assert all(np.allclose(c['wp'],c['markers'][2]-c['markers'][1],atol=1e-7) for c in configs)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,5.6))
    width=.326;step=.332
    height=width*7.1/5.6
    positions=[(.005+i*step,.98-height,width,height) for i in range(3)]
    left=(1-(2*width+(step-width)))/2
    positions.extend([(left+i*step,.102,width,height) for i in range(2)])
    for i,(c,rect) in enumerate(zip(configs,positions)):
        ax=fig.add_axes(rect,projection='3d',computed_zorder=False)
        for label,color in ((0,BLUE),(1,ORANGE)):
            points=c['xyz'][c['y']==label]
            ax.scatter(*points.T,s=1.1,color=color,alpha=.17,edgecolors='none',depthshade=False,zorder=1)
        center,sn,sa=c['markers']
        ax.quiver(*sn,*c['wp'],length=1,normalize=False,color='#303438',linewidth=1,
                  arrow_length_ratio=.17,zorder=5)
        for point,color in ((sn,TEAL),(sa,RED)):
            ax.scatter(*point,s=25,color=color,edgecolors='white',linewidths=.4,depthshade=False,zorder=7)
        ax.scatter(*center,s=52,color='#202020',marker='*',edgecolors='white',linewidths=.4,depthshade=False,zorder=8)
        limits=np.array(c['limits'])
        ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2])
        ax.set_box_aspect((1,1,1),zoom=1.10);ax.view_init(elev=21,azim=-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False;axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.055,f'({"abcde"[i]}) {c["source"]} - {c["name"]}',
            transform=ax.transAxes,ha='center',va='top',fontsize=7.5,fontweight='semibold')
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=3.5,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=3.5,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=6,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=3.5,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.018),ncol=6,
        frameon=False,columnspacing=.85,handletextpad=.35,handlelength=1)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',
        panel_ids=list('abcde'),row_groups=[['a','b','c'],['d','e']],strict=True)
    stem='visa_mvtec_geometry_combined'
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)
    assert all(sha(path)==digest for path,digest in protected.items())
    audits={}
    for script,label,extra in [('audit_pdf_text.py','font',('--json','--min-pt','7.5')),
                               ('audit_figure_collisions.py','collision',('--json',))]:
        result=subprocess.run([sys.executable,str(QA/script),str(OUT/f'{stem}.pdf'),*extra],
            capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        payload=json.loads(result.stdout)
        (OUT/f'{label}_audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
        assert result.returncode==0,payload
        audits[label]=payload
    evidence=dict(input_files=[dict(path=str(p),sha256=sha(p)) for p in (vp,mp)],
        protected_previous_files=len(protected),old_files_unchanged=True,
        layout='Top3 VisA, bottom2 MVTec centered, identical panel sizes; single shared bottom legend',
        dimensions_inches=[7.1,5.6],coordinates_unchanged=True,pca_fit=False,
        score_distributions=False,training=False,camera=[21,-58],uniform_zoom=1.1,
        per_panel_limits=[c['limits'] for c in configs],audits=audits)
    (OUT/'render_identity.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(
        '# Combined source geometry\n\n'
        'Top: VisA direct initialization, prepared initialization, update30. '
        'These use the latest joint raw/prepared PCA. Bottom: saved MVTec '
        'matched-control initialization and VisA-style preparation initialization, '
        'using their original shared MVTec PCA. PCA axes cannot be compared '
        'between source rows. All patches, markers and arrows are unchanged. '
        'No score-distribution panels are included. No original figure was overwritten.\n',encoding='utf-8')
    print(OUT)
    print('Font and collision checks passed; original files unchanged.')


if __name__=='__main__':
    main()
