"""Compact1x4 variant, using only saved coordinates; all prior files retained."""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

from refine_figures_final import OUT, BLUE, ORANGE, TEAL, RED, sha, audit
from audit_panel_alignment import require_matplotlib_panel_alignment


def main():
    existing={p.name:sha(p) for p in OUT.iterdir() if p.is_file() and 'one_row' not in p.name}
    arrays=np.load(OUT/'sampled_features_geometry_pca.npz')
    prior=json.loads((OUT/'final_render_identity.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'legend.fontsize':8,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,2.55))
    fig.subplots_adjust(left=.005,right=.995,bottom=.34,top=.99,wspace=.005)
    titles=('(a) VisA\nInitialization','(b) VisA\nAfter 30 Updates',
            '(c) MVTec\nMatched Initialization','(d) MVTec\nVisA-style Preparation')
    for i in range(4):
        ax=fig.add_subplot(1,4,i+1,projection='3d',computed_zorder=False)
        xyz=arrays[f'panel_{i}_xyz'];y=arrays[f'panel_{i}_y']
        center,sn,sa=arrays[f'panel_{i}_markers']
        vector=np.array(prior['geometry'][i]['arrow_vector'])
        assert np.allclose(vector,sa-sn,atol=1e-12)
        for label,color in ((0,BLUE),(1,ORANGE)):
            points=xyz[y==label]
            ax.scatter(*points.T,s=.85,color=color,alpha=.17,edgecolors='none',
                       depthshade=False,zorder=1)
        ax.quiver(*sn,*vector,length=1,normalize=False,color='#303438',
                  linewidth=.9,arrow_length_ratio=.17,zorder=5)
        for point,color in ((sn,TEAL),(sa,RED)):
            ax.scatter(*point,s=22,color=color,edgecolors='white',linewidths=.4,
                       depthshade=False,zorder=7)
        ax.scatter(*center,s=45,color='#202020',marker='*',edgecolors='white',
                   linewidths=.4,depthshade=False,zorder=8)
        setting=prior['geometry'][i]
        ax.set(xlim=setting['limits'][0],ylim=setting['limits'][1],zlim=setting['limits'][2])
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=setting['elevation'],azim=setting['azimuth'])
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False
            axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.015,titles[i],transform=ax.transAxes,ha='center',va='top',
                  fontsize=8,fontweight='semibold',linespacing=1.25)
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=4,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=4,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=7,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4.5,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4.5,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=4,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3,
        frameon=False,columnspacing=1.5,handletextpad=.55,handlelength=1.25,labelspacing=.55)
    fig.canvas.draw()
    stem='ces_initialization_3d_one_row'
    require_matplotlib_panel_alignment(fig,json_out=OUT/f'{stem}.alignment.json',strict=True)
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/f'{stem}_preview.png',dpi=180)
    plt.close(fig)
    results={}
    for script,suffix,extra in [('audit_pdf_text.py','font',('--json','--min-pt','8')),
                              ('audit_figure_collisions.py','collision',('--json',))]:
        code,record=audit(script,stem+'.pdf',f'{stem}.{suffix}.json',extra)
        assert code==0,(script,record)
        results[suffix]=record
    assert all(sha(OUT/name)==value for name,value in existing.items())
    results['all_previous_files_unchanged']=True
    results['coordinates_and_arrow_vectors']='Unmodified saved arrays; inherited limits and camera.'
    results['layout']='7.1in x2.55in,1x4; two-line subtitles, bottom shared legend.'
    (OUT/f'{stem}.qa.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
