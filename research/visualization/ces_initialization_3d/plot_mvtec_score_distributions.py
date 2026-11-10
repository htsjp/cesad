"""Extract the two saved MVTec score panels; no rescoring or resampling."""
import json
import os
import subprocess
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from refine_figures_final import OUT as BASE, BLUE, ORANGE, QA, sha
from audit_panel_alignment import require_matplotlib_panel_alignment

OUT=BASE/'mvtec_score_distributions'


def main():
    OUT.mkdir(exist_ok=True)
    source=BASE/'final_render_identity.json'
    protected={p:sha(p) for p in (source,BASE/'score_distributions_final.pdf',
        BASE/'sampled_features_geometry_pca.npz')}
    record=json.loads(source.read_text(encoding='utf-8'))
    histograms=record['histograms'][2:4]
    assert [r['panel'] for r in histograms]==['c','d']
    assert histograms[0]['bin_edges']==histograms[1]['bin_edges']
    assert histograms[0]['xlim']==histograms[1]['xlim']
    assert histograms[0]['ylim']==histograms[1]['ylim']
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':8,'axes.labelsize':8.5,'legend.fontsize':8,
        'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.6})
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8))
    fig.subplots_adjust(left=.075,right=.98,bottom=.33,top=.97,wspace=.24)
    data=np.load(BASE/'sampled_features_geometry_pca.npz')
    titles=('(a) MVTec - Matched Initialization','(b) MVTec - VisA-style Preparation')
    for i,(ax,hist) in enumerate(zip(axes,histograms)):
        bins=np.array(hist['bin_edges'])
        for label,color,density in zip((0,1),(BLUE,ORANGE),hist['densities']):
            saved=np.array(density)
            q=data[f'panel_{i+2}_scores'];y=data[f'panel_{i+2}_y']
            check=np.histogram(q[y==label],bins=bins,density=True)[0]
            assert np.array_equal(check,saved)
            ax.stairs(saved,bins,fill=True,color=color,alpha=.075,linewidth=0)
            ax.stairs(saved,bins,color=color,linewidth=1.15)
        ax.set(xlim=hist['xlim'],ylim=hist['ylim'],xlabel='Anomaly score',ylabel='Density')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#707070')
        ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(width=.55,length=3)
        ax.text(.5,-.30,titles[i],transform=ax.transAxes,ha='center',va='top',
                fontsize=8,fontweight='semibold')
    fig.legend(handles=[Line2D([],[],color=BLUE,lw=1.5,label='Normal'),
        Line2D([],[],color=ORANGE,lw=1.5,label='Anomaly')],loc='lower center',
        bbox_to_anchor=(.5,.012),ncol=2,columnspacing=2.2,frameon=False)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',strict=True)
    stem='score_distributions_mvtec'
    fig.savefig(OUT/f'{stem}.png',dpi=600)
    fig.savefig(OUT/f'{stem}.pdf')
    fig.savefig(OUT/f'{stem}.svg')
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)
    for p,digest in protected.items():assert sha(p)==digest
    audits={}
    for script,label,extra in [('audit_pdf_text.py','font',('--json','--min-pt','8')),
                               ('audit_figure_collisions.py','collision',('--json',))]:
        r=subprocess.run([sys.executable,str(QA/script),str(OUT/f'{stem}.pdf'),*extra],
            capture_output=True,text=True,encoding='utf-8',env=dict(os.environ,PYTHONIOENCODING='utf-8'))
        audits[label]=json.loads(r.stdout)
        assert r.returncode==0,audits[label]
    (OUT/'qa.json').write_text(json.dumps(dict(originals_unchanged=True,
        saved_densities_bins_and_limits_unchanged=True,audits=audits),indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':
    main()
