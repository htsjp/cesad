"""Display-only larger panels and single-line bottom annotations; no PCA fitting."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, NullFormatter

from refine_figures_final import OUT as BASE, BLUE, ORANGE, TEAL, RED, sha
from audit_panel_alignment import require_matplotlib_panel_alignment

SOURCE=BASE/'visa_three_configurations_fixed_basis'
OUT=SOURCE/'compact_single_line'


def main():
    OUT.mkdir(exist_ok=True)
    originals={p.name:sha(p) for p in SOURCE.iterdir() if p.is_file()}
    data=np.load(SOURCE/'saved_geometry_and_projection.npz')
    prior=json.loads((BASE/'final_render_identity.json').read_text(encoding='utf-8'))
    limits=np.array(prior['geometry'][0]['limits'])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(7.1,2.7))
    fig.subplots_adjust(left=.005,right=.995,bottom=.19,top=.99,wspace=.005)
    titles=('(a) VisA - Direct Initialization','(b) VisA - Prepared Initialization',
            '(c) VisA - After 30 Updates')
    for i in range(3):
        ax=fig.add_subplot(1,3,i+1,projection='3d',computed_zorder=False)
        xyz=data[f'panel_{i}_xyz']
        center,sn,sa=data[f'panel_{i}_markers']
        wp=data[f'panel_{i}_wp']
        assert np.allclose(wp,sa-sn,atol=1e-12)
        # The locked label sequence is common to every panel.
        import pandas as pd
        y=pd.read_csv(SOURCE/'sample_manifest.csv').label.to_numpy()
        for label,color in ((0,BLUE),(1,ORANGE)):
            points=xyz[y==label]
            ax.scatter(*points.T,s=1.1,color=color,alpha=.17,edgecolors='none',depthshade=False,zorder=1)
        ax.quiver(*sn,*wp,length=1,normalize=False,color='#303438',linewidth=1,
                  arrow_length_ratio=.17,zorder=5)
        for point,color in ((sn,TEAL),(sa,RED)):
            ax.scatter(*point,s=25,color=color,edgecolors='white',linewidths=.4,depthshade=False,zorder=7)
        ax.scatter(*center,s=52,color='#202020',marker='*',edgecolors='white',linewidths=.4,depthshade=False,zorder=8)
        ax.set(xlim=limits[0],ylim=limits[1],zlim=limits[2])
        # Uniform camera zoom enlarges the entire frame, never individual states.
        ax.set_box_aspect((1,1,1),zoom=1.12);ax.view_init(elev=21,azim=-58)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3));axis.set_major_formatter(NullFormatter())
            axis.label.set_visible(False);axis.pane.fill=False;axis.pane.set_edgecolor('#E2E2E2')
            axis.line.set_color('#CBCBCB');axis.line.set_linewidth(.4)
            axis._axinfo['grid'].update(color=(.65,.65,.65,.13),linewidth=.35)
            axis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
        ax.text2D(.5,-.055,titles[i],transform=ax.transAxes,ha='center',va='top',
                  fontsize=7.5,fontweight='semibold')
    handles=[Line2D([],[],ls='',marker='o',color=BLUE,markersize=3.5,label='Normal patches'),
        Line2D([],[],ls='',marker='o',color=ORANGE,markersize=3.5,label='Anomalous patches'),
        Line2D([],[],ls='',marker='*',color='#202020',markersize=6,label='Construction center'),
        Line2D([],[],ls='',marker='o',color=TEAL,markersize=4,label='Normal state'),
        Line2D([],[],ls='',marker='o',color=RED,markersize=4,label='Anomaly state'),
        Line2D([],[],color='#303438',marker='>',markersize=3.5,label='Scoring direction')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.008),ncol=6,
        frameon=False,columnspacing=.85,handletextpad=.35,handlelength=1)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'alignment.json',strict=True)
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT/f'visa_three_configurations_compact.{ext}',dpi=600)
    fig.savefig(OUT/'preview.png',dpi=180);plt.close(fig)
    assert all(sha(SOURCE/name)==identity for name,identity in originals.items())
    (OUT/'render_identity.json').write_text(json.dumps(dict(
        input_sha256=originals,source_files_unchanged=True,coordinates_unchanged=True,
        axis_limits_unchanged=True,view=[21,-58],uniform_zoom=1.12,
        title_font_pt=7.5,legend_columns=6,figure_inches=[7.1,2.7]),indent=2),encoding='utf-8')
    print(OUT)


if __name__=='__main__':
    main()
