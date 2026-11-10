"""Accept the expanded 48-case delivery using the same raw-map and PDF checks."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.verify_supplementary_qualitative_20261102 import main

if __name__=='__main__':
    main(out=ROOT/'results/supplementary_qualitative_expanded_20261102',
         expected_cases=48,
         figure_names=('supp_fig_s1_part1','supp_fig_s1_part2','supp_fig_s1_part3',
                       'supp_fig_s2a_ntr_8cases','supp_fig_s2b_lrp_8cases',
                       'supp_fig_s3_visa_boundary_8cases'),
         plotting_script=ROOT/'experiments/plot_supplementary_qualitative_expanded_20261102.py')
