import os
import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# This figure asks whether a participant's Integrated iHH (the sweep signal
# measured at their closest post-nadir timepoint) relates to their Day 0
# viral load or Day 0 sequence diversity. It formalizes two of the panels
# from the 07-09-2026_iihh_vs_baseline exploratory analysis into a single
# supplemental figure.

###############################################################################
# Plotting style
###############################################################################
params = {'figure.figsize': (3.5, 1.75), 'axes.labelsize': 6, 'axes.titlesize': 6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)

par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD10K': '#6A3D9A',
               '1HD11K': '#B15928'}
PAR_LIST_1074 = list(par_palette.keys())

###############################################################################
# Paths
###############################################################################
area_file = '../../../../results/paper_draft_04-13/figure_5/inv_area_df.csv'
comp_time_file = '../../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'
vl_file = '../../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
pi_file = '../../../../results/paper_draft_04-13/figure_2/inv_pi_summary.csv'
out_dir = '../../../../results/paper_draft_04-13/supp_figs/iihh_vs_baseline/'

os.makedirs(out_dir, exist_ok=True)

###############################################################################
# Data prep
###############################################################################
# Get each participant's Integrated iHH at their own closest post-nadir
# timepoint (same selection used in paper_draft_04-13/figure_5/summary_ihh_one_panel.py)
area_df = pd.read_csv(area_file)
area_df = area_df[(area_df['dataset'] == '10-1074') &
                   (area_df['participant'].isin(PAR_LIST_1074))]

comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))
area_df['closest_post_nadir'] = area_df['participant'].map(
    lambda x: 'W' + str(int(nadir_time_dict[x])))
iihh_df = area_df[area_df['time_label'] == area_df['closest_post_nadir']]
iihh_df = iihh_df[['participant', 'area']]

# Day 0 viral load
vl_df = pd.read_csv(vl_file)
vl_df = vl_df.rename(columns={'Study_ID': 'participant', 'D0': 'viral_load'})
vl_df = vl_df[vl_df['participant'].isin(PAR_LIST_1074)][['participant', 'viral_load']]

# Day 0 diversity, averaged across the full sequence
pi_df = pd.read_csv(pi_file)
pi_df = pi_df[pi_df['time'] == 'D0']
pi_df = pi_df.rename(columns={'pi': 'diversity'})
pi_df = pi_df[pi_df['participant'].isin(PAR_LIST_1074)][['participant', 'diversity']]

print('average day 0 diversity:', pi_df['diversity'].mean(), file=sys.stderr)

vl_merged = iihh_df.merge(vl_df, on='participant')
pi_merged = iihh_df.merge(pi_df, on='participant')


###############################################################################
# Plot
###############################################################################
def scatter_by_participant(ax, df, x_col, x_label, y_col='area', y_label='Integrated iHH'):
    for participant, row in df.set_index('participant').iterrows():
        ax.scatter(row[x_col], row[y_col], color=par_palette[participant],
                   label=participant, edgecolor='black', linewidth=0.5, s=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


fig, axs = plt.subplots(1, 2, sharey=True)

scatter_by_participant(axs[0], vl_merged, 'viral_load', 'Day 0 viral load\n(copies/mL)')
scatter_by_participant(axs[1], pi_merged, 'diversity', 'Day 0 diversity (π)')

axs[1].set_ylabel('')  # Remove y-axis label from the second panel
axs[0].set_xscale('log')
axs[0].set_xlim(5e3, 1e5)

for curr_ax in axs:
    curr_ax.tick_params(axis='x', rotation=0)
    for tick_label in curr_ax.get_xticklabels():
        tick_label.set_horizontalalignment('center')

# One x label spans two lines ("Day 0 viral load" / "(copies/mL)") and the
# other is a single line, so by default their top rows don't line up.
# align_xlabels pins both labels to the same vertical position so the first
# row of text matches across panels regardless of line count.
fig.align_xlabels(axs)

handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles, labels, title='Participant', loc='upper left',
           bbox_to_anchor=(1.0, 1.05), frameon=False)
plt.subplots_adjust(left=0.1, right=0.8, bottom = 0.3, top = 0.95)

out_path = out_dir + 'iihh_vs_baseline.png'
fig.savefig(out_path, dpi=300)
print('Saved figure to', out_path)
