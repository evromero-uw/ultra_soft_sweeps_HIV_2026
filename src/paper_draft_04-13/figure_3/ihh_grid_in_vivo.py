import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic

from par_data_class import Pardata

# Line plots showing standardized iHH increases over time in vivo:
# one panel per participant plus a single summary panel.

###############################################################################
# Plotting style
###############################################################################

params = {'figure.figsize': (7, 3), 'axes.labelsize': 6, 'axes.titlesize':6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
fontsize = 6
linewidth = 0.5

#Commented out slide size parameters for now, but keeping them in case we want to use them later
# params = {'figure.figsize': (14, 6), 'axes.labelsize': 14, 'axes.titlesize': 14,
#           'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
#           'legend.title_fontsize': 14, 'font.family': 'sans-serif',
#           'font.sans-serif': 'Arial'}
# linewidth = 1
# fontsize = 14
rcParams.update(params)
###############################################################################
# Paths
###############################################################################
inVivo_inDir = '../../../results/paper_draft_04-13/02-19-2026_EHH_10-1074/'
inVivo_inDir = '../../../results/02-19-2026_exploratory/02-19-2026_EHH_10-1074/'
sequence_path = '../../../data/clyde_westfall_2024_final/10-1074/'
out_dir = '../../../results/paper_draft_04-13/figure_3/'
comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'

###############################################################################
# Analysis parameters
###############################################################################
PAR_LIST = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K',
            '1HD7K', '1HD10K', '1HD11K']
ALLELE_FREQ_THRESH = 0
MULTI_SEG = True
FREQ_BIN = 0.2
BIN_NUMBER = 100
TIMES_TO_KEEP = set(['D0', 'W4', 'W8'])

GENOME_MAX = 2500
YLIM = (-3, 20)

INV_PALETTE = sns.color_palette('rocket', n_colors=3)
INV_PALETTE_DICT = {x: INV_PALETTE[i] for i, x in enumerate(['D0', 'W4', 'W8'])}

# Summary panels: Day 0 is red (dashed), post-nadir is light gray (solid)
SUMMARY_COLOR = {'Day 0': 'black', 'Post-nadir': 'lightseagreen'}
SUMMARY_DASHES = {'Day 0': '', 'Post-nadir': ''}


###############################################################################
# Helpers
###############################################################################
def standardize_ihh(iHH_df):
    """Standardize iHH against the Day-0 distribution within SNP-frequency bins.

    For each frequency bin (FREQ_BIN-wide over [0, 1]) we compute the Day-0
    mean and std of iHH, then z-score every row by the bin it falls into.
    Returns the df with an added 'adj_iHH' column.
    """
    iHH_df_day0 = iHH_df[iHH_df['time_label'] == 'D0']

    binned_means, bin_edges, binnumber = binned_statistic(
        iHH_df_day0['snp_freq'], iHH_df_day0['iHH'],
        statistic='mean', bins=np.arange(0, 1.05, FREQ_BIN))

    binned_sds, bin_edges, binnumber = binned_statistic(
        iHH_df_day0['snp_freq'], iHH_df_day0['iHH'],
        statistic='std', bins=np.arange(0, 1.05, FREQ_BIN))

    for index, row in iHH_df.iterrows():
        curr_freq = row['snp_freq']
        curr_mean = binned_means[np.where(bin_edges < curr_freq)[0][-1]]
        curr_sd = binned_sds[np.where(bin_edges < curr_freq)[0][-1]]
        iHH_df.loc[index, 'adj_iHH'] = (row['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan

    return iHH_df


def bin_by_position(df, value_col):
    """Bin value_col by HXB2 position (BIN_NUMBER-bp bins) and return the mean
    with a 95% CI band: (bin_mids, mean, lower_conf, upper_conf).
    """
    bins = np.arange(0, GENOME_MAX, BIN_NUMBER)

    binned_averages_std, bin_edges, binnumber = binned_statistic(
        df['hxb2_coord'], df[value_col], statistic='std', bins=bins)
    binned_averages_count, bin_edges, binnumber = binned_statistic(
        df['hxb2_coord'], df[value_col], statistic='count', bins=bins)
    binned_averages, bin_edges, binnumber = binned_statistic(
        df['hxb2_coord'], df[value_col], statistic='mean', bins=bins)

    bin_mids = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
    sem = binned_averages_std / np.sqrt(binned_averages_count)

    lower_conf = binned_averages - 1.96 * sem
    upper_conf = binned_averages + 1.96 * sem

    return bin_mids, binned_averages, lower_conf, upper_conf


def plot_line_with_ci(ax, x, mean, lower, upper, color):
    """Draw a mean line plus a shaded 95% CI band."""
    sns.lineplot(x=x, y=mean, color=color, linewidth=1, ax=ax)
    ax.fill_between(x, lower, upper, alpha=0.2, color=color, linewidth=0.5)


###############################################################################
# Plot in vivo results
###############################################################################
fig, ax = plt.subplots(2, 6, sharex=True, sharey=True)
ax = ax.flatten()

summary_panel_df = []
summary_iHH_mean_df = []

for ax_ind, curr_par in enumerate(PAR_LIST):
    curr_ax = ax[ax_ind]

    # Load the precomputed iHH results and the sequence data
    iHH_df = pd.read_csv(inVivo_inDir + curr_par + '_ihh_results.csv')
    inFile = sequence_path + curr_par + '/885_' + curr_par + '_NT_filtered.fasta'
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)
    participant_dat.load_data_3BNC117(ALLELE_FREQ_THRESH, MULTI_SEG)

    hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath

    # Standardize iHH against the Day-0 frequency distribution
    iHH_df = standardize_ihh(iHH_df)

    # Filter to the timepoints of interest and map sites to HXB2 coordinates
    iHH_df = iHH_df[iHH_df['time_label'].isin(TIMES_TO_KEEP)]
    iHH_df = iHH_df[iHH_df['snp_freq'] > ALLELE_FREQ_THRESH]
    iHH_df['hxb2_coord'] = iHH_df['site'].map(hxb2_nuc_coords_ath)

    # Drop rows whose standardized value is NaN (empty Day-0 frequency bin)
    pre_len = len(iHH_df)
    iHH_df = iHH_df.dropna(subset=['adj_iHH'])
    post_len = len(iHH_df)
    if pre_len != post_len:
        print(f"Warning: Dropped {pre_len - post_len} of {pre_len} rows with NaN" +
              "(indicating there were not enough loci in the day 0 frequency bin)" +
              f" when calculating standardized iHH values for {curr_par}.")
        print('----------------------------------------------------------------')

    # Bin by genome position and plot a line + CI band per timepoint
    for curr_stage in iHH_df['time_label'].unique():
        curr_iHH_df = iHH_df[iHH_df['time_label'] == curr_stage]

        bin_mids, binned_averages, lower_conf, upper_conf = \
            bin_by_position(curr_iHH_df, 'adj_iHH')

        plot_line_with_ci(curr_ax, bin_mids, binned_averages, lower_conf,
                          upper_conf, INV_PALETTE_DICT[curr_stage])

        # Save the binned means for the summary panel
        temp_df = pd.DataFrame({'hxb2_coord': bin_mids,
                                'mean_adj_iHH': binned_averages,
                                'time_label': curr_stage})
        temp_df['participant'] = curr_par
        summary_panel_df.append(temp_df)

    curr_ax.set_title(curr_par)
    curr_ax.set_ylim(*YLIM)
    curr_ax.set_xlim(0, GENOME_MAX)
    curr_ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    # Make a summary csv showing the mean ihh at each time and its iqr
    iHH_mean_df = iHH_df.groupby(['time_label'])['adj_iHH'].agg(
        mean='mean', std='std', count='count').reset_index()
    summary_iHH_mean_df.append(iHH_mean_df.assign(participant=curr_par))

summary_iHH_mean_df = pd.concat(summary_iHH_mean_df)
summary_iHH_mean_df.to_csv(out_dir + 'in_vivo_ihh_summary.csv', index=False)

#Show x ticklabels on the rightmost axis of the top row since the axis
#below it (ax[-1]) is invisible and sharex hides them by default
ax[5].tick_params(axis='x', labelbottom=True)

###############################################################################
# Summary panel: all participants at the closest post-nadir timepoint
###############################################################################
summary_panel_df = pd.concat(summary_panel_df)

# Restrict each participant to Day 0 and their closest post-nadir timepoint
comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))

summary_panel_df['closest_post_nadir'] = summary_panel_df['participant'].map(nadir_time_dict)
summary_panel_df['closest_post_nadir'] = ['W' + str(int(x)) for x in summary_panel_df['closest_post_nadir']]
summary_panel_df = summary_panel_df[
    (summary_panel_df['time_label'] == summary_panel_df['closest_post_nadir']) |
    (summary_panel_df['time_label'] == 'D0')]

# Collapse each participant's post-nadir week (W4/W8) into one shared label so the
# summary panel shows exactly two lines: Day 0 and the merged post-nadir line.
summary_panel_df['summary_time'] = np.where(
    summary_panel_df['time_label'] == 'D0', 'Day 0', 'Post-nadir')

curr_ax = ax[len(PAR_LIST)]

# Group medians across participants at each HXB2 coordinate, with the IQR
# (25th-75th percentile) drawn as a shaded band
summary_stats = summary_panel_df.groupby(
    ['hxb2_coord', 'summary_time'])['mean_adj_iHH'].agg(
        median='median',
        iqr_lower=lambda x: x.quantile(0.25),
        iqr_upper=lambda x: x.quantile(0.75)).reset_index()

for curr_time in ['Day 0', 'Post-nadir']:
    curr_stats = summary_stats[summary_stats['summary_time'] == curr_time].sort_values('hxb2_coord')
    curr_ax.plot(curr_stats['hxb2_coord'], curr_stats['median'],
                 color=SUMMARY_COLOR[curr_time], linewidth=1,
                 linestyle='dotted' if SUMMARY_DASHES[curr_time] else '-')
    curr_ax.fill_between(curr_stats['hxb2_coord'], curr_stats['iqr_lower'],
                         curr_stats['iqr_upper'], alpha=0.2,
                         color=SUMMARY_COLOR[curr_time], linewidth=0.5)

curr_ax.set_title('Median')
curr_ax.set_xlabel('')
curr_ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

# Turn off the unused trailing panel
ax[len(PAR_LIST) + 1].axis('off')

# Two-entry legend for the summary panel (Day 0 vs Post-nadir)
summary_handles = [plt.Line2D([0], [0], color=SUMMARY_COLOR[x], linewidth=1,
                              linestyle='dotted' if SUMMARY_DASHES[x] else '-')
                   for x in ['Day 0', 'Post-nadir']]
curr_ax.legend(summary_handles, ['Day 0', 'Post-nadir'],
               title='', loc='upper left',
               frameon=False, bbox_to_anchor=(1.05, 1.05))

# Three-entry legend for the individual participant panels (Day 0 / Week 4 / Week 8),
# placed inside the final single-participant panel
indiv_handles = [plt.Line2D([0], [0], color=INV_PALETTE_DICT[x], linewidth=1)
                 for x in INV_PALETTE_DICT.keys()]
indiv_labels = [x.replace('D', 'Day ').replace('W', 'Week ') for x in INV_PALETTE_DICT.keys()]
ax[len(PAR_LIST) - 1].legend(indiv_handles, indiv_labels, title='',
                             loc='upper right', frameon=False, bbox_to_anchor=(1, 1))

fig.supxlabel('HXB2 nucleotide position', y=0.04, fontsize=fontsize)
fig.supylabel('Standardized iHH', x=0.05, fontsize=fontsize, ha='center', va='center')
plt.subplots_adjust(hspace=0.3, top=0.90, bottom=0.15, left=0.1, right=0.83)
plt.savefig(out_dir + 'in_vivo_ihh_grid.png', dpi=300)
plt.close()
