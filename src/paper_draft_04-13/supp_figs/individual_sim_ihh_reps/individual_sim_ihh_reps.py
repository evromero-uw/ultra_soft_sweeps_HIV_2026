import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
from scipy.stats import binned_statistic

# Faceted grid of individual-replicate iHH decay curves: one row per origin
# count, one column per example replicate. Each panel overlays standardized
# iHH decay (vs. HXB2 nucleotide position) for several sampled timepoints.

params = {'figure.figsize': (7, 8.5), 'axes.labelsize': 8, 'axes.titlesize': 8,
          'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
          'legend.title_fontsize': 8, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
font_size = 8

BIN_NUMBER = 250
FREQ_BIN = 0.2

SIMS_DIR = '../../../../data/slim_simulations/04-23-2026_sims/'
OUT_DIR = '../../../../results/paper_draft_04-13/supp_figs/individual_sim_ihh_reps/'
os.makedirs(OUT_DIR, exist_ok=True)

sim_dir_stub = 'origins_{origins}_min_{min_num}_rep_{rep}/origins_{origins}_min_{min_num}_rep_{rep}_ihh_results.csv'

ORIGIN_NUM_LIST = [1, 2, 10, 20, 30, 50, 100]
MIN_ORIGIN_DICT = {1: 1, 2: 2, 10: 5, 20: 10, 30: 20, 50: 40, 100: 80}
REP_EXAMPLES = range(0, 5)

TIME_CONVERT_DICT = {299: 'Pre', 300: 'D0', 303: 'W1', 311: 'W4', 322: 'W8'}
TIMES_TO_KEEP = set(['D0', 'W4', 'W8'])

palette = sns.color_palette('rocket', n_colors=4)
palette_dict = {x: palette[i] for i, x in enumerate(['W4', 'W8', 'W12', 'W16'])}
palette_dict['D0'] = 'tab:blue'
palette_dict['Pre'] = 'black'

fig, ax = plt.subplots(len(ORIGIN_NUM_LIST), len(REP_EXAMPLES),
                        sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.3, wspace=0.1)

for row, curr_num_origins in enumerate(ORIGIN_NUM_LIST):
    for col, curr_rep in enumerate(REP_EXAMPLES):
        curr_ax = ax[row, col]
        ehh_sim_file = SIMS_DIR + sim_dir_stub.format(
            origins=curr_num_origins,
            min_num=MIN_ORIGIN_DICT[curr_num_origins],
            rep=curr_rep)

        ehh_df = pd.read_csv(ehh_sim_file)
        ehh_df['time_label'] = ehh_df['time_label'].map(TIME_CONVERT_DICT)

        # Standardize based on the Pre timepoint's frequency-binned iHH distribution.
        ehh_df_pre = ehh_df[ehh_df['time_label'] == 'Pre']

        binned_means, bin_edges, binnumber = binned_statistic(
            ehh_df_pre['snp_freq'], ehh_df_pre['iHH'],
            statistic='mean', bins=np.arange(0, 1.05, FREQ_BIN))

        binned_sds, bin_edges, binnumber = binned_statistic(
            ehh_df_pre['snp_freq'], ehh_df_pre['iHH'],
            statistic='std', bins=np.arange(0, 1.05, FREQ_BIN))

        ehh_df = ehh_df[ehh_df['time_label'].isin(TIMES_TO_KEEP)]

        for index, row_data in ehh_df.iterrows():
            curr_freq = row_data['snp_freq']
            curr_mean = binned_means[np.where(bin_edges < curr_freq)[0][-1]]
            curr_sd = binned_sds[np.where(bin_edges < curr_freq)[0][-1]]

            # Report NaN if the standard deviation is 0 to avoid division by zero errors.
            # This occurs when we don't have any sites in a particular frequency bin.
            ehh_df.loc[index, 'adj_iHH'] = (row_data['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan

        pre_len = len(ehh_df)
        ehh_df = ehh_df.dropna(subset=['adj_iHH'])
        post_len = len(ehh_df)
        if pre_len != post_len:
            print(f"Warning: Dropped {pre_len - post_len} of {pre_len} rows with NaN" +
                  "(indicating there were not enough loci in the day 0 frequency bin)" +
                  f" when calculating standardized iHH values for origins {curr_num_origins}, rep {curr_rep}.")
            print('----------------------------------------------------------------')

        for time_label in ehh_df['time_label'].unique():
            subset_df = ehh_df[ehh_df['time_label'] == time_label]

            binned_averages, bin_edges, binnumber = binned_statistic(
                subset_df['site'], subset_df['adj_iHH'],
                statistic='mean', bins=np.arange(0, 2500, BIN_NUMBER))
            bin_mids = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
            curr_ax.plot(bin_mids, binned_averages, color=palette_dict[time_label],
                         linewidth=1, label=time_label)

            binned_averages_std, bin_edges, binnumber = binned_statistic(
                subset_df['site'], subset_df['adj_iHH'],
                statistic='std', bins=np.arange(0, 2500, BIN_NUMBER))
            binned_averages_count, bin_edges, binnumber = binned_statistic(
                subset_df['site'], subset_df['adj_iHH'],
                statistic='count', bins=np.arange(0, 2500, BIN_NUMBER))
            binned_averages_std = binned_averages_std / np.sqrt(binned_averages_count)

            lower_conf = binned_averages - 1.96 * binned_averages_std
            upper_conf = binned_averages + 1.96 * binned_averages_std

            curr_ax.fill_between(bin_mids, lower_conf, upper_conf, alpha=0.2,
                                 color=palette_dict[time_label], linewidth=0.5)

        if col == 0:
            origin_label = f'{curr_num_origins} origin' if curr_num_origins == 1 else f'{curr_num_origins} origins'
            curr_ax.set_ylabel(f'{origin_label}\nStandardized iHH')

        if row == 0:
            curr_ax.set_title(f'Replicate {curr_rep + 1}')

    last_ax = ax[3, len(REP_EXAMPLES) - 1]
    handles, labels = last_ax.get_legend_handles_labels()
    last_ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), title='Timepoint', frameon=True, fontsize=font_size, title_fontsize=font_size)

fig.supxlabel('HXB2 Nucleotide Position', fontsize = font_size)
plt.tight_layout()

out_path = OUT_DIR + 'individual_sim_ihh_reps.png'
fig.savefig(out_path, dpi=300)
plt.close()
print('Saved figure to', out_path)
