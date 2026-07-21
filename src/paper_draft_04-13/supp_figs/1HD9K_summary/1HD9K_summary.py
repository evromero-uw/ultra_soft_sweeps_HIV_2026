import os
import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import binned_statistic

import dataset_metadata
from par_data_class import Pardata

# Participant 1HD9K is a non-responder (early rebound at Week 1, no defined
# post-nadir timepoint), so it is excluded from the main multi-participant
# figures. This standalone supplemental figure summarizes its viral load,
# diversity, and linkage (iHH) trajectories instead.

###############################################################################
# Plotting style
###############################################################################
params = {'figure.figsize': (7, 2.2), 'axes.labelsize': 8, 'axes.titlesize': 8,
          'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
          'legend.title_fontsize': 8, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 8
rcParams.update(params)

###############################################################################
# Paths
###############################################################################
PARTICIPANT = '1HD9K'
vl_file = '../../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
pi_file = '../../../../results/paper_draft_04-13/figure_2/10-1074_pi_vs_time.csv'
ihh_file = '../../../../results/paper_draft_04-13/02-19-2026_EHH_10-1074/' + PARTICIPANT + '_ihh_results.csv'
seq_file = ('../../../../data/clyde_westfall_2024_final/10-1074/' + PARTICIPANT +
            '/885_' + PARTICIPANT + '_NT_filtered.fasta')
out_dir = '../../../../results/paper_draft_04-13/supp_figs/1HD9K_summary/'

os.makedirs(out_dir, exist_ok=True)

ALLELE_FREQ_THRESH = 0
MULTI_SEG = True
FREQ_BIN = 0.2
BIN_NUMBER = 100
GENOME_MAX = 2500
RESISTANCE_POS_NT_HXB2 = dataset_metadata.RESISTANCE_POS_NT_HXB2

TIME_ORDER = ['D0', 'W1', 'W4']
TIME_LABELS = {'D0': 'Day 0', 'W1': 'Week 1', 'W4': 'Week 4'}
PALETTE = sns.color_palette('rocket', n_colors=len(TIME_ORDER))
PALETTE_DICT = {t: PALETTE[i] for i, t in enumerate(TIME_ORDER)}


###############################################################################
# Helpers (mirrors figure_3/ihh_grid_in_vivo.py)
###############################################################################
def standardize_ihh(iHH_df):
    """Standardize iHH against the Day-0 distribution within SNP-frequency bins."""
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
    with a 95% CI band: (bin_mids, mean, lower_conf, upper_conf)."""
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


fig, axs = plt.subplots(1, 3, figsize=(7, 2.2), dpi=300)

###############################################################################
# Panel 1: viral load over time
###############################################################################
ax = axs[0]

vl_df = pd.read_csv(vl_file)
vl_df = vl_df[vl_df['Study_ID'] == PARTICIPANT]
vl_df = vl_df.melt(id_vars=['Study_ID'], var_name='Timepoint', value_name='Viral_Load')
vl_df = vl_df[~vl_df['Timepoint'].isin(['screen', 'pre', 'Rebound', 'Nadir'])]
vl_df['Timepoint'] = [t if t != 'D7' else 'W1' for t in vl_df['Timepoint']]
vl_df['time_label_int'] = [int(tp[1:]) * 7 for tp in vl_df['Timepoint']]
vl_df = vl_df.dropna(subset=['Viral_Load'])
vl_df = vl_df[vl_df['time_label_int'] <= 112]
vl_df = vl_df.sort_values('time_label_int')

ax.plot(vl_df['time_label_int'], vl_df['Viral_Load'], color='black',
        marker='o', markersize=2, linewidth=1)
ax.set_yscale('log')
ax.set_xlim(0, 112)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_xticklabels(['0', '4', '8', '12', '16'])
ax.set_xlabel('Trial week', labelpad=0.6)
ax.set_ylabel('Viral load (copies/mL)', labelpad=0.6)
ax.set_title(PARTICIPANT)

###############################################################################
# Panel 2: change in diversity, Day 0 -> Week 4
###############################################################################
ax = axs[1]

pi_df = pd.read_csv(pi_file)
print(pi_df.head())
pi_df = pi_df[pi_df['participant'] == PARTICIPANT].copy()
print(pi_df.head())
pi_df['hxb2_mid'] = (pi_df['hxb2_start'] + pi_df['hxb2_end']) / 2

pi_change_records = []
for hxb2_mid, group in pi_df.groupby('hxb2_mid'):
    d0_pi = group[group['time'] == 'D0']['pi'].values
    w4_pi = group[group['time'] == 'W4']['pi'].values
    if len(d0_pi) == 0 or len(w4_pi) == 0:
        continue
    pi_change_records.append([hxb2_mid, w4_pi[0] - d0_pi[0]])
pi_change_df = pd.DataFrame(pi_change_records, columns=['hxb2_mid', 'pi_change'])
pi_change_df = pi_change_df.sort_values('hxb2_mid')

ax.plot(pi_change_df['hxb2_mid'], pi_change_df['pi_change'], color='black',
        linewidth=1)
ax.axhline(0, color='gray', linestyle='dashed', linewidth=linewidth)

# Shade and label the drug-resistance locus
res_start = min(pos[0] for pos in RESISTANCE_POS_NT_HXB2)
res_end = max(pos[1] for pos in RESISTANCE_POS_NT_HXB2)
for res_pos in RESISTANCE_POS_NT_HXB2:
    ax.axvspan(res_pos[0], res_pos[1], color='red', alpha=0.15)
ax.text(900, 0.97, 'Escape\nloci',
        transform=ax.get_xaxis_transform(), ha='right', va='top',
        fontsize=fontsize, color='red')

ax.set_xlabel('HXB2 nucleotide position', labelpad=0.6)
ax.set_ylabel('Change in diversity (π)', labelpad=0.6)
ax.set_title('Day 0 → Week 4 diversity change')

###############################################################################
# Panel 3: standardized iHH per timepoint
###############################################################################
ax = axs[2]

participant_dat = Pardata(seq_file, 'clyde2024', PARTICIPANT)
participant_dat.load_data_10_1074(dataset_metadata.RESISTANCE_POS_NT_HXB2,
                                   ALLELE_FREQ_THRESH, MULTI_SEG)
hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath

iHH_df = pd.read_csv(ihh_file)
iHH_df = standardize_ihh(iHH_df)
iHH_df = iHH_df[iHH_df['time_label'].isin(TIME_ORDER)]
iHH_df['hxb2_coord'] = iHH_df['site'].map(hxb2_nuc_coords_ath)

pre_len = len(iHH_df)
iHH_df = iHH_df.dropna(subset=['adj_iHH'])
post_len = len(iHH_df)
if pre_len != post_len:
    print(f"Warning: Dropped {pre_len - post_len} of {pre_len} rows with NaN" +
          "(indicating there were not enough loci in the day 0 frequency bin)" +
          f" when calculating standardized iHH values for {PARTICIPANT}.")

for curr_time in TIME_ORDER:
    curr_iHH_df = iHH_df[iHH_df['time_label'] == curr_time]
    if curr_iHH_df.shape[0] == 0:
        continue
    bin_mids, binned_averages, lower_conf, upper_conf = bin_by_position(curr_iHH_df, 'adj_iHH')
    sns.lineplot(x=bin_mids, y=binned_averages, color=PALETTE_DICT[curr_time],
                 linewidth=1, ax=ax, label=TIME_LABELS[curr_time])
    ax.fill_between(bin_mids, lower_conf, upper_conf, alpha=0.2,
                     color=PALETTE_DICT[curr_time], linewidth=0.5)

ax.axhline(0, color='gray', linestyle='dashed', linewidth=linewidth)
ax.set_xlim(0, GENOME_MAX)
ax.set_xlabel('HXB2 nucleotide position', labelpad=0.6)
ax.set_ylabel('Genetic linkage\n(standardized iHH)', labelpad=0.6)
ax.set_title('Linkage over time')
ax.legend(title='', loc='upper left', frameon=True, fontsize=fontsize)

plt.tight_layout()
out_path = out_dir + '1HD9K_summary.png'
fig.savefig(out_path, dpi=300)
print('Saved figure to', out_path)
