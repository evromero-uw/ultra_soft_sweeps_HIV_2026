import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

import data_util as du
import dataset_metadata
import diversity_stats as div
from par_data_class import Pardata

# In this file I am going to calculate the overall diversity
# (not windowed by hxb2 coordinate)

# I'll run the calculations for the simulated and in vivo data.

# Here is the output folder
outDir = '../../../results/paper_draft_04-13/figure_2/'

# Here is the path to the data
inDir_10_1074 = '../../../data/clyde_westfall_2024_final/10-1074/'
sim_data_dir = '../../../data/slim_simulations/02-20-2026_sims/'
sim_dir_stub = sim_data_dir + 'origins_{origins}_min_{min_num}_rep_{rep}/analysis/FilteredGenotypes'
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
par_order = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
time_filter_out = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2']

comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'


###############################################################################
# In vivo data 10-1074


# We will only take the sites with a minor allele frequency above this 
# threshold
ALLELE_FREQ_THRESH = 0
# We will use all segregating alleles at each site
MULTI_SEG = True
hxb2_res_positions = dataset_metadata.RESISTANCE_POS_NT_HXB2

# Make a dataframe to store the results
all_pi_df = []

# We will loop through each of the participants and calculate the sequence
# diversity in a sliding window
for curr_par in par_list:
    print(curr_par)

    #Load the data
    inFile = inDir_10_1074 + curr_par + '/885_' + curr_par  + '_NT_filtered.fasta'

    # First, I need to load the data
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)
    participant_dat.load_data_10_1074(hxb2_res_positions, ALLELE_FREQ_THRESH, MULTI_SEG)

    # Next, I will get a couple of items out of the dataset
    hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
    hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath
    seq_info_df = participant_dat.seq_info_df.copy()
    seq_info_df = seq_info_df[~seq_info_df['time_label'].isin(time_filter_out)]
    seq_arr = participant_dat.seq_arr

    seq_length = seq_arr.shape[1]

 
    for curr_time in seq_info_df['time_label'].unique():
        # Get the dataframe and the sequences for the current timepoint
        curr_time_df = seq_info_df[seq_info_df['time_label'] == curr_time]


        # Now I will calculate the sequence diversity for the current window
        curr_div = div.calc_ave_pairwise_hamming(seq_arr, curr_time_df)
        all_pi_df.append([curr_par, curr_time, curr_div, '10-1074'])


all_pi_df = pd.DataFrame(all_pi_df, columns=['participant', 'time', 'pi', 'dataset'])
all_pi_df.to_csv(outDir + 'inv_pi_summary.csv', index=False)

# I also want to calculate the distribution with IQR at the comparison timepoint
comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))


all_pi_df['comp_time'] = all_pi_df['participant'].map(nadir_time_dict)
all_pi_df['comp_time'] = ['W' + str(x) if x != 0 else 'D0' for x in all_pi_df['comp_time']]
comp_time_pi_df = all_pi_df[(all_pi_df['time'] == all_pi_df['comp_time']) | (all_pi_df['time'] == 'D0')]

# Calculate the diversity change from D0 to the comparison timepoint
pi_change_df = []
for name, group in comp_time_pi_df.groupby('participant'):
    d0_pi = group[group['time'] == 'D0']['pi'].values[0]
    comp_time_pi = group[group['time'] == group['comp_time'].values[0]]['pi'].values[0]
    pi_change = comp_time_pi - d0_pi
    
    # Also calculate the diversity change as a percentage of the D0 diversity
    pi_change_percent = (pi_change / d0_pi) * 100 if d0_pi != 0 else np.nan
    pi_change_df.append([name, d0_pi, comp_time_pi, pi_change, pi_change_percent])
pi_change_df = pd.DataFrame(pi_change_df, columns=['participant', 'd0_pi', 'comp_time_pi', 'pi_change', 'pi_change_percent'])

# Print the mean and IQR of the diversity change in scientific notation
mean_change = pi_change_df['pi_change_percent'].mean()
lower_q = pi_change_df['pi_change_percent'].quantile(0.25)
upper_q = pi_change_df['pi_change_percent'].quantile(0.75)
print(f'Mean change in diversity: {mean_change:.2e}')
print(f'IQR of change in diversity: [{lower_q:.2e}, {upper_q:.2e}]')


# Plot a histogram of the percent change in diversity from D0 to the
# comparison timepoint
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(y = 'pi_change_percent', x = 'participant', data = pi_change_df, ax=ax,
            order=par_order)
ax.axhline(0, color = 'black', linestyle = '--')
ax.set_xlabel('Participant')
ax.set_ylabel('Change in diversity (% of D0)')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
plt.savefig(outDir + 'inv_pi_change_percent_hist.png', dpi=300)
plt.close()

