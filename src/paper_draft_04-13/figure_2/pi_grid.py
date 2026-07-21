import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import dataset_metadata
from matplotlib import rcParams
from scipy.stats import binned_statistic
from par_data_class import Pardata
from matplotlib.patches import Patch

#plot the estimates to show how accurate they are
params = {'figure.figsize': (6.5, 2), 'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6

rcParams.update(params)

par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD9K': '#CAB2D6',
               '1HD10K': '#6A3D9A', '1HD11K': '#B15928'}
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']

# In this file I am going to plot the sequence diversity results for the 10-1074 data
# I'll be making a grid where there's a panel for each participant and then a summary

#file paths
out_dir = '../../../results/paper_draft_04-13/figure_2/'
in_vivo_indir = '../../../data/clyde_westfall_2024_final/10-1074/'
in_vivo_csv_dir = '../../../results/paper_draft_04-13/figure_2/'
comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'

# IN VIVO PARAMETERS
# We will only take the sites with a minor allele frequency above this 
# threshold
ALLELE_FREQ_THRESH = 0

# We will only use the two most frequent alleles at each site
MULTI_SEG = True

# The sequence diversity in a sliding window for each rep
COMP_TIME = 'W4'
WINDOW_SIZE = 60
WINDOW_STEP = 20
HXB2_RES_POS = dataset_metadata.RESISTANCE_POS_NT_HXB2 
VARIABLE_LOOP_COORDS = [(459, 470),(384, 418), (294, 331), (155, 196), (129, 157)]
VARIABLE_LOOP_COORDS = list(map(lambda x: (x[0] * 3, x[1] * 3),
                                VARIABLE_LOOP_COORDS))


# IN VIVO DATA
###############################################################################
fig, ax = plt.subplots(2, 6, figsize = (7, 3), sharex = True, sharey = True)
ax = ax.flatten()

# Here we will plot the in vivo data
# Make a dictionary to store the participant coordinate mappings
all_maps_ath = {}
all_maps_hta = {}
seq_length_dict = {}

comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))

# We will loop through each of the participants and get their array to
# hxb2 coordinate mapping and rebound time
for curr_par in par_list:
    #Load the data
    inFile = in_vivo_indir + curr_par + '/885_' + curr_par  +\
             '_NT_filtered.fasta'

    # First, I need to load the data
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)
    participant_dat.load_data_10_1074(HXB2_RES_POS, ALLELE_FREQ_THRESH, MULTI_SEG)

    # Next, I will get a couple of items out of the dataset
    hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
    hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath
    seq_length  = participant_dat.seq_arr
    seq_length = seq_length.shape[1]

    seq_length_dict[curr_par] = seq_length
    all_maps_ath[curr_par] = hxb2_nuc_coords_ath
    all_maps_hta[curr_par] = hxb2_nuc_coords_hta

# Now, I will load the diversity data
all_pi_df = pd.read_csv(in_vivo_csv_dir + '10-1074_pi_vs_time.csv')

#Next, I will plot the results but with change in diversity on the y
pi_change_all = []

for i, curr_par in enumerate(par_list):
    curr_ax = ax[i]


    #Get the data for the current participant
    curr_df = all_pi_df[all_pi_df['participant'] == curr_par]
    seq_length = seq_length_dict[curr_par]
    hxb2_nuc_coords_ath = all_maps_ath[curr_par]
    hxb2_nuc_coords_hta = all_maps_hta[curr_par]

    curr_df['arr_mid'] = curr_df['arr_start'] + WINDOW_SIZE / 2

    #Filter the data to the nadir timepoint
    curr_df = curr_df[curr_df['time'].isin(['W' + str(nadir_time_dict[curr_par]), 'D0'])]
    curr_df['hxb2_mid'] = curr_df['arr_mid'].map(hxb2_nuc_coords_ath)


    

    #Now I will calculate the change in diversity
    pi_change_df = []

    for window_mid, group in curr_df.groupby('hxb2_mid'):
        d0_pi = group[group['time'] == 'D0']['pi'].values
        nadir_pi = group[group['time'] == 'W' + str(nadir_time_dict[curr_par])]['pi'].values

        pi_diff = nadir_pi - d0_pi
        pi_change_df.append([curr_par, window_mid, pi_diff[0]])
    pi_change_df = pd.DataFrame(pi_change_df, columns=['participant', 'hxb2_mid', 'pi_change'])
    pi_change_all.append(pi_change_df)

    curr_ax.plot(pi_change_df['hxb2_mid'], pi_change_df['pi_change'], linewidth=linewidth,
             label = curr_par, color = par_palette[curr_par])
    
    curr_ax.set_xlim(0, seq_length)
    curr_ax.set_title(curr_par, pad = 0.5)
   

pi_change_all = pd.concat(pi_change_all, ignore_index=True)


    
#Plot a windowed median of the data
curr_ax = ax[len(par_list)]
summary_median =  binned_statistic(pi_change_all['hxb2_mid'], pi_change_all['pi_change'],
                                       bins = range(0, 3000, 20), statistic = 'median')
summary_median = pd.DataFrame(zip(summary_median.statistic, summary_median.bin_edges[:-1]),
                                      columns = ['pi', 'hxb2_mid'])
sns.lineplot(data=summary_median, x='hxb2_mid', y='pi', ax=curr_ax, linewidth=linewidth,
             color = 'black')

curr_ax.set_xlabel('')
curr_ax.set_xlim(0, 2500)
curr_ax.set_ylim(-0.05, 0.05)
curr_ax.set_title('Median', pad = 0.5)

#Set the last axis to be invisible
ax[-1].set_visible(False)

#Show x ticklabels on the rightmost axis of the top row since the axis
#below it (ax[-1]) is invisible and sharex hides them by default
ax[5].tick_params(axis='x', labelbottom=True)



#Label the escape loci in every panel
#Figure out which windows overlap with resistance
for i in range(len(par_list) + 1):
    curr_ax = ax[i]
    bin_edges = list(range(0, 3000, 20))
    resistance_bins = set()
    for res_pos in HXB2_RES_POS:
        res_bins = np.digitize([res_pos[0], res_pos[1]], bin_edges)
        for i in range(res_bins[0], res_bins[1]+1):
            resistance_bins.add(i)
    resistance_bins = list(resistance_bins)
    for res_bin in resistance_bins:
        curr_ax.axvspan(bin_edges[res_bin], bin_edges[res_bin + 1], color='red', alpha = 0.05)

    curr_ax.axhline(0, color='black', linewidth=linewidth, linestyle='dotted')

#Make a patch legend for the resistance loci

legend_elements = [Patch(facecolor='red', edgecolor='red', alpha = 0.1, label='Escape loci')]
ax[len(par_list)].legend(handles=legend_elements, loc='upper left', fontsize = fontsize, frameon=False, bbox_to_anchor=(1, 1.05), title = '', title_fontsize = fontsize)
fig.supylabel('Change in diversity (π)', fontsize = fontsize)
fig.supxlabel('HXB2 nucleotide position', fontsize = fontsize)

plt.subplots_adjust(left = 0.1, right = 0.95)
plt.savefig(out_dir + 'pi_vs_time_inVivo_sim.png', dpi = 300)
plt.close()