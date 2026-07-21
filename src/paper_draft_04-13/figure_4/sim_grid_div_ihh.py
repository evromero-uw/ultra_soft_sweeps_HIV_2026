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

#plot the estimates to show how accurate they are
params = {'figure.figsize': (7, 3), 'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6

rcParams.update(params)

# In this file I am going to plot the sequence diversity and iHH results
# for the simulated data 

#For now I am going to run the comparison at week 4 in the simulation

#file paths
out_dir = '../../../results/paper_draft_04-13/figure_4/'
data_dir = '../../../data/slim_simulations/04-23-2026_sims/'
sim_dir_stub_div = 'origins_{origins}_min_{min_num}_rep_{rep}/origins_{origins}_min_{min_num}_rep_{rep}_pi.csv'
sim_dir_stub_ihh = 'origins_{origins}_min_{min_num}_rep_{rep}/origins_{origins}_min_{min_num}_rep_{rep}_ihh_results.csv'

#SIMULATION PARAMETERS
REP = range(0, 10)
ORIGIN_NUM_LIST = [1, 2, 10, 20, 50, 100]
MIN_ORIGIN_DICT= {1: 1, 2: 2, 10: 5, 20: 10, 50: 40, 100: 80}
TIME_CONVERT_DICT = {299: 'Pre', 300: 'D0', 303: 'W1', 311: 'W4', 322: 'W8'}
SIM_PALETTE = sns.color_palette('viridis', n_colors=3)
SIM_PALETTE_DICT = {x: SIM_PALETTE[i] for i, x in enumerate(['D0', 'W4', 'W8'])}

# The sequence diversity in a sliding window for each rep
COMP_TIME = 'W4'
WINDOW_SIZE = 60
WINDOW_STEP = 20
HXB2_RES_POS = dataset_metadata.RESISTANCE_POS_NT_HXB2

FREQ_BIN = 0.2
BIN_NUMBER_IHH = 100
TIMES_TO_KEEP = set(['D0', 'W4', 'W8'])

# Load the simulated data and plot the individual replicates.
#############################################################################
fig, ax = plt.subplots(2, len(ORIGIN_NUM_LIST), sharex = True)

# First, i am going to plot the diversity change in the simulated data
for ind_1, curr_num_origins in enumerate(ORIGIN_NUM_LIST):
    curr_ax_col = ax[:, ind_1]
    curr_div_ax = curr_ax_col[0]
    curr_ihh_ax = curr_ax_col[1]

    # We'll save the change in diversity and iHH for each replicate
    all_pi_change_df = []
    all_ehh_list = []

    for curr_rep in REP:
        #Diversity analysis
        ################################################################
        pi_sim_file = data_dir + sim_dir_stub_div.format(
                                    origins=curr_num_origins,
                                    min_num=MIN_ORIGIN_DICT[curr_num_origins],
                                    rep=curr_rep)

        #Load the pi calculations for this simulation
        sim_pi_df = pd.read_csv(pi_sim_file)
        sim_pi_df['timepoint'] = sim_pi_df['timepoint'].map(TIME_CONVERT_DICT)
        sim_pi_df['window_mid'] = sim_pi_df['window_start'] + WINDOW_SIZE / 2

        #Just plot the difference between week 4 and day 0 for each of the replicates
        sim_pi_df = sim_pi_df[sim_pi_df['timepoint'].isin(['D0', COMP_TIME])]

        #Calculate the change in diversity for each window compared to day 0
        pi_change_df = []
        for window_mid, group in sim_pi_df.groupby('window_mid'):
            d0_pi = group[group['timepoint'] == 'D0']['avg_hamming_dist'].values
            later_pi = group[group['timepoint'] == COMP_TIME]['avg_hamming_dist'].values

            pi_diff = later_pi - d0_pi
            pi_change_df.append([curr_rep, window_mid, pi_diff[0]])
        pi_change_df = pd.DataFrame(pi_change_df, columns=['replicate', 'window_mid', 'pi_change'])
        all_pi_change_df.append(pi_change_df)
        
        curr_div_ax.plot(pi_change_df['window_mid'], pi_change_df['pi_change'],
                    linewidth=linewidth, alpha = 0.2, color = 'gray')
        
        #iHH analysis
        ################################################################
        ehh_sim_file = data_dir + sim_dir_stub_ihh.format(origins=curr_num_origins,
                                                        min_num=MIN_ORIGIN_DICT[curr_num_origins],
                                                        rep=curr_rep)
        
        #Load the EHH results
        ehh_df = pd.read_csv(ehh_sim_file)
        ehh_df['time_label'] = ehh_df['time_label'].map(TIME_CONVERT_DICT)

        #I want to perform the normalization just based on day 0
        ehh_df_pre = ehh_df[ehh_df['time_label'] == 'Pre']
        
        binned_means, bin_edges, binnumber = binned_statistic(ehh_df_pre['snp_freq'],
                                                    ehh_df_pre['iHH'], 
                                                    statistic='mean', 
                                                    bins=np.arange(0, 1.05, FREQ_BIN))

        binned_sds, bin_edges, binnumber = binned_statistic(ehh_df_pre['snp_freq'],
                                                            ehh_df_pre['iHH'], 
                                                            statistic='std', 
                                                            bins=np.arange(0, 1.05, FREQ_BIN))

        #Filter out any additional timepoints
        ehh_df = ehh_df[ehh_df['time_label'].isin(TIMES_TO_KEEP)]

        #Standardize the metric
        for index, row in ehh_df.iterrows():
            curr_freq = row['snp_freq']
            curr_mean = binned_means[np.where(bin_edges < curr_freq)[0][-1]]
            curr_sd = binned_sds[np.where(bin_edges < curr_freq)[0][-1]]

            #Report naN if the standard deviation is 0 to avoid division by zero errors
            #This occurs when we don't have any sites in a particular frequency bin
            ehh_df.loc[index, 'adj_iHH'] = (row['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan
        
        #Filter out any rows with NaN values in the adj_iHH column
        pre_len = len(ehh_df)

        ehh_df = ehh_df.dropna(subset=['adj_iHH'])
        post_len = len(ehh_df)
        if pre_len != post_len:
            print(f"Warning: Dropped {pre_len - post_len} of {pre_len} rows with NaN" +
                  "(indicating there were not enough loci in the day 0 frequency bin)" +
                  f" when calculating standardized iHH values for origins {curr_num_origins}, rep {curr_rep}.")
            print('----------------------------------------------------------------')
        ehh_df['origins'] = curr_num_origins
        all_ehh_list.append(ehh_df)

    
    # Plot a windowed mean of the diversity data
    all_pi_change_df = pd.concat(all_pi_change_df, ignore_index=True)
    summary_mean =  binned_statistic(all_pi_change_df['window_mid'], all_pi_change_df['pi_change'],
                                           bins = range(0, 3000, 20), statistic = 'mean')
    summary_mean = pd.DataFrame(zip(summary_mean.statistic, summary_mean.bin_edges[:-1]),
                                          columns = ['pi', 'window_mid'])
    sns.lineplot(data=summary_mean, x='window_mid', y='pi', ax=curr_div_ax, linewidth=linewidth,
                 color = 'black')


    #Plot the windowed mean of the iHH data
    all_ehh_df = pd.DataFrame() 
    while len(all_ehh_list) > 0:
        df = all_ehh_list.pop()
        all_ehh_df = pd.concat([all_ehh_df, df])

    # Separate out the timepoints and plot the mean for each timepoint
    # across the replicates.
    for ind_2, time_label in enumerate(all_ehh_df['time_label'].unique()):
        subset_df = all_ehh_df[all_ehh_df['time_label'] == time_label]

        binned_averages, bin_edges, binnumber = binned_statistic(subset_df['site'],
                                                            subset_df['adj_iHH'], 
                                                            statistic='mean', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER_IHH))
        bin_mids = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
        sns.lineplot(x = bin_mids, y=binned_averages, 
                    color= SIM_PALETTE_DICT[time_label],
                    linewidth=1, ax = curr_ihh_ax, label = time_label)

        #look at the error bars
        binned_averages_std, bin_edges, binnumber = binned_statistic(subset_df['site'],
                                                            subset_df['adj_iHH'], 
                                                            statistic='std', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER_IHH))
        binned_averages_count, bin_edges, binnumber = binned_statistic(subset_df['site'],
                                                            subset_df['adj_iHH'], 
                                                            statistic='count', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER_IHH))
        binned_averages_std = binned_averages_std / np.sqrt(binned_averages_count)

        ehh_df_d0 = ehh_df[ehh_df['time_label'] == 'D0']
        binned_averages_d0, bin_edges, binnumber = binned_statistic(ehh_df_d0['site'],
                                                            ehh_df_d0['adj_iHH'], 
                                                            statistic='mean', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER_IHH))
        
        #plot the errorbars
        lower_conf = binned_averages - 1.96 * binned_averages_std
        upper_conf = binned_averages + 1.96 * binned_averages_std

        curr_ihh_ax.fill_between(bin_mids, lower_conf, upper_conf, alpha=0.2,
                            color=SIM_PALETTE_DICT[time_label], linewidth=0.5)
                


    #Figure out which windows overlap with resistance
    bin_edges = list(range(0, 3000, 20))
    resistance_bins = set()
    for res_pos in HXB2_RES_POS:
        res_bins = np.digitize([res_pos[0], res_pos[1]], bin_edges)
        for j in range(res_bins[0], res_bins[1]+1):
            resistance_bins.add(j)
    resistance_bins = list(resistance_bins)
    for res_bin in resistance_bins:
        curr_div_ax.axvspan(bin_edges[res_bin], bin_edges[res_bin + 1], color='red', alpha = 0.1)

    #Axis formatting for each of the two plots
    curr_div_ax.set_title(f'{curr_num_origins} origins')
    curr_div_ax.axhline(0, color='black', linestyle='dashed', linewidth=linewidth)
    curr_div_ax.set_xlim(0, 2500)
    curr_div_ax.set_ylim(-0.01, 0.015)
    curr_div_ax.set_xlabel('')
    curr_div_ax.set_ylabel('')

    curr_ihh_ax.set_xlim(0, 2500)
    curr_ihh_ax.set_ylim(-2, 25)
    curr_ihh_ax.set_xlabel('')
    curr_ihh_ax.set_ylabel('')
    curr_ihh_ax.axhline(0, color='black', linestyle='dashed', linewidth=linewidth, zorder=0)

    if curr_num_origins == 1:
        curr_div_ax.set_title(f'{curr_num_origins} origin')
        curr_div_ax.set_ylabel('Change in diversity (π)')
        curr_ihh_ax.set_ylabel('Standardized iHH')
    else:
        #turn off the xticks
        curr_div_ax.set_yticks([])
        curr_ihh_ax.set_yticks([])
    
    #Give the final plot a legend
    if curr_num_origins == ORIGIN_NUM_LIST[-1]:
        curr_ihh_ax.legend(title = 'Timepoint', loc = 'upper right', bbox_to_anchor=(1.3, 1))
        #reset the legend labels
        handles, labels = curr_ihh_ax.get_legend_handles_labels()
        new_labels = ['Week ' + label[1:] if label[0] == 'W' else 'Day ' + label[1:] for label in labels]
        curr_ihh_ax.legend(handles=handles, labels=new_labels, title='Timepoint', loc='upper right', bbox_to_anchor=(1, 1))
    else:
        curr_ihh_ax.legend().set_visible(False)


fig.supxlabel('HXB2 nucleotide position', fontsize = fontsize)
    
plt.subplots_adjust(top=0.9, bottom=0.1)
plt.savefig(out_dir + 'sim_grid_div_ihh.png', dpi = 300)
plt.close()