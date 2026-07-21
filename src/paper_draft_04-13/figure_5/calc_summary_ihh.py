import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/3BNC117/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import rcParams
from scipy.stats import binned_statistic

import data_util as du
import area_between_curves
from par_data_class import Pardata

#In this file I want to plot the area between curves side by side for
#the 10-1075 and 3BNC117 datasets
params = {'figure.figsize': (5, 2), 'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial', 'axes.titlesize': 6}
linewidth = 0.5
fontsize = 6
rcParams.update(params)


out_dir = '../../../results/paper_draft_04-13/figure_5/'

#EHH file paths
sequence_path_1074 = '../../../data/clyde_westfall_2024_final/10-1074/'
sequence_path_3BNC117 = '../../../data/clyde_westfall_2024_final/3BNC117/'
ehh_sim_dir = '../../../data/slim_simulations/04-23-2026_sims/'
ehh_inv_dir_1074 = '../../../results/paper_draft_04-13/02-19-2026_EHH_10-1074/'
ehh_inv_dir_3BNC117 = '../../../results/02-19-2026_exploratory/02-19-2026_EHH_3BNC117/'
sim_dir_stub = 'origins_{origins}_min_{min_num}_rep_{rep}/origins_{origins}_min_{min_num}_rep_{rep}_ihh_results.csv'




#Shared parameters for both the simulations and the participant data
BIN_NUMBER = 250
FREQ_BIN = 0.2

#participant parameters
PAR_LIST_1074 = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K',
                 '1HD7K', '1HD9K','1HD10K', '1HD11K']
PAR_LIST_3BNC117 = ['2C1', '2C5', '2E1', '2E2', '2E3', '2E4', '2E5', '2E7']
TIME_FILTER_OUT = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2', 'W1']
ALLELE_FREQ_THRESH = 0
MULTI_SEG = True
par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD9K': '#CAB2D6',
               '1HD10K': '#6A3D9A', '1HD11K': '#B15928'}
par_palette_3BNC117 = {
               '2C1': '#A6CEE3', '2C5': '#1F78B4', '2E1': '#B2DF8A',
                        '2E2': '#33A02C', '2E3': '#FB9A99', '2E4': '#E31A1C',
                        '2E5': '#FDBF6F', '2E7': '#FF7F00'}

#simulation parameters
ORIGIN_NUM_LIST = [1, 2, 10, 20, 30, 50, 100]
MIN_DICT = {1: 1, 2: 2, 10: 5, 20: 10, 30: 20, 50: 40, 100: 80}
REP = range(0, 100)
FRAGMENT_LEN = 2571

STRIPPLOT_ORDER = ORIGIN_NUM_LIST + ['10-1074'] #+ ['3BNC117']
print(STRIPPLOT_ORDER)

TIME_CONVERT_DICT = {299: 'Pre', 300: 'D0', 303: 'W1', 311: 'W4', 322: 'W8'}
TIMES_TO_KEEP = set(['Pre', 'D0', 'W4', 'W8'])
LEGEND_ORDERS = ['W4', 'W8']

###############################################################################
# Here I am going to calculate the area between the curves for the simulated
# data.

sim_area_df = []

for ind_1, curr_num_origins in enumerate(ORIGIN_NUM_LIST):
    for curr_rep in tqdm(REP, desc=f"Processing {curr_num_origins} origins"):
        ehh_sim_file = ehh_sim_dir + sim_dir_stub.format(origins=curr_num_origins,
                                                         min_num=MIN_DICT[curr_num_origins],
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
            ehh_df.loc[index, 'adj_iHH'] = (row['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan
        
        
        #Filter out any rows with NaN values in the adj_iHH column
        ehh_df = ehh_df.dropna(subset=['adj_iHH'])

        
        #Get the binned averages for day 0 as a comparison point
        ehh_df_d0 = ehh_df[ehh_df['time_label'] == 'D0']
        binned_averages_d0, bin_edges, binnumber = binned_statistic(ehh_df_d0['site'],
                                                            ehh_df_d0['adj_iHH'], 
                                                            statistic='mean', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER))
        binned_averages_d0_std, bin_edges, binnumber = binned_statistic(ehh_df_d0['site'],
                                                            ehh_df_d0['adj_iHH'], 
                                                            statistic='std', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER))
        binned_averages_d0_count, bin_edges, binnumber = binned_statistic(ehh_df_d0['site'],
                                                            ehh_df_d0['adj_iHH'], 
                                                            statistic='count', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER))

        binned_averages_d0_std = binned_averages_d0_std / np.sqrt(binned_averages_d0_count)

        #For each time_label, calculate the area between the curves
        mean_averages = []
        y_text_pos = 0.9
        for ind_2, time_label in enumerate(ehh_df['time_label'].unique()):
            if time_label == 'D0':
                continue
            subset_df = ehh_df[ehh_df['time_label'] == time_label]

            #Calculate the binned averages for the current time_label
            binned_averages, bin_edges, binnumber = binned_statistic(subset_df['site'],
                                                                subset_df['adj_iHH'], 
                                                                statistic='mean', 
                                                                bins=np.arange(0, 2500, BIN_NUMBER))
            mean_averages.append(binned_averages)
            bin_mids = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2

            #If the current time isn't day 0, calculate the area between the curves
            calculated_area = area_between_curves.area_between_curves(bin_mids, binned_averages_d0, binned_averages)

            #Normalize the calculated area over the fragment length
            calculated_area = calculated_area / FRAGMENT_LEN

            #Record the area between the curves for each timepoint
            sim_area_df.append({'origins': curr_num_origins, 'rep': curr_rep,
                                'time_label': time_label, 'area': calculated_area})

#make a dataframe of the area between each of the curves and save it
sim_area_df = pd.DataFrame(sim_area_df)

sim_area_df.to_csv(out_dir + 'sim_area_df.csv', index=False)

###############################################################################
# Load the 10-1074 in vivo data
###############################################################################
inv_area_df = []

# Here I am going to load and calculate the area between the curves for the
# 10-1074 dataset. 
for ax_ind, curr_par in enumerate(PAR_LIST_1074):
        
    iHH_df = pd.read_csv(ehh_inv_dir_1074 + curr_par + '_ihh_results.csv')
    # First, I need to load the data
    inFile = sequence_path_1074 + curr_par + '/885_' + curr_par + \
                    '_NT_filtered.fasta'
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)
    participant_dat.load_data_3BNC117(ALLELE_FREQ_THRESH, MULTI_SEG)

    #Get the length of the current fragment
    seqArr = participant_dat.seq_arr

    #Filter out the array columns which are more than half gaps
    gap_cols = du.make_gap_col_set(seqArr, gap_thresh = 0.5)
    seqArr = seqArr[:, [i for i in range(seqArr.shape[1]) if i not in gap_cols]]
    par_frag_length = seqArr.shape[1]


    # Now I will get some datastructures out of the participant data object
    hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
    hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath

    #First I need to standardize the values based on the day 0 values
    iHH_df_day0 = iHH_df[iHH_df['time_label'] == 'D0']
    
    binned_means, bin_edges, binnumber = binned_statistic(iHH_df_day0['snp_freq'],
                                                        iHH_df_day0['iHH'], 
                                                        statistic='mean', 
                                                        bins=np.arange(0, 1.05, FREQ_BIN))
    

    binned_sds, bin_edges, binnumber = binned_statistic(iHH_df_day0['snp_freq'],
                                                        iHH_df_day0['iHH'], 
                                                        statistic='std', 
                                                        bins=np.arange(0, 1.05, FREQ_BIN))

    #Next I will go through and calculate the standardized iHS for each haplotype at each site
    for index, row in iHH_df.iterrows():
        curr_freq = row['snp_freq']
        curr_mean = binned_means[np.where(bin_edges < curr_freq)[0][-1]]
        curr_sd = binned_sds[np.where(bin_edges < curr_freq)[0][-1]]
        #Report naN if the standard deviation is 0 to avoid division by zero errors
        #This occurs when we don't have any sites in a particular frequency bin
        iHH_df.loc[index, 'adj_iHH'] = (row['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan
    
    
    #Filter out any rows with NaN values in the adj_iHH column
    iHH_df = iHH_df.dropna(subset=['adj_iHH'])
    iHH_df = iHH_df[~iHH_df['time_label'].isin(TIME_FILTER_OUT)]

    iHH_df = iHH_df[iHH_df['snp_freq'] > ALLELE_FREQ_THRESH]
    iHH_df['hxb2_coord'] = iHH_df['site'].map(hxb2_nuc_coords_ath)

    #Get the binned averages for day 0 as a comparison point
    iHH_df_d0 = iHH_df[iHH_df['time_label'] == 'D0']
    binned_averages_d0, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
                                                        iHH_df_d0['adj_iHH'], 
                                                        statistic='mean', 
                                                        bins=np.arange(0, 2500, BIN_NUMBER))
    binned_averages_d0_std, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
                                                        iHH_df_d0['adj_iHH'], 
                                                        statistic='std', 
                                                        bins=np.arange(0, 2500, BIN_NUMBER))
    binned_averages_d0_count, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
                                                        iHH_df_d0['adj_iHH'], 
                                                        statistic='count', 
                                                        bins=np.arange(0, 2500, BIN_NUMBER))
    
    binned_averages_d0_std = binned_averages_d0_std / np.sqrt(binned_averages_d0_count)

    #Loop through each stage and calculate the area between the curves for each timepoint compared to day 0
    for curr_stage in iHH_df['time_label'].unique():
        if curr_stage == 'D0':
            continue
        curr_iHH_df = iHH_df[iHH_df['time_label'] == curr_stage]
        
        #Check how many sequences are sampled at this timepoint
        time_sample_size = participant_dat.seq_info_df[participant_dat.seq_info_df['time_label'] == curr_stage].shape[0]


        binned_averages, bin_edges, binnumber = binned_statistic(curr_iHH_df['hxb2_coord'],
                                                            curr_iHH_df['adj_iHH'], 
                                                            statistic='mean', 
                                                            bins=np.arange(0, 2500, BIN_NUMBER))
        
        #If the current time isn't day 0, calculate the area between the curves and print it
        calculated_area = area_between_curves.area_between_curves(bin_mids,
                                                                binned_averages_d0,
                                                                binned_averages)
        
        calculated_area = calculated_area / par_frag_length
        
        #Record the area between the curves for each timepoint
        inv_area_df.append({'participant': curr_par,
                             'time_label': curr_stage, 'area': calculated_area})

inv_area_df = pd.DataFrame(inv_area_df)
inv_area_df['data_type'] = 'In vivo'
inv_area_df['dataset'] = '10-1074'

# ###############################################################################
# # Load the 3BNC117 in vivo data
# ###############################################################################
# inv_area_df_3BNC117 = []

# # Here I am going to load and calculate the area between the curves for the
# # 3BNC117 dataset. 
# for ax_ind, curr_par in enumerate(PAR_LIST_3BNC117):
        
#     iHH_df = pd.read_csv(ehh_inv_dir_3BNC117 + curr_par + '_ihh_results.csv')
#     # First, I need to load the data
#     inFile = sequence_path_3BNC117 + curr_par + '/835_' + curr_par + \
#                     '_NT.fasta'
#     participant_dat = Pardata(inFile, 'clyde2024', curr_par)
#     participant_dat.load_data_3BNC117(ALLELE_FREQ_THRESH, MULTI_SEG)


#     # Now I will get some datastructures out of the participant data object
#     hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
#     hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath

#     #First I need to standardize the values based on the day 0 values
#     iHH_df_day0 = iHH_df[iHH_df['time_label'] == 'D0']
    
#     binned_means, bin_edges, binnumber = binned_statistic(iHH_df_day0['snp_freq'],
#                                                         iHH_df_day0['iHH'], 
#                                                         statistic='mean', 
#                                                         bins=np.arange(0, 1.05, FREQ_BIN))

#     binned_sds, bin_edges, binnumber = binned_statistic(iHH_df_day0['snp_freq'],
#                                                         iHH_df_day0['iHH'], 
#                                                         statistic='std', 
#                                                         bins=np.arange(0, 1.05, FREQ_BIN))

#     #Next I will go through and calculate the standardized iHS for each haplotype at each site
#     for index, row in iHH_df.iterrows():
#         curr_freq = row['snp_freq']
#         curr_mean = binned_means[np.where(bin_edges < curr_freq)[0][-1]]
#         curr_sd = binned_sds[np.where(bin_edges < curr_freq)[0][-1]]
#         iHH_df.loc[index, 'adj_iHH'] = (row['iHH'] - curr_mean) / curr_sd if curr_sd > 0 else np.nan

#     iHH_df = iHH_df[iHH_df['time_label'].isin(TIMES_TO_KEEP)]

#     iHH_df = iHH_df[iHH_df['snp_freq'] > ALLELE_FREQ_THRESH]
#     iHH_df['hxb2_coord'] = iHH_df['site'].map(hxb2_nuc_coords_ath)

#     #Filter out any rows with NaN values in the adj_iHH column
#     iHH_df = iHH_df.dropna(subset=['adj_iHH'])

#     #Get the binned averages for day 0 as a comparison point
#     iHH_df_d0 = iHH_df[iHH_df['time_label'] == 'D0']
#     binned_averages_d0, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
#                                                         iHH_df_d0['adj_iHH'], 
#                                                         statistic='mean', 
#                                                         bins=np.arange(0, 2500, BIN_NUMBER))
#     binned_averages_d0_std, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
#                                                         iHH_df_d0['adj_iHH'], 
#                                                         statistic='std', 
#                                                         bins=np.arange(0, 2500, BIN_NUMBER))
#     binned_averages_d0_count, bin_edges, binnumber = binned_statistic(iHH_df_d0['site'],
#                                                         iHH_df_d0['adj_iHH'], 
#                                                         statistic='count', 
#                                                         bins=np.arange(0, 2500, BIN_NUMBER))
#     binned_averages_d0_std = binned_averages_d0_std / np.sqrt(binned_averages_d0_count)

#     #Loop through each stage and calculate the area between the curves for each timepoint compared to day 0
#     for curr_stage in iHH_df['time_label'].unique():
#         if curr_stage == 'D0':
#             continue
#         curr_iHH_df = iHH_df[iHH_df['time_label'] == curr_stage]
        
#         #Check how many sequences are sampled at this timepoint
#         time_sample_size = participant_dat.seq_info_df[participant_dat.seq_info_df['time_label'] == curr_stage].shape[0]


#         binned_averages, bin_edges, binnumber = binned_statistic(curr_iHH_df['hxb2_coord'],
#                                                             curr_iHH_df['adj_iHH'], 
#                                                             statistic='mean', 
#                                                             bins=np.arange(0, 2500, BIN_NUMBER))
        
#         #If the current time isn't day 0, calculate the area between the curves and print it
#         calculated_area = area_between_curves.area_between_curves(bin_mids,
#                                                                 binned_averages_d0,
#                                                                 binned_averages)
        
#         #Try to record the area between the curves for each timepoint
#         inv_area_df_3BNC117.append({'participant': curr_par,
#                              'time_label': curr_stage, 'area': calculated_area})

# inv_area_df_3BNC117 = pd.DataFrame(inv_area_df_3BNC117)
# inv_area_df_3BNC117['data_type'] = 'In vivo'
# inv_area_df_3BNC117['dataset'] = '3BNC117'

# inv_area_df = pd.concat([inv_area_df, inv_area_df_3BNC117])
#save the area between curves dataframe
inv_area_df.to_csv(out_dir + 'inv_area_df.csv', index=False)

# # ###############################################################################
#now I want to plot the area between the curves for the simulated data and
#the in vivo data side by side


fig, ax = plt.subplots(1, len(LEGEND_ORDERS), sharey=True)

for ind, time_label in enumerate(LEGEND_ORDERS):
    if time_label == 'Pre':
        plot_order = ORIGIN_NUM_LIST
    else:
        plot_order = STRIPPLOT_ORDER
    subset_df_sim = sim_area_df[sim_area_df['time_label'] == time_label]
    sns.stripplot(x = subset_df_sim['origins'], y = subset_df_sim['area'], ax=ax[ind],
                order = plot_order,
                  alpha = 0.3, marker='.', color='black')
    ax[ind].set_title(f'Area between D0 and {time_label}')
    ax[ind].set_ylabel('Area between curves')
    ax[ind].set_xlabel('Number of Origins')


    subset_df_inv = inv_area_df[inv_area_df['time_label'] == time_label]
    subset_df_inv_1074 = subset_df_inv[subset_df_inv['dataset'] == '10-1074']
    # subset_df_inv_3BNC117 = subset_df_inv[subset_df_inv['dataset'] == '3BNC117']
    sns.stripplot(x = 'dataset', y = 'area', data=subset_df_inv_1074, ax=ax[ind],
                  hue = 'participant', palette = par_palette, dodge=True, jitter=True,
                  order = plot_order,
                  alpha = 1, color = 'black',
                  linewidth=linewidth, size=4)
    # sns.stripplot(x = 'dataset', y = 'area', data=subset_df_inv_3BNC117, ax=ax[ind],
    #               hue = 'participant', palette = par_palette_3BNC117, dodge=True, jitter=True,
    #               order = plot_order, marker = '^',
    #               alpha = 1, color = 'black',
    #               linewidth=linewidth, size=4)
    ax[ind].set_title(f'{time_label} compared to D0')
    ax[ind].set_ylabel('Area between curves')
    ax[ind].set_xlabel('Number of Origins')
    ax[ind].axhline(0, color='gray', linestyle='--', linewidth=linewidth)

    #Rotate the x labels for the participant plot
    ax[ind].set_xticklabels(ax[ind].get_xticklabels(), rotation=90, ha='right')

    #Put a legend in the last plot
    if ind == len(LEGEND_ORDERS) - 1:
        labels_1 = list(par_palette_3BNC117.keys())
        handles_1 = [plt.Line2D([0], [0], marker='^', color='black', markerfacecolor=par_palette_3BNC117[x], markersize=5, linewidth=linewidth) for x in labels_1]
        labels_2 = list(par_palette.keys())
        handles_2 = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=par_palette[x], markersize=5, linewidth=linewidth) for x in labels_2]
        handles = handles_1 + handles_2
        labels = labels_1 + labels_2

        handles = handles_2
        labels = labels_2
        ax[ind].legend(handles, labels, title='Participant', loc='upper left', fontsize=6, title_fontsize=6,
                    bbox_to_anchor=(1.05, 0.95), borderaxespad=0, ncol = 2, frameon=False)
    else:
        ax[ind].get_legend().remove()


plt.subplots_adjust(wspace=0.2, top =0.9, bottom=0.3, left=0.1, right=0.8)
plt.savefig(out_dir + 'area_between_curves_stripplot_no_pre.png', dpi = 300)
plt.close()