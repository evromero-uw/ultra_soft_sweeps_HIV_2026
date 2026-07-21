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
outDir = '../../../results/paper_draft_04-13/figure_5/'

# Here is the path to the data
inDir_10_1074 = '../../../data/clyde_westfall_2024_final/10-1074/'
inDir_3BNC117 = '../../../data/clyde_westfall_2024_final/3BNC117/'
sim_data_dir = '../../../data/slim_simulations/02-20-2026_sims/'
sim_dir_stub = sim_data_dir + 'origins_{origins}_min_{min_num}_rep_{rep}/analysis/FilteredGenotypes'
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
# PAR_LIST_3BNC117 = ['2C1', '2C5', '2E1', '2E2', '2E3', '2E4', '2E5', '2E7']
time_filter_out = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2']


#SIMULATION PARAMETERS
REP = range(0, 10)
ORIGIN_NUM_LIST = [1, 2, 10, 20, 50, 100]
MIN_ORIGIN_DICT= {1: 1, 2: 2, 10: 5, 20: 10, 50: 40, 100: 80}
TIME_CONVERT_DICT = {299: 'Pre', 300: 'D0', 303: 'W1', 311: 'W4', 322: 'W8'}
HXB2_ENV_LEN = 2571

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



###############################################################################
# # In vivo data 3BNC117

# # We will only take the sites with a minor allele frequency above this 
# # threshold
# ALLELE_FREQ_THRESH = 0
# # We will use all segregating alleles at each site
# MULTI_SEG = True
# hxb2_res_positions = dataset_metadata.RESISTANCE_POS_NT_HXB2

# # Make a dataframe to store the results
# all_pi_df = []

# # We will loop through each of the participants and calculate the sequence
# # diversity in a sliding window
# for curr_par in par_list:
#     print(curr_par)

#     # First, I need to load the data
#     inFile = inDir_3BNC117 + curr_par + '/835_' + curr_par + \
#                     '_NT.fasta'
#     participant_dat = Pardata(inFile, 'clyde2024', curr_par)
#     participant_dat.load_data_3BNC117(ALLELE_FREQ_THRESH, MULTI_SEG)

#     # Next, I will get a couple of items out of the dataset
#     hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
#     hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath
#     seq_info_df = participant_dat.seq_info_df.copy()
#     seq_info_df = seq_info_df[~seq_info_df['time_label'].isin(time_filter_out)]
#     seq_arr = participant_dat.seq_arr

#     seq_length = seq_arr.shape[1]

 
#     for curr_time in seq_info_df['time_label'].unique():
#         # Get the dataframe and the sequences for the current timepoint
#         curr_time_df = seq_info_df[seq_info_df['time_label'] == curr_time]


#         # Now I will calculate the sequence diversity for the current window
#         curr_div = div.calc_ave_pairwise_hamming(seq_arr, curr_time_df)
#         all_pi_df.append([curr_par, curr_time, curr_div, '3BNC117'])

all_pi_df = pd.DataFrame(all_pi_df, columns=['participant', 'time', 'pi', 'dataset'])
all_pi_df.to_csv(outDir + 'inv_pi_summary.csv', index=False)



###############################################################################
# Simulated data
out_sim_all = []

for ind_1, curr_num_origins in enumerate(ORIGIN_NUM_LIST):

    for curr_rep in REP:
        sim_file = sim_dir_stub.format(origins=curr_num_origins, 
                                       min_num=MIN_ORIGIN_DICT[curr_num_origins],
                                       rep=curr_rep)

        #Load the data
        genotype_df = pd.read_pickle(sim_file)
        sim_diversity_df = []

        #Loop through the timepoints
        for curr_time, time_group in genotype_df.groupby('timepoint'):
            #Drop any nan columns, these are sites which are not polymorphic
            #at/before this timepoint
            time_group = time_group.dropna(axis=1, how='all')
            time_cols = [col for col in time_group.columns if col not in ['individual', 'timepoint']]
            time_seqs = time_group[time_cols].to_numpy()

            if len(time_cols) == 0:
                #Set pairwise distance to 0 if no polymorphic sites
                sim_diversity_df.append([curr_rep, curr_time, 0])

            else:
                #Calculate the average pairwise distance
                #I need my letters to be integers for this to work
                letter_to_int = {'A':0, 'C':1, 'G':2, 'T':3}
                for letter, integer in letter_to_int.items():
                    time_seqs[time_seqs == letter] = integer
                time_seqs = time_seqs.astype(int)

                #Calculate the pairwise hamming distance
                hamming_dist = pdist(time_seqs, metric='hamming')

                #I need to actually normalize this by the window size
                avg_hamming_dist = hamming_dist.mean()
                norm_hamming_dist = (avg_hamming_dist * len(time_cols)) / HXB2_ENV_LEN
                
                #Add to the dataframe
                sim_diversity_df.append([curr_rep, curr_time, norm_hamming_dist])

        #Convert to a dataframe
        sim_diversity_df = pd.DataFrame(sim_diversity_df, columns=['replicate', 'timepoint', 'avg_hamming_dist'])
        sim_diversity_df['replicate'] = curr_rep
        sim_diversity_df['num_origins'] = curr_num_origins
        sim_diversity_df['min_num_origins'] = MIN_ORIGIN_DICT[curr_num_origins]
        out_sim_all.append(sim_diversity_df)

out_sim_all = pd.concat(out_sim_all, ignore_index=True)
out_sim_all.to_csv(outDir + 'summary_simulated_pi_vs_time.csv', index=False)