import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import data_util as du
import dataset_metadata
import diversity_stats as div
from par_data_class import Pardata

# In this file, I am going to loop through the hxb2 positions and calculate
# The sequence diversity in a sliding window for each individual
WINDOW_SIZE = 60
WINDOW_STEP = 20

# Here is the output folder
outDir = '../../../results/paper_draft_04-13/figure_2/'

# Here is the path to the data
inDir = '../../../data/clyde_westfall_2024_final/10-1074/'
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD9K', '1HD10K', '1HD11K']
time_filter_out = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2']


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
    inFile = inDir + curr_par + '/885_' + curr_par  + '_NT_filtered.fasta'

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

    # Now, I will loop through each timepoint
    for i in range(0, seq_length - WINDOW_SIZE, WINDOW_STEP):
        # Get the sequences in the current window
        curr_window_arr = seq_arr[:, i:i + WINDOW_SIZE]
 
        for curr_time in seq_info_df['time_label'].unique():
     
                
            # Get the dataframe and the sequences for the current timepoint
            curr_time_df = seq_info_df[seq_info_df['time_label'] == curr_time]


            # Now I will calculate the sequence diversity for the current window
            curr_div = div.calc_ave_pairwise_hamming(curr_window_arr, curr_time_df)
            win_start_hxb2 = hxb2_nuc_coords_ath[i]
            win_end_hxb2 = hxb2_nuc_coords_ath[i + WINDOW_SIZE]
            all_pi_df.append([curr_par, curr_time, win_start_hxb2, win_end_hxb2, i, curr_div])

all_pi_df = pd.DataFrame(all_pi_df, columns=['participant', 'time', 
                                             'hxb2_start', 'hxb2_end', 'arr_start', 'pi'])
all_pi_df.to_csv(outDir + '10-1074_pi_vs_time.csv', index=False)