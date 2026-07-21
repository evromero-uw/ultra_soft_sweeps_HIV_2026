import os
import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import data_util as du
import dataset_metadata

import r2Analysis as r2
from par_data_class import Pardata

#Today I am going to rerun the D' analysis, but try filtering using the same 
#conditions that Zanini et al used, so that I can have a direct comparison

inDir = '../../../../data/clyde_westfall_2024_final/10-1074/'
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
time_filter_out = ['HXB2', 'PR', 'Rebound', 'screen', 'pre', 'Nadir']
outDirAll = '../../../../results/paper_draft_04-13/supp_figs/sim_matching/'
if not os.path.exists(outDirAll):
        os.makedirs(outDirAll)

#We will only take the sites with a minor allele frequency above this threshold
ALLELE_FREQ_THRESH = 0.05
# We will use all segregating alleles at each site
MULTI_SEG = True
#We will include DRM sites in the analysis
INCLUDE_DRM_SEG = True

###############################################################################
all_linkage_df = []

for curr_par in par_list:
    print(curr_par)
    ################################ Data Loading #############################
    ###########################################################################
    inFile = inDir + curr_par + '/885_' + curr_par  + \
                    '_NT_filtered.fasta'

    #This file just makes a plot of a heatmap from a fasta file.
    #The path to the data and the resistance positions in hxb2 coordinates
    hxb2_res_positions = dataset_metadata.RESISTANCE_POS_NT_HXB2

    #Construct the data object and load the data
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)

    #Load the data including all minor variants
    participant_dat.load_data_10_1074(hxb2_res_positions, ALLELE_FREQ_THRESH, MULTI_SEG,
                              INCLUDE_DRM_SEG)

    #Get the sequence array and the sequence information
    seq_arr = participant_dat.seq_arr
    seq_info_df = participant_dat.seq_info_df
    seq_info_df = seq_info_df[~seq_info_df['time_label'].isin(time_filter_out)]
    seq_info_df = seq_info_df.copy()

    #Loop through the segregating sites
    seg_freq_dict = participant_dat.seg_freq_dict

    #Load the sequence information
    seq_info_df['time_label_int'] = [int(x[1:]) for x in \
                                     seq_info_df['time_label'].values]

    #Calculate the rebound time
    time_points = seq_info_df['time_label_int'].unique()
    time_points = [int(x) for x in time_points]
    time_points = np.sort(time_points)
    time_points = np.asarray(time_points)


    #Calculate allele frequencies by timepoint
    allele_freq_df, all_time_freqs, time_sample_sizes = \
        du.seg_freq_by_timepoint(seg_freq_dict, seq_arr, seq_info_df)

    ###########################################################################
    ######################## Calculating the Linkage ##########################
    d_prime_df = []

    #Loop through the timepoints
    for curr_time in time_points:
        allele_freq_df['time_label_int'] = [int(x[1:]) for x in \
                                        allele_freq_df['time'].values]
        #Get the sequences at the timepoint
        curr_allele_freq_df = allele_freq_df[
                                    allele_freq_df['time_label_int'] == curr_time]
        
        
        #We'll need to loop through pairs of sites and calculate D'
        #I'll be using the 06-05-2023.py file as a reference for this
        seg_positions = list(seg_freq_dict.keys())
        seg_positions = np.sort(seg_positions)


        for i in range(len(seg_positions)):
            pos1 = seg_positions[i]
            #Get the sequences at the two sites, skipping row 0 since that
            #is the reference sequence
            seq1 = seq_arr[1:, pos1]
            pos1_allele_freq_df = curr_allele_freq_df[
                                        curr_allele_freq_df['position'] == pos1]
            
            #Get the alleles and their frequencies at the two sites
            allele_A = seg_freq_dict[pos1][0][0]
            allele_a = seg_freq_dict[pos1][1][0]
            allele_A_freq = pos1_allele_freq_df[
                        pos1_allele_freq_df['allele'] == allele_A]['freqs'].values[0]
            allele_a_freq = pos1_allele_freq_df[
                        pos1_allele_freq_df['allele'] == allele_a]['freqs'].values[0]
            
            #If the allele frequency is below the threshold, skip this site
            if allele_A_freq < ALLELE_FREQ_THRESH or allele_a_freq < ALLELE_FREQ_THRESH:
                continue

            for j in range(i+1, len(seg_positions)):
                pos2 = seg_positions[j]
                #Get the sequences at the two sites, skipping row 0 since that
                #is the reference sequence
                seq2 = seq_arr[1:, pos2]
                pos2_allele_freq_df = curr_allele_freq_df[
                                        curr_allele_freq_df['position'] == pos2]
                
                #Get the alleles and their frequencies at the two sites
                allele_B = seg_freq_dict[pos2][0][0]
                allele_b = seg_freq_dict[pos2][1][0]
                allele_B_freq = pos2_allele_freq_df[
                            pos2_allele_freq_df['allele'] == allele_B]['freqs'].values[0]
                allele_b_freq = pos2_allele_freq_df[
                            pos2_allele_freq_df['allele'] == allele_b]['freqs'].values[0]         

                #If the allele frequency is below the threshold, skip this site
                if allele_B_freq < ALLELE_FREQ_THRESH or allele_b_freq < ALLELE_FREQ_THRESH:
                    continue

                #Now horizontally stack the sequences
                stacked_seqs = np.column_stack((seq1, seq2))

                #Get the unique haplotypes
                curr_haps_unique, curr_haps_counts = np.unique(stacked_seqs, 
                                                        axis = 0, return_counts=True)
                
                #AB
                if [allele_A, allele_B] not in curr_haps_unique.tolist():
                    AB_count = 0
                else:
                    AB_index = np.where((curr_haps_unique[:,0] == allele_A) & (curr_haps_unique[:,1] == allele_B))[0][0]
                    AB_count = curr_haps_counts[AB_index]

                #Ab
                if [allele_A, allele_b] not in curr_haps_unique.tolist():
                    Ab_count = 0
                else:
                    Ab_index = np.where((curr_haps_unique[:,0] == allele_A) & (curr_haps_unique[:,1] == allele_b))[0][0]
                    Ab_count = curr_haps_counts[Ab_index]
                
                #aB
                if [allele_a, allele_B] not in curr_haps_unique.tolist():
                    aB_count = 0
                else:
                    aB_index = np.where((curr_haps_unique[:,0] == allele_a) & (curr_haps_unique[:,1] == allele_B))[0][0]
                    aB_count = curr_haps_counts[aB_index]
                
                #ab
                if [allele_a, allele_b] not in curr_haps_unique.tolist():
                    ab_count = 0
                else:
                    ab_index = np.where((curr_haps_unique[:,0] == allele_a) & (curr_haps_unique[:,1] == allele_b))[0][0]
                    ab_count = curr_haps_counts[ab_index]
                
                #put all of the D' values in a dataframe
                r_squared = r2.r2(AB_count, Ab_count, aB_count, ab_count)
                d_prime = r2.calc_D_prime(AB_count, Ab_count, aB_count, ab_count)
                d_val = r2.calc_D(AB_count, Ab_count, aB_count, ab_count)

                distance = du.calc_bp_dist(pos1, pos2, seq_arr)
                
                d_prime_df.append([pos1, pos2, distance, allele_A_freq, allele_B_freq,
                            AB_count, Ab_count, aB_count, ab_count,
                            r_squared, d_prime, d_val, curr_time])

    d_prime_df = pd.DataFrame(d_prime_df, columns = ['Locus_1', 'Locus_2', 'Distance', 'p_A', 'p_B',
                    'AB_obs', 'Ab_obs', 'aB_obs', 'ab_obs', 'r_squared', 'd_prime', 'd_val', 'time_label_int'])  
    d_prime_df['participant'] = curr_par
    all_linkage_df.append(d_prime_df)


all_linkage_df = pd.concat(all_linkage_df, ignore_index=True)
all_linkage_df.to_csv(outDirAll + 'linkage_info_rare_alleles.csv', index = False)