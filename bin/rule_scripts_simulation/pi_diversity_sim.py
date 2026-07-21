import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import dataset_metadata
from scipy.spatial.distance import pdist


# Today I am calculating a windowed diversity metric on the simulated data
# In this file, I am going to loop through the hxb2 positions and calculate
# The sequence diversity in a sliding window for each individual
WINDOW_SIZE = snakemake.params.window_size
WINDOW_STEP = snakemake.params.window_step

rep = snakemake.wildcards.rep
num_origins = snakemake.params.num_origins
min_num_origins = snakemake.params.min_num_origins


# For the simulations I introduced mutations at position 1000
HXB2_RES_POS = dataset_metadata.RESISTANCE_POS_NT_HXB2 

# Get the simulation directory
sim_file = snakemake.input[0]
hxb2_file = snakemake.params.hxb2_file
out_file = snakemake.output[0]

#Make a dataframe to save our output diversity results
out_diversity_df = []

#
with open(hxb2_file, 'r') as f:
    hxb2_seq = f.read().strip()

#Make a dataframe to hold the results
sim_diversity_df = []

#Load the data
genotype_df = pd.read_pickle(sim_file)

#Loop through the timepoints
for curr_time, time_group in genotype_df.groupby('timepoint'):
    #Drop any nan columns, these are sites which are not polymorphic
    #at/before this timepoint
    time_group = time_group.dropna(axis=1, how='all')
    #Now I want to calculate the average pairwise diversity in each sliding window
    # Now, I will loop through each timepoint
    for i in range(0, len(hxb2_seq) - WINDOW_SIZE, WINDOW_STEP):
        window_start = i
        window_end = i + WINDOW_SIZE
        #Get all of the columns in this window
        window_cols = [col for col in time_group.columns[2:] if (int(col) >= window_start) \
                    and (int(col) < window_end)]
        if len(window_cols) == 0:
            #Set pairwise distance to 0 if no polymorphic sites
            sim_diversity_df.append([rep, curr_time, 0, window_start, window_end])
        else:
            window_seqs = time_group[window_cols].to_numpy()

            #Calculate the average pairwise distance
            #I need my letters to be integers for this to work
            letter_to_int = {'A':0, 'C':1, 'G':2, 'T':3}
            for letter, integer in letter_to_int.items():
                window_seqs[window_seqs == letter] = integer
            window_seqs = window_seqs.astype(int)

            #Calculate the pairwise hamming distance
            hamming_dist = pdist(window_seqs, metric='hamming')

            #I need to actually normalize this by the window size
            avg_hamming_dist = hamming_dist.mean()
            norm_hamming_dist = (avg_hamming_dist * len(window_cols)) / WINDOW_SIZE
            
            #Add to the dataframe
            sim_diversity_df.append([rep, curr_time, norm_hamming_dist, window_start, window_end])

#Convert to a dataframe
sim_diversity_df = pd.DataFrame(sim_diversity_df, columns=['replicate', 'timepoint', 'avg_hamming_dist', 'window_start', 'window_end'])
sim_diversity_df['replicate'] = rep
sim_diversity_df['num_origins'] = num_origins
sim_diversity_df['min_num_origins'] = min_num_origins
out_diversity_df.append(sim_diversity_df)

out_diversity_df = pd.concat(out_diversity_df, ignore_index=True)
out_diversity_df.to_csv(out_file, index=False)

    

