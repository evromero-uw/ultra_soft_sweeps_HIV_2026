import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd

import r2Analysis as r2

#Today I am calculating the change in D' before and after the simulated sweeps

#We will only take the sites with a minor allele frequency above this threshold
# ALLELE_FREQ_THRESH = 0.05
ALLELE_FREQ_THRESH = snakemake.params.allele_freq_thresh
rep = snakemake.wildcards.rep
num_origins = snakemake.params.num_origins
min_num_origins = snakemake.params.min_num_origins

#Each of the files we'll need
sim_file = snakemake.input[0]
hxb2_file = snakemake.params.hxb2_file
out_file = snakemake.output[0]

with open(hxb2_file, 'r') as f:
    hxb2_seq = f.read().strip()

all_linkage_df = []

#Load the data
genotype_df = pd.read_pickle(sim_file)

#Make a place to hold the D' values
d_prime_df = []

#Separate the data by timepoint
for curr_time, time_group in genotype_df.groupby('timepoint'):
    #Drop the all nan columns (sites which are invariant at this time)
    time_group = time_group.dropna(axis=1, how='all')

    #I need to filter out sites with a low minor allele frequency
    site_cols = time_group.columns[2:-1]
    for site in site_cols:
        site_df = time_group[site].value_counts(normalize=True)
        if site_df.shape[0] < 2:
            time_group = time_group.drop(columns=[site])
            continue
        if site_df.iloc[1] < ALLELE_FREQ_THRESH:
            time_group = time_group.drop(columns=[site])
            continue
    
    
    #Loop through all pairs of sites in the group
    site_cols = time_group.columns[2:-1]
    for i in range(len(site_cols)):
        site1 = site_cols[i]

        #I need to get the more and less frequent alleles at this site
        site1_df_counts = time_group[site1].value_counts(normalize=True)
        allele_A = site1_df_counts.index[0]
        allele_a = site1_df_counts.index[1]
        allele_A_freq = site1_df_counts.iloc[0]
        allele_a_freq = site1_df_counts.iloc[1]

        for j in range(i+1, len(site_cols)):
            site2 = site_cols[j]

            #I need to get the more and less frequent alleles at this site
            site2_df_counts = time_group[site2].value_counts(normalize=True)
            allele_B = site2_df_counts.index[0]
            allele_b = site2_df_counts.index[1]
            allele_B_freq = site2_df_counts.iloc[0]
            allele_b_freq = site2_df_counts.iloc[1]

            #Get the genotypes at these two sites and calc D'
            sites_1_2_df = time_group[[site1, site2]]
            curr_haps_unique = sites_1_2_df.groupby([site1, site2]).size().reset_index(name='counts')
            curr_haps_unique_np = curr_haps_unique[[site1, site2]].to_numpy()
                
            #AB
            if [allele_A, allele_B] not in curr_haps_unique_np.tolist():
                AB_count = 0
            else:
                AB_count = curr_haps_unique.loc[(curr_haps_unique[site1] == allele_A) & (curr_haps_unique[site2] == allele_B), 'counts'].values[0]

            #Ab
            if [allele_A, allele_b] not in curr_haps_unique_np.tolist():
                Ab_count = 0
            else:
                Ab_count = curr_haps_unique.loc[(curr_haps_unique[site1] == allele_A) & (curr_haps_unique[site2] == allele_b), 'counts'].values[0]
            
            #aB
            if [allele_a, allele_B] not in curr_haps_unique_np.tolist():
                aB_count = 0
            else:
                aB_count = curr_haps_unique.loc[(curr_haps_unique[site1] == allele_a) & (curr_haps_unique[site2] == allele_B), 'counts'].values[0]
            
            #ab
            if [allele_a, allele_b] not in curr_haps_unique_np.tolist():
                ab_count = 0
            else:
                ab_count = curr_haps_unique.loc[(curr_haps_unique[site1] == allele_a) & (curr_haps_unique[site2] == allele_b), 'counts'].values[0]
            
            #put all of the D' values in a dataframe
            r_squared = r2.r2(AB_count, Ab_count, aB_count, ab_count)
            d_prime = r2.calc_D_prime(AB_count, Ab_count, aB_count, ab_count)
            d_val = r2.calc_D(AB_count, Ab_count, aB_count, ab_count)

            distance = abs(int(site2) - int(site1))


            d_prime_df.append([site1, site2, distance, allele_A_freq, allele_B_freq,
                        AB_count, Ab_count, aB_count, ab_count,
                        r_squared, d_prime, d_val, curr_time])
d_prime_df = pd.DataFrame(d_prime_df, columns = ['pos1', 'pos2', 'distance',
                                'allele_A_freq', 'allele_B_freq',
                                'AB_count', 'Ab_count', 'aB_count', 'ab_count',
                                'r_squared', 'd_prime', 'd_val', 'timepoint'])
d_prime_df['replicate'] = rep
d_prime_df['num_origins'] = num_origins
d_prime_df['min_num_origins'] = min_num_origins

d_prime_df.to_csv(out_file, index=False)
