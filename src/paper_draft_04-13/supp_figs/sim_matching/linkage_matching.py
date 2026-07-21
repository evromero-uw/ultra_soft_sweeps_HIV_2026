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

import r2Analysis as r2


#We will only take the sites with a minor allele frequency above this threshold
ALLELE_FREQ_THRESH = 0.05

#The number of distance bins to use
NUM_BINS = 5
NUM_BOOTSTRAPS = 1000

SAMPLE_TIME = 300

SAVED_DATA = True

par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD9K': '#CAB2D6',
               '1HD10K': '#6A3D9A', '1HD11K': '#B15928'}

# #Slide size params for powerpoint
# params = {'figure.figsize': (8, 4), 'axes.labelsize': 16, 'axes.titlesize':16,  
#           'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
#           'legend.title_fontsize': 16, 'font.family': 'sans-serif',
#           'font.sans-serif': 'Arial'}
params = {'figure.figsize': (2.15, 2), 'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6
rcParams.update(params)



sim_dir_list = ['../../../../data/slim_simulations/sim_matching/']

in_vivo_dir = '../../../../data/clyde_westfall_2024_final/10-1074/'
out_dir = '../../../../results/paper_draft_04-13/supp_figs/sim_matching/'

inv_linkage_file = out_dir + 'linkage_info_rare_alleles.csv'

hxb2_file = '../../../../data/hxb2_nuc_env.txt'
with open(hxb2_file, 'r') as f:
    hxb2_seq = f.read().strip()

all_linkage_df = []
if SAVED_DATA:
    print("Loading saved data")
    all_linkage_df = pd.read_csv(out_dir + 'sim_matching_linkage.csv')
else:
    for sim_dir in sim_dir_list:
        #Loop though the directory and load each simulation dataset
        #individually
        for rep in os.listdir(sim_dir):
            if rep.startswith('.'):
                continue

            rep_dir = os.path.join(sim_dir, rep, 'analysis') + '/'

            #Load the data
            genotype_df = pd.read_pickle(rep_dir + 'FilteredGenotypes')

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
            all_linkage_df.append(d_prime_df)
    all_linkage_df = pd.concat(all_linkage_df, ignore_index=True)
    print(all_linkage_df.head())
    all_linkage_df.to_csv(out_dir + 'sim_matching_linkage.csv', index=False)

print(all_linkage_df.head())

linkage_summary_df = []
#Bin the distances and get the mean D' in each bin
for timepoint, group in all_linkage_df.groupby('timepoint'):
    if timepoint != SAMPLE_TIME:
        continue

    #Drop na values
    group = group.dropna(subset=['d_prime'])

    #Print the number of pairs of sites
    print(f"Timepoint {timepoint} has {group.shape[0]} pairs of sites")

    #Bin the D' values to view as an average
    binned_rat, binedges, bin_nums = binned_statistic(
        group['distance'].to_numpy(), 
        group['d_prime'].to_numpy(), bins = NUM_BINS)
    
    #Get the midpoints of the bins
    bin_midpoints = [(binedges[i] + binedges[i+1])/2 for i in range(len(binedges)-1)]



    #Now bootstrap confidence intervals on each of the linkage curves
    bootstrap_df = []
    for j in range(NUM_BOOTSTRAPS):
        boostrap_sample = group.sample(frac=1, replace=True)
        binned_stat = binned_statistic(
            boostrap_sample['distance'].to_numpy(), 
            boostrap_sample['d_prime'].to_numpy(), bins = NUM_BINS)
        bootstrap_df.append(pd.DataFrame({'distance': bin_midpoints,
                                            'd_prime': binned_stat[0]}))
    bootstrap_df = pd.concat(bootstrap_df, ignore_index=True)

    #Get the 95% confidence intervals on each bin
    conf_intervals = bootstrap_df.groupby('distance').quantile([0.025, 0.975]).unstack()
    conf_intervals.columns = conf_intervals.columns.droplevel()


    linkage_summary_df.append(pd.DataFrame({'distance': bin_midpoints,
                                        'd_prime': binned_rat,
                                        'timepoint': timepoint,
                                        'lower': conf_intervals[0.025],
                                        'upper': conf_intervals[0.975]}))
    
    #Now we'll plot the data
    palette_dict = sns.color_palette("viridis", n_colors=len(all_linkage_df['timepoint'].unique()))
    palette_dict = {tp: palette_dict[i] for i, tp in enumerate(sorted(all_linkage_df['timepoint'].unique()))}
    plt.plot(bin_midpoints, binned_rat, label='SLiM simulations',
                    color ='gray', linewidth= 1.5 * linewidth,
                    linestyle = 'dashed')
    plt.fill_between(bin_midpoints, conf_intervals[0.025],
                            conf_intervals[0.975], alpha=0.2,
                            color = 'gray')
#Make a plot of D' vs distance for the simulations and the in vivo data
# sns.lineplot(data=all_linkage_df, x='distance', y='d_prime', errorbar='sd', estimator='mean',
#              hue='timepoint', palette='viridis', lw=linewidth)


#Now plot the in vivo data for comparison
print("Loading in vivo data")
inv_linkage_df = pd.read_csv(inv_linkage_file)
inv_linkage_df = inv_linkage_df[inv_linkage_df['time_label_int'] == 0]
print(inv_linkage_df.head())

#Now we'll group and summarize the statistics by time point
for curr_par, group in inv_linkage_df.groupby('participant'):
    #Drop the na values
    group = group.dropna(subset=['d_prime'])

    #Bin the D' values to view as an average
    binned_rat, binedges, bin_nums = binned_statistic(
        group['Distance'].to_numpy(), 
        group['d_prime'].to_numpy(), bins = NUM_BINS)
    
    #Get the midpoints of the bins
    bin_midpoints = [(binedges[i] + binedges[i+1])/2 for i in range(len(binedges)-1)]

    #Now bootstrap confidence intervals on each of the linkage curves
    bootstrap_df = []
    for j in range(NUM_BOOTSTRAPS):
        boostrap_sample = group.sample(frac=1, replace=True)
        binned_stat = binned_statistic(
            boostrap_sample['Distance'].to_numpy(), 
            boostrap_sample['d_prime'].to_numpy(), bins = NUM_BINS)
        bootstrap_df.append(pd.DataFrame({'Distance': bin_midpoints,
                                            'd_prime': binned_stat[0]}))
    bootstrap_df = pd.concat(bootstrap_df, ignore_index=True)

    #Get the 95% confidence intervals on each bin
    conf_intervals = bootstrap_df.groupby('Distance').quantile([0.025, 0.975]).unstack()
    conf_intervals.columns = conf_intervals.columns.droplevel()

        
    linkage_summary_df.append(pd.DataFrame({'Distance': bin_midpoints,
                                        'd_prime': binned_rat,
                                        'participant': curr_par,
                                        'lower': conf_intervals[0.025],
                                        'upper': conf_intervals[0.975]}))
    
    #Now we'll plot the data
    plt.plot(bin_midpoints, binned_rat, label=curr_par,
                    color = par_palette[curr_par], linewidth=linewidth)
    plt.fill_between(bin_midpoints, conf_intervals[0.025],
                            conf_intervals[0.975], alpha=0.2, 
                            color = par_palette[curr_par])
    
plt.xlabel('Distance between loci (nt)', fontsize=fontsize, labelpad=0.6)
plt.ylabel("Linkage disequilibrium (D')", fontsize=fontsize, labelpad=0.6)

plt.subplots_adjust(bottom= 0.15, left=0.15, right=0.95)
plt.savefig(out_dir + 'sim_matching_linkage.png', dpi=300)