import os
import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.stats import binned_statistic

import dataset_metadata
import data_util as du
import diversity_stats as div
from matplotlib import rcParams
from par_data_class import Pardata

#Today I am going to try and make plots comparing the burn in
#period to the diversity in the population against the 10-1074 cohort
params = {'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5

#Slide size params for powerpoint
# params = {'axes.labelsize': 16, 'axes.titlesize':16,  
#           'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
#           'legend.title_fontsize': 16, 'font.family': 'sans-serif',
#           'font.sans-serif': 'Arial'}

plt.rcParams.update(params)

#I'm also going to try and compare my site frequency spectra between
# the simulations and the in vivo data

sim_dir = '../../../../data/slim_simulations/sim_matching/'
in_vivo_dir = '../../../../data/clyde_westfall_2024_final/10-1074/'
zanini_dir = '../../../../data/elife-11282-fig5-data1-v2/'
out_dir = '../../../../results/paper_draft_04-13/supp_figs/sim_matching/'

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

hxb2_file = '../../../../data/hxb2_nuc_env.txt'
with open(hxb2_file, 'r') as f:
    hxb2_seq = f.read().strip()

print(os.listdir(sim_dir))


#if the data is already saved just replot it
SAVE_DATA = False
SAMPLE_TIME = 300

HXB2_RES_POS = dataset_metadata.RESISTANCE_POS_NT_HXB2 

# We will only take the sites with a minor allele frequency above this 
# threshold
ALLELE_FREQ_THRESH = 0
# We will only use the two most frequent alleles at each site
MULTI_SEG = True

PAR_LIST = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
#I'm removing 1HD7K bc they have some kind of quasi-species situation going on
# PAR_LIST = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K',
#              '1HD10K', '1HD11K']
par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD9K': '#CAB2D6',
               '1HD10K': '#6A3D9A', '1HD11K': '#B15928'}

###############################################################################
sim_diversity_df = []
sim_minor_allele_freqs = []
if SAVE_DATA and os.path.exists(out_dir + 'sim_diversity.csv'):
    print('Using previously saved data')
    sim_diversity_df = pd.read_csv(out_dir + 'sim_diversity.csv')
    sim_minor_allele_freqs = pd.read_csv(out_dir + 'sim_site_freqs.csv')
else:
    #Loop though the directory and load each simulation dataset
    #individually
    for rep in os.listdir(sim_dir):
        if rep.startswith('.'):
            continue
        print(rep)
        rep_dir = os.path.join(sim_dir, rep, 'analysis') + '/'

        #Load the data
        genotype_df = pd.read_pickle(rep_dir + 'FilteredGenotypes')

        #Loop through the timepoints
        for curr_time, time_group in genotype_df.groupby('timepoint'):
            #Drop any nan columns, these are sites which are not polymorphic
            #at/before this timepoint
            time_group = time_group.dropna(axis=1, how='all')

            #Now I want to calculate the average pairwise hamming distance
            sequence_arr = np.array(time_group.iloc[:, 2:])
            #I need my letters to be integers for this to work
            letter_to_int = {'A':0, 'C':1, 'G':2, 'T':3}
            for letter, integer in letter_to_int.items():
                sequence_arr[sequence_arr == letter] = integer
            sequence_arr = sequence_arr.astype(int)

            #Calculate the pairwise hamming distance
            hamming_dist = pdist(sequence_arr, metric='hamming')
            avg_hamming_dist = hamming_dist.mean()

            #Add to the dataframe
            sim_diversity_df.append([rep, curr_time, avg_hamming_dist])

            #Now load the site frequency spectrum data
            if curr_time == SAMPLE_TIME:
                site_freq_df = pd.read_pickle(rep_dir + 'FilteredLoci')
                site_freq_df = site_freq_df[site_freq_df['timepoint'] == SAMPLE_TIME]
                minor_allele_freqs = np.hstack((site_freq_df['freq_2'].to_numpy(),
                                            site_freq_df['freq_3'].to_numpy(), 
                                            site_freq_df['freq_4'].to_numpy()))
                #Now get rid of any zero frequency alleles
                minor_allele_freqs = minor_allele_freqs[minor_allele_freqs > 0]
                sim_minor_allele_freqs.append(pd.DataFrame({'rep': rep,
                                                        'minor_allele_freq': minor_allele_freqs}))

    sim_minor_allele_freqs = pd.concat(sim_minor_allele_freqs, ignore_index=True)
    sim_minor_allele_freqs['dataset'] = 'SLiM simulations'
    sim_minor_allele_freqs.to_csv(out_dir + 'sim_site_freqs.csv', index=False)

    sim_diversity_df = pd.DataFrame(sim_diversity_df, columns=['rep', 'timepoint', 'avg_hamming_dist'])
    sim_diversity_df['year'] = sim_diversity_df['timepoint'] / (365/2)
    sim_diversity_df.to_csv(out_dir + 'sim_diversity.csv', index=False)

###############################################################################
in_vivo_diversity_df = []
in_vivo_sfs_df = []

#Now I need to compare the burn-in period to the diversity in the in vivo trial
# We will loop through each of the participants and get their array to
# hxb2 coordinate mapping and rebound time
for curr_par in PAR_LIST:
    #Load the data
    inFile = in_vivo_dir + curr_par + '/885_' + curr_par  + '_NT_filtered.fasta'

    # First, I need to load the data
    participant_dat = Pardata(inFile, 'clyde2024', curr_par)
    participant_dat.load_data_10_1074(HXB2_RES_POS, ALLELE_FREQ_THRESH, MULTI_SEG)

    # Next, I'll get only the day 0 sequences
    seq_info_df = participant_dat.seq_info_df.copy()
    seq_arr = participant_dat.seq_arr.copy()
    seq_arr, seq_info_df = du.filter_timepoints(seq_arr, seq_info_df, set(['D0']))

    gap_col_set = du.make_gap_col_set(seq_arr, gap_thresh=0.5)
    seq_arr = np.delete(seq_arr, list(gap_col_set), axis=1)


    # Now, I need to calculate the average pairwise hamming distance
    curr_pi = div.calc_ave_pairwise_hamming(seq_arr, seq_info_df)
    in_vivo_diversity_df.append([curr_par, curr_pi])

    #Next I want to calculate the site frequency spectrum
    #The first sequence is HXB2
    seg_sites_dict = du.get_seg_sites(seq_arr[1:,], set(), ALLELE_FREQ_THRESH, MULTI_SEG)

    #Loop through each site and put it in the spectrum dataframe
    site_spectrum_df = []
    for site, freqs in seg_sites_dict.items():
        #Remove the highest frequency allele
        freqs = sorted(freqs, key = lambda x: x[1],reverse=True)[1:]
        site_spectrum_df.extend([ [site, allele, freq] for allele, freq in freqs])
    site_spectrum_df = pd.DataFrame(site_spectrum_df, columns=['site', 'allele', 'minor_allele_freq'])
    site_spectrum_df['participant'] = curr_par
    in_vivo_sfs_df.append(site_spectrum_df)
in_vivo_sfs_df = pd.concat(in_vivo_sfs_df, ignore_index=True)
in_vivo_sfs_df['dataset'] = 'In vivo (trial day 0)'
in_vivo_diversity_df = pd.DataFrame(in_vivo_diversity_df, columns=['participant', 'avg_hamming_dist'])
in_vivo_diversity_df['dataset'] = 'In vivo (trial day 0)'

#################################################################################
#Now I want to make some plots comparing the site frequency spectra


#I want to bin the data into 20 bins and get confidence intervals on each bin
rep_err_df = []
for name, rep_group in sim_minor_allele_freqs.groupby('rep'):
    bin_means, bin_edges, binnumber = binned_statistic(rep_group['minor_allele_freq'], [],
                                                    statistic='count', bins=np.arange(0, 0.6, 0.1))
    bin_centers = bin_edges[:-1]
    rep_group_df = pd.DataFrame({'rep': name,
                                'bin_center': bin_centers,
                                'bin_count': bin_means})
    rep_group_df['bin_prop'] = rep_group_df['bin_count'] / rep_group_df['bin_count'].sum()
    rep_err_df.append(rep_group_df)
rep_err_df = pd.concat(rep_err_df, ignore_index=True)

#Now I want to get 95% confidence intervals on each bin
sim_bin_summary_df = []
for name, bin_group in rep_err_df.groupby('bin_center'):
    bin_mean = bin_group['bin_prop'].mean()
    lower = np.percentile(bin_group['bin_prop'], 2.5)
    upper = np.percentile(bin_group['bin_prop'], 97.5)
    sim_bin_summary_df.append([name, bin_mean, lower, upper])
sim_bin_summary_df = pd.DataFrame(sim_bin_summary_df, columns=['bin_center', 'mean_prop', 'lower_ci', 'upper_ci'])
sim_bin_summary_df['dataset'] = 'SLiM simulations'


#Next I'll bin and combine the in vivo data
normalized_in_vivo_sfs_df = []
for name, par_group in in_vivo_sfs_df.groupby('participant'):
    bin_means, bin_edges, binnumber = binned_statistic(par_group['minor_allele_freq'], [],
                                                    statistic='count', bins=np.arange(0, 0.6, 0.1))
    bin_centers = bin_edges[:-1]
    par_group_df = pd.DataFrame({'participant': name,
                                'bin_center': bin_centers,
                                'bin_prop': bin_means,
                                'lower_ci': np.nan,
                                'upper_ci': np.nan})
    par_group_df['bin_prop'] = par_group_df['bin_prop'] / par_group_df['bin_prop'].sum()
    normalized_in_vivo_sfs_df.append(par_group_df)
normalized_in_vivo_sfs_df = pd.concat(normalized_in_vivo_sfs_df, ignore_index=True)

fig, ax = plt.subplots(1, 1, figsize=(2.43,2))
#Slide size param
# fig, ax = plt.subplots(1, 1, figsize=(6,4))
plt.bar(sim_bin_summary_df['bin_center'] - 0.005, sim_bin_summary_df['mean_prop'], width=0.005, 
        yerr=[sim_bin_summary_df['mean_prop'] - sim_bin_summary_df['lower_ci'], 
              sim_bin_summary_df['upper_ci'] - sim_bin_summary_df['mean_prop']],
        label='SLiM simulations', color='black', capsize=2, alpha = 0.5,
        error_kw={'elinewidth':0.5})

for i, par in enumerate(normalized_in_vivo_sfs_df['participant'].unique()):
    par_data = normalized_in_vivo_sfs_df[normalized_in_vivo_sfs_df['participant'] == par]
    plt.bar(par_data['bin_center'] + 0.005*(i), par_data['bin_prop'], color= par_palette[par], alpha = 1,
            width=0.005, label=par)
plt.legend(bbox_to_anchor=(1, 1), loc='upper right', ncol = 2, columnspacing = 0.5)
plt.subplots_adjust(bottom= 0.15, left=0.15, right=0.95)
plt.xlabel('Allele frequency', labelpad=0.6)
plt.ylabel('Proportion of segregating sites', labelpad=0.6)
# plt.title(f'Allele frequency spectrum at generation {SAMPLE_TIME}\n('+ r'$\approx$' + f' {SAMPLE_TIME/(365/2):.1f} years)',
#           pad=4)
plt.savefig(out_dir + f'site_freq_spectrum_day0.png', dpi=300)
plt.close()


#################################################################################
#Now I want to make some plots comparing the diversity through time

#Figure out what generations produced the mean diversity matching in vivo
in_vivo_mean = in_vivo_diversity_df['avg_hamming_dist'].mean()
sim_gen_avg = sim_diversity_df.groupby('timepoint')['avg_hamming_dist'].mean().reset_index()
sim_gen_avg['diff_from_in_vivo'] = np.abs(sim_gen_avg['avg_hamming_dist'] - in_vivo_mean)
best_gen = sim_gen_avg.loc[sim_gen_avg['diff_from_in_vivo'].idxmin()]
print(f'Simulation generation with mean diversity closest to in vivo: {best_gen["timepoint"]} ({best_gen["timepoint"]/ (365/2):.2f} years)')


fig, ax = plt.subplots(1, 2, sharey=True, figsize=(2.5,2))
#Slide size param
# fig, ax = plt.subplots(1, 2, sharey=True, figsize=(7, 4.25))
bin_means, bin_edges, binnumber = binned_statistic(sim_diversity_df['year'], sim_diversity_df['avg_hamming_dist'], statistic='mean', bins=np.arange(0,3.5,0.5))

bin_centers = 0.5*(bin_edges[1:] + bin_edges[:-1])

for i in range(sim_diversity_df['rep'].nunique()):
    rep_data = sim_diversity_df[sim_diversity_df['rep'] == sim_diversity_df['rep'].unique()[i]]

    ax[0].plot(rep_data['year'], rep_data['avg_hamming_dist'], color='black', alpha=0.1, linewidth=linewidth)
# ax[0].plot(bin_centers, bin_means, color='gray', linewidth=linewidth * 2, label='Simulation\nmean')
# ax[0].legend()

ax[0].set_xlabel('Timepoint (years)', labelpad=0.6)
ax[0].set_ylabel('Average pairwise hamming distance', labelpad=0.6)
ax[0].set_title('SLiM Simulations', pad=4)



print('made new legend')


#Make a transparent boxplot for the in vivo data
sns.boxplot(data=in_vivo_diversity_df, y='avg_hamming_dist', ax=ax[1], fill = False, color='black',
            linewidth=linewidth)
sns.stripplot(data=in_vivo_diversity_df, y='avg_hamming_dist', ax=ax[1], hue='participant',
              palette=par_palette, size=3, edgecolor='none', jitter=0.3)
ax[1].set_xlabel('In vivo (trial day 0)', labelpad=0.6)
ax[1].set_ylabel('Average pairwise hamming distance', labelpad=0.6)
ax[1].set_title('Caskey et al., 2017', pad=4)
ax[0].set_xlim(0,3)

#Turn off the legend
ax[1].get_legend().remove()

plt.subplots_adjust(bottom= 0.1, left=0.15, right=0.95)
plt.tight_layout()
plt.savefig(out_dir + 'sim_vs_in_vivo_diversity.png', dpi=300)
plt.close()



