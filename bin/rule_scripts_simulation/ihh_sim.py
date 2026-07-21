import os
import sys
import glob

all_paths = glob.glob('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/*')
sys.path.extend(all_paths)
sys.path.append('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/')

import numpy as np
import pandas as pd

import linkage_stats
import data_util as du
from sklearn.metrics import auc


#In this file I am going to be calculating EHH on the simulated data

#Testing files
inDir = "/net/feder/vol1/home/evromero/2025_hiv_linkage/data/slim_simulations/12-11-2025_sims/origins_10_min_5_rep_0/"
inFile_genotypes = inDir + 'analysis/FilteredGenotypes'
inFile_loci = inDir + 'analysis/FilteredLoci'
hxb2_file = '/net/feder/vol1/home/evromero/2025_hiv_linkage/data/hxb2_nuc_env.txt'
with open(hxb2_file, 'r') as f:
    hxb2_seq = f.read().strip()

inFile_genotypes = snakemake.input[0]
inFile_loci = snakemake.input[1]
hxb2_file = snakemake.params.hxb2_file
out_file = snakemake.output[0]



# We will only take the sites with a minor allele frequency above this 
# threshold
ALLELE_FREQ_THRESH = 0
# Use all segregating alleles at the site
MULTI_SEG = True

CORE_FREQ_THRESH = 0.05


############################### Load the data #################################

# First, I need to load the data
genotype_df = pd.read_pickle(inFile_genotypes)
loci_df = pd.read_pickle(inFile_loci)

# Create an array to hold the final sequences with hxb2 alleles filled in
hxb2_arr = np.array(list(hxb2_seq))
seq_arr = np.vstack([hxb2_arr] * len(genotype_df))


# Fill the sequence array with any segregating alleles from the genotype array
for col in genotype_df.columns:
    if col in ['individual', 'timepoint']:
        continue
    genotype_alleles = genotype_df[col].to_numpy()
    genotype_alleles_nan = ~pd.isna(genotype_alleles)
    seq_arr[genotype_alleles_nan, int(col)] = genotype_alleles[genotype_alleles_nan]


# Now make the sequence info dataframe
seq_info_df = genotype_df[['individual', 'timepoint']].copy()
seq_info_df = seq_info_df.reset_index().rename(columns={'index': 'seq_index',
                                                        'individual': 'orig_name',
                                                        'timepoint': 'time_label'})

# Lastly get the sample sizes at each timepoint
seg_freq_dict = du.get_seg_sites(seq_arr, {}, 
                                    allele_freq_thresh= ALLELE_FREQ_THRESH,
                                    return_multi= MULTI_SEG)
allele_freq_df, all_time_freqs, time_sample_sizes = \
                du.seg_freq_by_timepoint(seg_freq_dict, seq_arr, seq_info_df)

################## Scan through the genome and calculate EHH ##################
summary_ihh_results = []

for curr_site in loci_df['position'].unique():
    # print('Starting site: ' + str(curr_site), file = sys.stderr)

    # Check if any alleles are at high enough frequency to calculate EHH in the sample
    # This can happen when an allele is at low frequency in the population but not sampled
    if curr_site not in seg_freq_dict:
        print('No segregating sampled alleles at site: ' + str(curr_site), file = sys.stderr)
        continue

  # First I will label all of the core haplotypes
    focal_sites = (curr_site, curr_site)
    core_hap_labels, label_conv_dict = linkage_stats.label_ehh_core_general(
                                                        seq_arr, seq_info_df,
                                                        focal_sites)

    
    # Label each dataframe line with the core haplotype
    labeled_seq_info = seq_info_df.copy()
    core_hap_labels_dict = dict(zip(range(0, len(core_hap_labels)), core_hap_labels))
    labeled_seq_info['core'] = labeled_seq_info['seq_index'].map(core_hap_labels_dict)

    # Next, I will loop through the timepoints and calculate EHH for each one
    all_ehh_results = []
    site_freqs = allele_freq_df[allele_freq_df['position'] == curr_site]

    for curr_time in seq_info_df['time_label'].unique():
        # remove any cores that are singletons
        curr_labeled_info = labeled_seq_info[labeled_seq_info['time_label'] == curr_time]
        curr_core_labels = curr_labeled_info['core']
        core_counts = curr_core_labels.value_counts()
        # print('core_counts: ' + str(core_counts))
        core_ignores = core_counts[core_counts <= 1]
        # print('core_ignores: ' + str(core_ignores))

        # remove any cores with frequency less than the allele freq threshold
        core_freqs = core_counts / time_sample_sizes[curr_time]
        # print('corr_freqs: ' + str(core_freqs))
        core_ignores = pd.concat([core_ignores, core_freqs[core_freqs < CORE_FREQ_THRESH]])
        # print('core_ignores: ' + str(core_ignores))

        core_str_removed = [label_conv_dict[x] for x in core_ignores.index.tolist()]
        # print('Removing cores: ' + str(core_str_removed), file = sys.stderr)
        
        curr_labeled_info = curr_labeled_info[~curr_labeled_info['core'].isin(core_ignores.index)]
        

        #Get the sequence array and core labels for the current time and cores
        curr_seq_arr = seq_arr[curr_labeled_info['seq_index'], :]
        curr_core_labels = core_hap_labels[curr_labeled_info['seq_index']]


        # Now I will calculate EHH for each core haplotype
        curr_seg_sites = du.get_seg_sites(curr_seq_arr, set(), ALLELE_FREQ_THRESH,
                                            MULTI_SEG)
        if curr_seg_sites == {}:
            continue

        curr_ehh_results = linkage_stats.calculate_EHH(curr_seq_arr,
                                                        curr_core_labels,
                                                        curr_seg_sites,
                                                        focal_pos = focal_sites)
        curr_ehh_results['time_label'] = curr_time
        all_ehh_results.append(curr_ehh_results)


    all_ehh_results = pd.concat(all_ehh_results, ignore_index=True)
    all_ehh_results['core'] = all_ehh_results['core'].map(label_conv_dict)


    #Next, I need to calculate the IHH
    all_ihh_results = []
    for name, group in all_ehh_results.groupby(['time_label', 'core']):
        curr_group = group.sort_values('seg_site')

        # We need to add a nan if there aren't enough points to calculate the
        # area under the curve
        if len(curr_group) < 2:
            iHH = np.nan
        else:
            iHH = auc(curr_group['seg_site'], curr_group['ehh'])
        

        #Also get the core allele's frequency
        allele_name = name[1][0]
        if allele_name == '-':
            continue
        
        
        allele_freq = site_freqs[site_freqs['time'] == name[0]]
        allele_freq = allele_freq[allele_freq['allele'] == allele_name].iloc[0]['freqs']
    

        all_ihh_results.append([name[0], name[1], iHH, allele_freq])

    #Now I will store all the ihh results together
    all_ihh_results = pd.DataFrame(all_ihh_results, columns = ['time_label',
                                                                'core', 'iHH',
                                                                'snp_freq'])
    all_ihh_results['core'] = all_ihh_results['core'].map(lambda x: x[0])
    all_ihh_results['site'] = curr_site

    summary_ihh_results.append(all_ihh_results)

summary_ihh_results = pd.concat(summary_ihh_results, ignore_index=True)
summary_ihh_results.to_csv(out_file, index=False)