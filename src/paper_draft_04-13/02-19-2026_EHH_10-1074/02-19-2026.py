import os
import sys
import glob


all_paths = glob.glob('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/*')
sys.path.extend(all_paths)
sys.path.append('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/')

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import linkage_stats
import data_util as du
from sklearn.metrics import auc
from par_data_class import Pardata


#This script is a snakemake rule for calculating IHH on in vivo datasets
time_filter_out = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2']

inFile = snakemake.input[0]
outDataDir = '/'.join(snakemake.output[0].split('/')[:-1]) + '/'
curr_par = snakemake.params.participant


# We will only take the sites with a minor allele frequency above this 
# threshold
ALLELE_FREQ_THRESH = 0
# Use all segregating alleles at the site
MULTI_SEG = True

CORE_FREQ_THRESH = 0.05


############################### Load the data #################################

# First, I need to load the data
participant_dat = Pardata(inFile, 'clyde2024', curr_par)
participant_dat.load_data_3BNC117(ALLELE_FREQ_THRESH, MULTI_SEG)

# Now I will get some datastructures out of the participant data object
hxb2_nuc_coords_hta = participant_dat.hxb2_nuc_coords_hta
hxb2_nuc_coords_ath = participant_dat.hxb2_nuc_coords_ath
seq_info_df = participant_dat.seq_info_df
seq_info_df = seq_info_df[~seq_info_df['time_label'].isin(time_filter_out)]
seqArr = participant_dat.seq_arr

################## Scan through the genome and calculate EHH ##################
seg_freq_dict = participant_dat.seg_freq_dict
allele_freq_df, all_time_freqs, time_sample_sizes = \
                du.seg_freq_by_timepoint(seg_freq_dict, seqArr, seq_info_df)
summary_ihh_results = []


for curr_site in seg_freq_dict:
    print('outDataDir: ' + outDataDir, file = sys.stderr)
    if not os.path.exists(outDataDir):
        os.makedirs(outDataDir)

    #Get the current allele frequency
    curr_freqs = allele_freq_df[allele_freq_df['position'] == curr_site]
    

    #Calculate the allele at higher frequency at day 0
    day_0_freqs = curr_freqs[curr_freqs['time'] == 'D0']
    day_0_freqs = day_0_freqs.sort_values('freqs', ascending=False)
    anc_allele = day_0_freqs.iloc[0]['allele']
    anc_freq = day_0_freqs.iloc[0]['freqs']

    
    # First I will label all of the core haplotypes
    focal_sites = (curr_site, curr_site)
    core_hap_labels, label_conv_dict = linkage_stats.label_ehh_core_general(
                                                        seqArr, seq_info_df,
                                                        focal_sites)

    
    # Label each dataframe line with the core haplotype
    labeled_seq_info = seq_info_df.copy()
    core_hap_labels_dict = dict(zip(range(0, len(core_hap_labels)), core_hap_labels))
    labeled_seq_info['core'] = labeled_seq_info['seq_index'].map(core_hap_labels_dict)

    # Next, I will loop through the timepoints and calculate EHH for each one
    all_ehh_results = []
    for curr_time in seq_info_df['time_label'].unique():
        # remove any cores that are singletons
        curr_labeled_info = labeled_seq_info[labeled_seq_info['time_label'] == curr_time]
        curr_core_labels = curr_labeled_info['core']
        core_counts = curr_core_labels.value_counts()
        print('core_counts: ' + str(core_counts))
        core_ignores = core_counts[core_counts <= 1]
        print('core_ignores: ' + str(core_ignores))

        # remove any cores with frequency less than the allele freq threshold
        core_freqs = core_counts / time_sample_sizes[curr_time]
        print('corr_freqs: ' + str(core_freqs))
        core_ignores = pd.concat([core_ignores, core_freqs[core_freqs < CORE_FREQ_THRESH]])
        print('core_ignores: ' + str(core_ignores))

        core_str_removed = [label_conv_dict[x] for x in core_ignores.index.tolist()]
        print('Removing cores: ' + str(core_str_removed), file = sys.stderr)
        
        curr_labeled_info = curr_labeled_info[~curr_labeled_info['core'].isin(core_ignores.index)]
        
        #Get the sequence array and core labels for the current time and cores
        curr_seq_arr = seqArr[curr_labeled_info['seq_index'], :]
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
        iHH = auc(curr_group['seg_site'], curr_group['ehh'])
        
        print(name)
        print(curr_freqs)
        print(iHH)
        sequence_at_site = seqArr[:, curr_site]
        print(np.unique(sequence_at_site, return_counts=True))
        print('**************')
        #Also get the day 0 frequency
        allele_name = name[1][0]
        if allele_name == '-':
            continue

        allele_freq = curr_freqs[curr_freqs['time'] == name[0]]
        allele_freq = allele_freq[allele_freq['allele'] == allele_name]


        allele_freq = allele_freq[allele_freq['allele'] == allele_name].iloc[0]['freqs']
        anc_bool = allele_name[0] == anc_allele
    

        all_ihh_results.append([name[0], name[1], iHH, allele_freq, anc_bool, anc_freq])

    #Now I will store all the ihh results together
    all_ihh_results = pd.DataFrame(all_ihh_results, columns = ['time_label',
                                                                'core', 'iHH',
                                                                'snp_freq', 
                                                                'anc_snp', 
                                                                'anc_freq'])
    all_ihh_results['core'] = all_ihh_results['core'].map(lambda x: x[0])
    all_ihh_results['site'] = curr_site

    print('Finished site: ' + str(curr_site), file = sys.stderr)
    summary_ihh_results.append(all_ihh_results)

summary_ihh_results = pd.concat(summary_ihh_results, ignore_index=True)
summary_ihh_results.to_csv(outDataDir +  curr_par + '_ihh_results.csv', index=False)