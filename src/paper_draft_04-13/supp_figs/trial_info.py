import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams


import dataset_metadata

from par_data_class import Pardata

#In this file I am going to make a schematic of sampled time points
#for the trial
inDir = '../../../data/clyde_westfall_2024_final/10-1074/'
vl_file = '../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
outFolder = '../../../results/paper_draft_04-13/supp_figs/trial_info/'

par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
            '1HD10K','1HD11K']

par_order = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
par_order = reversed(par_order)

params = {'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6
rcParams.update(params)

#We will only take the sites with a minor allele frequency above this threshold
ALLELE_FREQ_THRESH = 0
#We will only use the two most frequent alleles at each site
MULTI_SEG = True
#We will include DRM sites in the analysis
INCLUDE_DRM_SEG = True
###############################################################################
all_timepoint_df = []

for curr_par in par_list:
    print(curr_par)
    ################################ Data Loading #############################
    ###########################################################################
    outDir = '../../results/' + curr_par + outFolder
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
    
    #Get the sequence info dataframe so we can see if there are sgs or smrt-umi
    #samples
    seq_info_df = participant_dat.seq_info_df
    seq_info_df = seq_info_df[~seq_info_df['orig_name'].str.contains('HXB2')]
    
    #Separate the smrt-umi and sgs samples based on the naming convention
    smrt_umi_names = seq_info_df[seq_info_df['orig_name'].str.contains('885_')]
    sgs_names = seq_info_df[~seq_info_df['orig_name'].str.contains('885_')]

    #Get the time points for each
    smrt_umi_timepoints = smrt_umi_names['time_label'].to_numpy()
    smrt_umi_timepoints = list(np.unique(smrt_umi_timepoints))
    sgs_timepoints = sgs_names['time_label'].to_numpy()
    sgs_timepoints = list(np.unique(sgs_timepoints))

    both_timepoints = list(set(sgs_timepoints) & set(smrt_umi_timepoints))
    sgs_only_timepoints = list(set(sgs_timepoints) - set(both_timepoints))
    smrt_umi_only_timepoints = list(set(smrt_umi_timepoints) - set(both_timepoints))

    #Make a dataframe with the time points and the method
    timepoint_df = pd.DataFrame({'time_point': both_timepoints + sgs_only_timepoints + smrt_umi_only_timepoints,
                                 'method': ['Both'] * len(both_timepoints) + 
                                           ['SGS'] * len(sgs_only_timepoints) +
                                           ['Deep'] * len(smrt_umi_only_timepoints)})
    timepoint_df['time_point'] = [int(x[1:]) * 7 for x in timepoint_df['time_point']]
    timepoint_df = timepoint_df.sort_values(by='time_point')
    timepoint_df = timepoint_df.reset_index(drop=True)
    timepoint_df['participant'] = curr_par

    all_timepoint_df.append(timepoint_df)

all_timepoint_df = pd.concat(all_timepoint_df, ignore_index=True)

#Only include time points through week 8
all_timepoint_df = all_timepoint_df[all_timepoint_df['time_point'] <= 8 * 7]

all_timepoint_df = all_timepoint_df.pivot(index='participant', columns='time_point', values='method')
all_timepoint_df = all_timepoint_df.fillna('None')


#Now I want to make a trial schematic of the sampling at each day
fig, ax = plt.subplots(figsize=(2, 1.5))
cmap = {'SGS': 'lightcoral', 'Deep': 'lightskyblue', 'Both': 'mediumorchid'}

#Make a line at the top for the trial duration
ax.hlines(y=10, xmin=min(all_timepoint_df.columns)-10, xmax=max(all_timepoint_df.columns)+ 10, color='black',
          linewidth=linewidth)
ax.plot(all_timepoint_df.columns, [10] * len(all_timepoint_df.columns), '|', color='black', markersize=2)
#Label the days
for day in all_timepoint_df.columns:
    ax.text(day, 10.1, str(int(day/7)), ha='center', va='bottom', fontsize=fontsize)

#Label the samples gathered for each participant
#Sort the participant index so they are in order based on the participant ID
all_timepoint_df.index = pd.CategoricalIndex(all_timepoint_df.index, categories=par_order, ordered=True)
all_timepoint_df = all_timepoint_df.sort_index()

for i, par in enumerate(all_timepoint_df.index):
    for j, day in enumerate(all_timepoint_df.columns):
        method = all_timepoint_df.loc[par, day]
        if method != 'None':
            ax.scatter(day, i, color='black', s=10, edgecolor= None, linewidth=linewidth, zorder=3)
    ax.text(min(all_timepoint_df.columns)-15, i, par, ha='right', va='center', fontsize=fontsize)

#Remove the axes
ax.text((max(all_timepoint_df.columns) - min(all_timepoint_df.columns)) / 2, 12, 'Trial week', ha='center', va='bottom', fontsize=fontsize)
ax.text(min(all_timepoint_df.columns)-35, 5.5, 'Participant', ha='right', va='center', fontsize=fontsize, rotation=90)
ax.axis('off')

plt.savefig(outFolder + 'trial_schematic.png', dpi=500, bbox_inches='tight')
