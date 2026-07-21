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
#In this file I am also going to save the data for the closest post nadir timepoint

#In this file I am going to make a plot of the escape mutations at weeks 4/8
#for each participant
inDir = '../../../data/clyde_westfall_2024_final/10-1074/'
vl_file = '../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
outFolder = '../../../results/paper_draft_04-13/figure_1/'

par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
            '1HD10K','1HD11K']
par_order = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
time_filter_out = ['HXB2', 'PR', 'Rebound', 'screen', 'pre', 'Nadir']

params = {'figure.figsize': (2.25, 1.5), 'axes.labelsize': 6, 'axes.titlesize': 6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6

# Slide size params for powerpoint
# params = {'figure.figsize': (5, 3), 'axes.labelsize': 16, 'axes.titlesize':16,
#           'legend.fontsize': 16, 'xtick.labelsize': 16, 'ytick.labelsize': 16,
#           'legend.title_fontsize': 16, 'font.family': 'sans-serif',
#           'font.sans-serif': 'Arial'}
# linewidth = 1
# fontsize = 18

rcParams.update(params)

#We will only take the sites with a minor allele frequency above this threshold
ALLELE_FREQ_THRESH = 0.2
#We will only use the two most frequent alleles at each site
MULTI_SEG = True
#We will include DRM sites in the analysis
INCLUDE_DRM_SEG = True

###############################################################################
nadir_dict = {}
all_post_seqs = []

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

    #Load the viral load data
    vl_df = pd.read_csv(vl_file)

    #Calculate the nadir time
    nadir_time = vl_df[vl_df['Study_ID'] == curr_par]['Nadir'].values[0]
    time_points = seq_info_df['time_label_int'].unique()
    time_points = [int(x) for x in time_points]
    time_points = np.sort(time_points)
    time_points = np.asarray(time_points)


    if isinstance(nadir_time, str):
        nadir_time_int = int(nadir_time[1:])

        #We need to find the timepoint that corresponds to the nadir time
        closest_ind = (np.abs(time_points - nadir_time_int)).argmin()

        
        if closest_ind == len(time_points):
            closest = np.nan
            closest_post = np.nan
        else:
            if time_points[closest_ind] < nadir_time_int:
                print('Closest post nadir timepoint for {} is {}'.format(curr_par, time_points[closest_ind]))
                print(time_points)
                print('Nadir time is {}'.format(nadir_time_int))
                closest = time_points[closest_ind + 1]
                
                closest_post = closest
            else:
                closest = time_points[closest_ind]
                closest_post = time_points[closest_ind + 1] if closest_ind + 1 < len(time_points) else np.nan
    else:
        closest = np.nan
    nadir_dict[curr_par] = (closest_post, nadir_time, closest)

    #Get only the sequences at the post nadir time point
    post_seqs = seq_info_df[seq_info_df['time_label_int'] == closest_post].copy()

    


    #Group by resistance mutations and get the frequencies

    post_seqs['res_muts'] = [tuple(x) for x in post_seqs['res_muts'].values]
    post_seqs['res_muts'] = [x if x != (None,) else ('Susceptible',) for x in post_seqs['res_muts'].values]

    #Drop any sequences with all gaps at the resistance positions
    post_seqs = post_seqs[post_seqs['res_muts'] != ('D/N325-', 'N332-', 'S334-')]
    post_seqs['res_muts'] = [x[0] if len(x) == 1 else 'Multiple' for x in post_seqs['res_muts'].values]

    post_seqs = post_seqs.groupby('res_muts').size().reset_index(name='counts')
    post_seqs['frequency'] = post_seqs['counts'] / np.sum(post_seqs['counts'])
    # post_seqs = post_seqs.drop(columns=['counts'])
    
    post_seqs['participant'] = curr_par

    all_post_seqs.append(post_seqs)

all_post_seqs = pd.concat(all_post_seqs, ignore_index=True)

#print the number of unique resistance mutations in each participant at the post nadir timepoint
res_mut_count_df = all_post_seqs.groupby('participant')['res_muts'].nunique().reset_index(name='num_res_muts')
res_mut_count_df.to_csv(outFolder + 'num_res_muts_post_nadir.csv', index=False)

#Save the closest post nadir timepoints
nadir_df = pd.DataFrame.from_dict(nadir_dict, orient='index', columns=['closest_post_nadir', 'nadir_time', 'closest_nadir'])
nadir_df = nadir_df.reset_index()
nadir_df = nadir_df.rename(columns={'index': 'participant'})
nadir_df.to_csv(outFolder + 'closest_post_nadir_timepoints.csv', index=False)

# Now I need to make an entry for every resistance mutation for each participant
all_res_muts = list(set(all_post_seqs['res_muts']))

for curr_res_mut in all_res_muts:
    curr_res_mut_data = all_post_seqs[all_post_seqs['res_muts'] == curr_res_mut]
    curr_pars = curr_res_mut_data['participant'].unique()
    missing_pars = list(set(par_list) - set(curr_pars))
    if missing_pars:
        for curr_par in missing_pars:
            new_row = pd.DataFrame({'res_muts': [curr_res_mut],
                                    'frequency': [0],
                                    'participant': [curr_par]})
            all_post_seqs = pd.concat([all_post_seqs, new_row], ignore_index=True)

#Make resistance mutation a categorical variable
all_post_seqs['res_muts'] = pd.Categorical(all_post_seqs['res_muts'],
                                          categories= dataset_metadata.RESISTANCE_HUE_ORDER[:-2] + ['Multiple', 'Susceptible'],
                                          ordered=True)
all_post_seqs['participant'] = pd.Categorical(all_post_seqs['participant'],
                                            categories=par_order,
                                            ordered=True)


print("Sequences carrying multiple resistance mutations:")
print(all_post_seqs[all_post_seqs['res_muts'] == 'Multiple'])


# Lastly I will plot the results
fig, ax = plt.subplots(dpi=300)

#We will make a stacked bar plot of the frequencies
bottom = np.zeros(len(all_post_seqs['participant'].unique()))

for curr_res_mut in dataset_metadata.RESISTANCE_HUE_ORDER[:-2] + ['Multiple', 'Susceptible']:
    curr_res_mut_data = all_post_seqs[all_post_seqs['res_muts'] == curr_res_mut].copy()
    if len(curr_res_mut_data) == 0:
        continue
    
    #Sort the data by resistance mutationparticipant order
    curr_res_mut_data = curr_res_mut_data.sort_values(by = ['res_muts', 'participant'], ascending=False)
    curr_res_mut_data = curr_res_mut_data.reset_index(drop=True)

    #Put each category on the bar plot
    my_plot = ax.barh(curr_res_mut_data['participant'], curr_res_mut_data['frequency'], 
                     left = bottom, color = dataset_metadata.RESISTANCE_HUE_DICT[curr_res_mut],
                     edgecolor='black', linewidth=linewidth, label=curr_res_mut)
            
    bottom += curr_res_mut_data['frequency'].values

plt.subplots_adjust(left = 0.25, right = 0.95, top = 0.9, bottom = 0.2)
plt.xlabel('Frequency', labelpad=0.6)
plt.ylabel('Participant', labelpad=0.6)
plt.savefig(outFolder + 'escape_mutations_post_nadir_barplot.png', dpi=300)
