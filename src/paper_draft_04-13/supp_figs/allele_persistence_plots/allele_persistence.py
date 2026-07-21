import os
import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import expit, logit
from statsmodels.tools import add_constant

import dataset_metadata
import data_util as du
import persistence_analysis as persist
from par_data_class import Pardata

params = {'figure.figsize':(4, 1.5),
          'axes.labelsize': 6,'axes.titlesize':6,  
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
linewidth = 0.5


#In this file I am running the logistic regression analysis on the persistence
#of alleles based on distance. Instead of using the seaborn logistic regression
#plot, I will run my own logistic regression and plot the results.

inDir = '../../../../data/clyde_westfall_2024_final/10-1074/'
vl_file = '../../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K', '1HD10K', '1HD9K', '1HD11K']
time_filter_out = ['PR', 'Rebound', 'screen', 'pre', 'Nadir']
outDir = '../../../../results/paper_draft_04-13/supp_figs/allele_persistence_plots/'


#Count all segregating sites
ALLELE_FREQ_THRESH = 0
TIME_SEG_FILTER = 0.05
MULTI_SEG = True
NUM_BOOTSTRAPS = 1000
DIST_LIM = 1000

SAVED_DATA = False

###############################################################################

fig, ax = plt.subplots(1, 2, sharey=True)

ax = ax.flatten()


if not SAVED_DATA:
    all_persistence_inv = []
    all_persistence_null = []
    for curr_par in par_list:
        print(curr_par)
        ################################ Data Loading #############################
        ###########################################################################
        inFile = inDir + curr_par + '/' '885_' + curr_par + '_NT_filtered.fasta'

        #This file just makes a plot of a heatmap from a fasta file.
        #The path to the data and the resistance positions in hxb2 coordinates
        hxb2_res_positions = dataset_metadata.RESISTANCE_POS_NT_HXB2

        #Construct the data object and load the data
        participant_dat = Pardata(inFile, 'clyde2024', curr_par)

        #Load the data including all minor variants
        participant_dat.load_data_10_1074(hxb2_res_positions, ALLELE_FREQ_THRESH, MULTI_SEG)

        #Get the timepoints we will be filtering out
        curr_filter_out = set(time_filter_out).intersection(set(
                                        participant_dat.seq_info_df['time_label']))
        
        #Filter out the timepoints that are not of interest
        participant_dat.time_filter_data(curr_filter_out)

        #Now filter out sites based on if they reach a threshold frequency at any
        #timepoint (rather than in the aggregate data)
        participant_dat.filter_seg_time(TIME_SEG_FILTER)
        seq_info_df = participant_dat.seq_info_df
        seq_info_df = seq_info_df[seq_info_df['time_label'] != 'HXB2']
        participant_dat.seq_info_df = seq_info_df

        seq_arr = participant_dat.seq_arr
        seg_freq_dict = participant_dat.seg_freq_dict


        seq_info_df['time_label_int'] = [int(x[1:]) for x in \
                                        seq_info_df['time_label'].values]
        
        #Load the viral load data
        vl_df = pd.read_csv(vl_file)

        #Calculate which timepoints are after the nadir
        rebound_time = vl_df[vl_df['Study_ID'] == curr_par]['Nadir'].values[0]
        time_points = seq_info_df['time_label_int'].unique()
        time_points = [int(x) for x in time_points]
        time_points = np.sort(time_points)
        time_points = np.asarray(time_points)

        if isinstance(rebound_time, str):
            rebound_time = int(rebound_time[1:])

            #We need to find the timepoint that corresponds to the timepoint 
            #directly after nadir
            closest = (np.abs(time_points - rebound_time)).argmin() + 1
            if closest == len(time_points):
                closest = np.nan

        else:
            closest = np.nan
        

        rebound_dict = {}
        #Sort the timepoints as before or after rebound
        for curr_label in seq_info_df['time_label'].unique():
            alt_label = 'Week ' + str(curr_label)
            curr_label = int(curr_label[1:])
            if np.isnan(closest):
                rebound_dict[curr_label] = 'Before'
                rebound_dict[alt_label] = 'Before'
            else:
                closest_rebound = time_points[closest]

                if curr_label < closest_rebound:
                    rebound_dict[curr_label] = 'Before'
                    rebound_dict[alt_label] = 'Before'
                else:
                    rebound_dict[curr_label] = 'After'
                    rebound_dict[alt_label] = 'After'
        
        seq_info_df['rebound'] = [rebound_dict[x] for x in seq_info_df['time_label_int']]


        ##################### Running Persistence Analysis ########################
        ###########################################################################
        seg_freq_dict_top2 = {}
        for key in seg_freq_dict:
            curr_values = seg_freq_dict[key]
            curr_values.sort(reverse=True, key=lambda x: x[1])
            seg_freq_dict_top2[key] = curr_values[:2]
        seg_freq_dict = seg_freq_dict_top2

        #Gather the allele frequency by timepoint data
        allele_freq_df, all_time_freqs, time_sample_sizes = \
                        du.seg_freq_by_timepoint(seg_freq_dict_top2, seq_arr, seq_info_df)


        
        #Null persistence analysis
        persistence_null = persist.allele_persist_nulls(all_time_freqs, 
                                                time_sample_sizes, NUM_BOOTSTRAPS)

        #In vivo persistence analysis
        persistence_inv = persist.in_vivo_persistence(allele_freq_df)

        ################### Probability of Persistence vs Distance #################
        ############################################################################
        #Filter out the other categories
        persistence_inv = persistence_inv[persistence_inv['persistence'].isin(
                                                        ['Persisting', 'Lost'])]
        persistence_null = persistence_null[persistence_null['persistence'].isin(
                                                ['Null Persisting', 'Null Lost'])]
        
        persistence_inv['rebound'] = [rebound_dict[x] for x in persistence_inv['time']]
        persistence_null['rebound'] = [rebound_dict[x] for x in persistence_null['time']]

        
        #Get the starting frequency of the allele
        #A dictionary where each key is a position and an allele, and the value is
        #the frequency of that allele at day 0
        zero_freq_dict = {}
        allele_freq_df_0 = allele_freq_df[allele_freq_df['time'] == 'D0']
        for index, row in allele_freq_df_0.iterrows():
            zero_freq_dict[(row['position'], row['allele'])] = row['freqs']
        
        persistence_inv['start_freq'] = [zero_freq_dict[(x, y)] for x, y in \
                                        zip(persistence_inv['position'], 
                                            persistence_inv['allele'])]
        persistence_null['start_freq'] = [x for x in persistence_null['freq']]
        
        resistance_pos = participant_dat.arr_res_positions
        resistance_min = min(resistance_pos[0][0], resistance_pos[1][0])
        resistance_max = max(resistance_pos[0][1], resistance_pos[1][1])

        def distance_from_resistance(pos, gap_col_set):

            if pos < resistance_min:
                return du.calc_bp_dist(pos, resistance_min, seq_arr, gap_col_set)
            elif pos > resistance_max:
                return du.calc_bp_dist(pos, resistance_max, seq_arr, gap_col_set)
            else:
                return 0

        #Make a new column for the distance from the resistance site 
        gap_col_set = du.make_gap_col_set(seq_arr)
        distance_dict = {}
        all_post = set(persistence_inv['position'].unique()).union(persistence_null['position'].unique())
        for i in all_post:
            distance_dict[i] = distance_from_resistance(i, gap_col_set)

        persistence_inv['distance'] = [distance_dict[x] for x in \
                                        persistence_inv['position']]
        persistence_null['distance'] = [distance_dict[x] for x in \
                                        persistence_null['position']]


        #Make a new column for the persistence as a binary variable
        persistence_inv['persisted'] = [1 if x == 'Persisting' else 0 for x in \
                                        persistence_inv['persistence']]
        persistence_null['persisted'] = [1 if x == 'Null Persisting' else 0 for x in \
                                        persistence_null['persistence']]
        
        #Label the participant and put it into the dataframe for larger analysis
        persistence_inv['par'] = curr_par
        persistence_null['par'] = curr_par
        all_persistence_inv.append(persistence_inv)
        all_persistence_null.append(persistence_null)    

    #Now make logistic regression plots for all the participants before and after rebound
    all_persistence_inv = pd.concat(all_persistence_inv, ignore_index=True)
    all_persistence_null = pd.concat(all_persistence_null, ignore_index=True)

    #Save the data
    all_persistence_inv.to_csv(outDir + 'all_persistence_inv_1K.csv', index = False)
    all_persistence_null.to_csv(outDir + 'all_persistence_null_1K.csv', index = False)
else:
    all_persistence_inv = pd.read_csv(outDir + 'all_persistence_inv_1K.csv')
    all_persistence_null = pd.read_csv(outDir + 'all_persistence_null_1K.csv')

#Filter out 1HD9K
all_persistence_inv = all_persistence_inv[all_persistence_inv['par'] != '1HD9K']
all_persistence_null = all_persistence_null[all_persistence_null['par'] != '1HD9K']
all_persistence_inv = all_persistence_inv[all_persistence_inv['distance'] <= DIST_LIM]
all_persistence_null = all_persistence_null[all_persistence_null['distance'] <= DIST_LIM]


#Make a plot for persistance over frequency
before_rebound_inv = all_persistence_inv[all_persistence_inv['rebound'] == 'Before']
after_rebound_inv = all_persistence_inv[all_persistence_inv['rebound'] == 'After']
before_rebound_null = all_persistence_null[all_persistence_null['rebound'] == 'Before']
after_rebound_null = all_persistence_null[all_persistence_null['rebound'] == 'After']

df_list = [before_rebound_null, after_rebound_null, before_rebound_inv, after_rebound_inv]
labels = ['Before Nadir Null', 'After Nadir Null',
          'Before Nadir In Vivo', 'After Nadir In Vivo']
colors = ['black', 'tab:gray', 'tab:blue', 'tab:orange', ]


#Only look after Nadir
df_list = [after_rebound_null, after_rebound_inv]
labels = ['After nadir null', 'After nadir in vivo']
colors = ['gray', 'tab:blue']


# df_list = [all_persistence_inv, all_persistence_null]
# labels = ['In vivo', 'null']
# colors = ['tab:red', 'black']

for i in range(len(df_list)):
    curr_df = df_list[i]
    curr_label = labels[i]
    curr_color = colors[i]
    my_slice = curr_df[['start_freq', 'persisted']]

    #Make a null model and predict the persistence
    freq_values = curr_df['start_freq'].values
    freq_values = add_constant(freq_values)

    logistic_reg = sm.Logit(curr_df['persisted'], 
                        freq_values).fit_regularized()
    print(f'Logistic regression results for frequency analysis: {curr_label}')
    print(logistic_reg.summary())
    print(logistic_reg.pvalues)
    freq_range = np.linspace(0, 1, 20)
    freq_range_no_const = freq_range
    freq_range = add_constant(freq_range)
    curr_pred = logistic_reg.predict(freq_range)

    #Get the standard error for the null model and plot it
    se = np.sqrt(np.array([x@logistic_reg.cov_params()@x for x in freq_range]))
    curr_pred_error = pd.DataFrame({'x':freq_range_no_const,
                                    'pred':curr_pred, 
                                    'ymin': expit(logit(curr_pred) - 1.96*se), 
                                    'ymax': expit(logit(curr_pred) + 1.96*se)})
    ax[0].fill_between(curr_pred_error['x'], curr_pred_error['ymin'], 
                            curr_pred_error['ymax'], alpha=0.5, color=curr_color)
    ax[0].plot(freq_range_no_const, curr_pred, color=curr_color, label= curr_label, linewidth=linewidth)
    if i ==1:
        ax[0].scatter(x = curr_df['start_freq'], y = curr_df['persisted'], 
                        color=curr_color, alpha=0.5, marker='|', linewidth=linewidth,
                        zorder = 3, s = 2)

ax[0].set_xlabel('Allele frequency at day 0', labelpad=0.4)
ax[0].set_ylabel('Probability of persistence', labelpad=0.4)
ax[0].set_title('Probability of persistence vs. frequency', pad = 0.9)



for i in range(len(df_list)):
    curr_df = df_list[i]
    curr_label = labels[i]
    curr_color = colors[i]

    #Make a null model and predict the persistence
    dist_values = curr_df['distance'].values
    dist_values = add_constant(dist_values)

    logistic_reg = sm.Logit(curr_df['persisted'], 
                        dist_values).fit_regularized()

    dist_range = np.linspace(0, 1000, 20)
    dist_range_no_const = dist_range
    dist_range = add_constant(dist_range)
    curr_pred = logistic_reg.predict(dist_range)
    print(f'Logistic regression results for distance analysis: {curr_label}')
    print(logistic_reg.summary())
    print(logistic_reg.llr_pvalue)

    #Get the standard error for the null model and plot it
    se = np.sqrt(np.array([x@logistic_reg.cov_params()@x for x in dist_range]))
    curr_pred_error = pd.DataFrame({'x':dist_range_no_const,
                                    'pred':curr_pred, 
                                    'ymin': expit(logit(curr_pred) - 1.96*se), 
                                    'ymax': expit(logit(curr_pred) + 1.96*se)})
    ax[1].fill_between(curr_pred_error['x'], curr_pred_error['ymin'], 
                            curr_pred_error['ymax'], alpha=0.5, color=curr_color)
    ax[1].plot(dist_range_no_const, curr_pred, color=curr_color, label= curr_label, linewidth=linewidth)
    if i==1:
        ax[1].scatter(x = curr_df['distance'], y = curr_df['persisted'], 
                            color=curr_color, alpha=0.1, marker='|', linewidth=linewidth,
                            zorder = 3, s = 2)

ax[1].set_xlabel('Distance from escape loci (nt)', labelpad= 0.4)
ax[1].set_ylabel('', labelpad=0.4)
ax[1].set_title('Probability of persistence vs. distance', pad=0.9)
ax[1].legend()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2, wspace = 0.2)
plt.savefig(outDir+ 'allele_persistence_regression.png', dpi = 300)
plt.close()