import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/3BNC117/')
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from scipy.stats import ks_2samp


#In this file I want to plot the area between curves side by side for
#the simulated and in vivo datasets
params = {'figure.figsize': (3, 2), 'axes.labelsize': 6,'axes.titlesize':6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial', 'axes.titlesize': 6}
linewidth = 1
fontsize = 6

#make the in vivo data take up more space on the x axis so that the points don't overlap as much
in_vivo_offset = 0.5
in_vivo_jitter = 10


# params = {'figure.figsize': (6, 4), 'axes.labelsize': 14,'axes.titlesize':14,
#           'legend.fontsize': 14, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
#           'legend.title_fontsize': 14, 'font.family': 'sans-serif',
#           'font.sans-serif': 'Arial', 'axes.titlesize': 14}
# linewidth = 1
# fontsize = 14
rcParams.update(params)

out_dir = '../../../results/paper_draft_04-13/figure_5/'

#Shared parameters for both the simulations and the participant data
BIN_NUMBER = 250
FREQ_BIN = 0.2

#participant parameters
PAR_LIST_1074 = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K',
                 '1HD7K', '1HD10K', '1HD11K']

HARDER_SWEEPERS = ['1HD6K', '1HD4K']

TIME_FILTER_OUT = ['Rebound', 'screen', 'pre', 'Nadir', 'HXB2', 'W1']
ALLELE_FREQ_THRESH = 0
MULTI_SEG = True
PAR_PALETTE = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD10K': '#6A3D9A',
               '1HD11K': '#B15928', '1HD9K': '#CAB2D6'}

comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'


#simulation parameters
ORIGIN_NUM_LIST = [1, 2, 10, 20, 30, 50, 100]
MIN_DICT = {1: 1, 2: 2, 10: 5, 20: 10, 30: 20, 50: 40, 100: 80}
REP = range(0, 100)
COMP_TIME = 'W4'


STRIPPLOT_ORDER = ORIGIN_NUM_LIST + ['10-1074'] #+ ['3BNC117']
print(STRIPPLOT_ORDER)

TIME_CONVERT_DICT = {299: 'Pre', 300: 'D0', 303: 'W1', 311: 'W4', 322: 'W8'}
TIMES_TO_KEEP = set(['Pre', 'D0', 'W4', 'W8'])

# # ###############################################################################
#now I want to plot the area between the curves for the simulated data and
#the in vivo data side by side

#Load each of the dataframes
sim_area_df = pd.read_csv(out_dir + 'sim_area_df.csv')
inv_area_df = pd.read_csv(out_dir + 'inv_area_df.csv')

fig, ax = plt.subplots(1, 1)


subset_df_sim = sim_area_df[sim_area_df['time_label'] == COMP_TIME]
subset_df_sim['x_scale'] = subset_df_sim['origins'].map(lambda x: STRIPPLOT_ORDER.index(x))

print(np.unique(subset_df_sim['x_scale']))
print(subset_df_sim['area'])
sns.stripplot(x = subset_df_sim['x_scale'], y = subset_df_sim['area'], ax=ax,
            native_scale=True,
                alpha = 0.3, marker='.', color='black')
#Put a boxplot on top to show the distribution of the simulated data
subset_df_sim['x_scale'] = subset_df_sim['x_scale'] + 0.3
sns.boxplot(x = subset_df_sim['x_scale'], y = subset_df_sim['area'], ax=ax,
                fill = False, width = 0.25, showfliers=False,
                native_scale=True, color = 'black', linewidth=0.5, zorder = 4)

ax.set_ylabel('Integrated iHH')
ax.set_xlabel('Number of Origins')
ax.set_yscale('symlog')


#Get the nadir time for each participant
comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))

subset_df_inv_1074 = inv_area_df[inv_area_df['dataset'] == '10-1074']
subset_df_inv_1074 = subset_df_inv_1074[subset_df_inv_1074['participant'].isin(PAR_LIST_1074)]
subset_df_inv_1074['closest_post_nadir'] = subset_df_inv_1074['participant'].map(nadir_time_dict)

subset_df_inv_1074['closest_post_nadir'] = ['W' + str(int(x)) for x in subset_df_inv_1074['closest_post_nadir']]
print(subset_df_inv_1074['closest_post_nadir'])
print(subset_df_inv_1074['time_label'])
subset_df_inv_1074 = subset_df_inv_1074[subset_df_inv_1074['time_label'] == subset_df_inv_1074['closest_post_nadir']]
#Give the in vivo data more space on the x axis
subset_df_inv_1074['x_scale'] = len(STRIPPLOT_ORDER) -1 + in_vivo_offset


# #Run all of the plot formatting so we can put an intermediate plot in the slides
# ax.set_ylabel('Integrated iHH')
# ax.set_xlabel('Number of Origins')
# ax.axhline(0, color='gray', linestyle='--', linewidth=linewidth)
# ax.set_xlim(-0.5, len(STRIPPLOT_ORDER) -1 + in_vivo_offset + 0.5)

# #Rotate the x labels for the participant plot
# ax.set_xticks(list(range(len(STRIPPLOT_ORDER[:-1]))) +  [len(STRIPPLOT_ORDER[:-1]) + in_vivo_offset])
# ax.set_xticklabels(STRIPPLOT_ORDER, rotation=90, ha='center')
# plt.subplots_adjust(wspace=0.2, top =0.9, bottom=0.3, left=0.15, right=0.75)

# plt.savefig(out_dir + 'iihh_no_in_vivo_slide_size.png', dpi = 300)



sns.stripplot(x = 'x_scale', y = 'area', data=subset_df_inv_1074, ax=ax,
                hue = 'participant', palette = PAR_PALETTE,
                native_scale=True,
                alpha = 1, color = 'black', dodge = True,
                linewidth=0.5, size=3, jitter = in_vivo_jitter)
#Put a transparent boxplot on top to show the distribution of the in vivo data
subset_df_inv_1074['x_scale'] = subset_df_inv_1074['x_scale'] + 1

# sns.boxplot(x = 'x_scale', y = 'area', data=subset_df_inv_1074, ax=ax,
#                 fill = False, width = 0.5, showfliers=False,
#                 native_scale=True, color = 'black', linewidth=linewidth, zorder = 4)

#Make separate boxplots for the harder sweepers and the softer sweepers
sweep_type_dict = {x: 'Harder Sweeper' if x in HARDER_SWEEPERS else 'Softer Sweeper' for x in subset_df_inv_1074['participant']}
subset_df_inv_1074['sweep_type'] = subset_df_inv_1074['participant'].map(sweep_type_dict)
sns.boxplot(x = 'x_scale', y = 'area', data=subset_df_inv_1074, ax=ax,
                fill = False, width = 0.5, showfliers=False,
                native_scale=True, color = 'black', linewidth=0.5, zorder = 4)

ax.set_ylabel('Integrated iHH')
ax.set_xlabel('Number of Origins')
ax.axhline(0, color='gray', linestyle='--', linewidth=linewidth)

#Rotate the x labels for the participant plot
ax.set_xticks(list(range(len(STRIPPLOT_ORDER[:-1]))) +  [len(STRIPPLOT_ORDER[:-1]) + in_vivo_offset])
ax.set_xticklabels(STRIPPLOT_ORDER, rotation=90, ha='center')

#Put a legend in the last plot
labels = list(PAR_PALETTE.keys())
handles = [plt.Line2D([0], [0], marker='o', color='black',
                        markerfacecolor=PAR_PALETTE[x], markersize=3,
                        markeredgewidth=0.5,
                        linewidth=0.5) for x in labels]


handles = handles
labels = labels
ax.legend(handles, labels, title='Participant', loc='upper left',
                bbox_to_anchor=(1.05, 1),
                borderaxespad=0, ncol = 1, frameon=False)



plt.subplots_adjust(wspace=0.2, top =0.9, bottom=0.3, left=0.15, right=0.75)
plt.savefig(out_dir + '_true_before_test_out.png', dpi = 300)
plt.close()
