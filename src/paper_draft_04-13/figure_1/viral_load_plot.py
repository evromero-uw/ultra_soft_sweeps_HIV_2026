import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rcParams

# Today I am going to make a plot of viral load trajectories over the 
# course of the trial for each participant

inDir = '../../../data/clyde_westfall_2024_final/10-1074/'
vl_file = '../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
outFolder = '../../../results/paper_draft_04-13/figure_1/'

params = {'axes.labelsize': 6,'axes.titlesize':6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
linewidth = 0.5
fontsize = 6
rcParams.update(params)

par_list = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']

par_order = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K', '1HD7K',
             '1HD10K', '1HD11K']
par_palette = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD9K': '#CAB2D6',
               '1HD10K': '#6A3D9A', '1HD11K': '#B15928'}
time_filter_out = ['HXB2', 'PR', 'Rebound', 'screen', 'pre', 'Nadir',
                   'W20','W24', 'D1', 'D4', 'D2']


###############################################################################
vl_df = pd.read_csv(vl_file)



vl_df = vl_df[vl_df['Study_ID'].isin(par_list)]
print(vl_df.columns)

#Now I need to melt the dataframe so that I have a column for time point
vl_df = vl_df.melt(id_vars=['Study_ID'], var_name='Timepoint', value_name='Viral_Load')
vl_df = vl_df[~vl_df['Timepoint'].isin(time_filter_out)]
vl_df['Timepoint'] = [x if x != 'D7' else 'W1' for x in vl_df['Timepoint']]
vl_df['time_label_int'] = [int(tp[1:]) * 7 for tp in vl_df['Timepoint']]

#Participant 1HC2 initiated ART at week 8, so we will drop any time points
#after week 8 for this participant
vl_df = vl_df[~((vl_df['Study_ID'] == '1HC2') & (vl_df['time_label_int'] >= 56))]

#Participant 1HD4K initiated ART at week 16, so we will drop any time points
#after week 16 for this participant
vl_df = vl_df[~((vl_df['Study_ID'] == '1HD4K') & (vl_df['time_label_int'] >= 112))]

vl_df.to_csv(outFolder + 'viral_load_trajectories.csv', index=False)

#Drop any rows with missing viral load values
vl_df = vl_df.dropna(subset=['Viral_Load'])
print(vl_df['Timepoint'].unique())

fig, axs = plt.subplots(2, 1, figsize=(2.5, 1.5), dpi=300, height_ratios=[1, 10],
                    sharex=True, sharey=False)

ax = axs[1]

for par in par_list:
    par_data = vl_df[vl_df['Study_ID'] == par]
    ax.plot(par_data['time_label_int'], par_data['Viral_Load'], 
            label=par, color=par_palette[par], linewidth=1)
ax.legend()

#Re sort the legend so that it is in order of participant ID
handles, labels = ax.get_legend_handles_labels()
label_dict = {label: handle for label, handle in zip(labels, handles)}
labels = sorted(label_dict.keys(), key = lambda x: int(x[3:4] if len(x) < 6 else x[3:5]))
handles = [label_dict[label] for label in labels]
ax.legend(handles, labels, bbox_to_anchor=(1.05, 1.3), loc='upper left',
          title='Participant', fontsize=fontsize, borderaxespad=0,
          frameon=False)
ax.set_yscale('log')
ax.set_xlabel('Trial week', labelpad=0.6)
ax.set_ylabel('Viral load (copies/mL)', labelpad=0.6)

#Redo the x ticks
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_xticklabels(['0', '4', '8', '12', '16'])

#Make a second subplot with the trial schematic of sampling time points
ax = axs[0]

ax.hlines(1, 0, 112, color='black', linewidth = linewidth)
ax.vlines([0, 7, 28, 56, 84, 112], 0.5, 1.5, color='black', linewidth = linewidth)
ax.set_title('Sequencing time points', pad=3, fontsize=fontsize)
ax.axis('off')

plt.subplots_adjust(left = 0.2, right = 0.65, top = 0.85, bottom = 0.2)

plt.savefig(outFolder + 'viral_load_trajectories.png', dpi=300)