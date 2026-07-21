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

params = {'figure.figsize': (4.5, 3), 'axes.labelsize': 6, 'axes.titlesize': 6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
linewidth = 0.5


#In this file I want to go through the parameter sweep runs and plot the
#overall diversity (mean pairwise Hamming distance) at each timepoint,
#varying s, rho, and mu in turn.

out_dir = '../../../results/paper_draft_04-13/figure_6/'
data_dir = '../../../data/slim_simulations/04-30-2026_sims/'
pi_file_stub = 'origins_{origins}_min_{min}_rep_{rep}_pi.csv'

#Walk the simulation directories, load each sim's windowed pi.csv, and
#collapse it to one overall diversity value per timepoint.
diversity_rows = []
for curr_sim_dir in os.listdir(data_dir):
    if curr_sim_dir.startswith('.'):
        continue

    #read the parameters for this simulation from the directory name
    curr_sim_params = curr_sim_dir.split('_')

    param_dict = {}
    while len(curr_sim_params) > 0:
        curr_param_name = curr_sim_params.pop(0)
        curr_param_value = curr_sim_params.pop(0)
        param_dict[curr_param_name] = curr_param_value

    #skip mu = 0.0001 since I cut those out for runtime reasons
    #also skip mu = 1e-08/5e-08 since the iHH analysis step never finished for those runs
    if param_dict['mu'] in ('0.0001', '1e-08', '5e-08') or param_dict['sel'] == '0.01':
        continue

    sim_data_file = data_dir + curr_sim_dir + '/' + pi_file_stub.format(
                                                        origins=param_dict['origins'],
                                                        min=param_dict['min'],
                                                        rep=param_dict['rep'])
    if not os.path.exists(sim_data_file):
        continue
    pi_df = pd.read_csv(sim_data_file)

    #label the first and last timepoints as Pre and Post
    sample_times = pi_df['timepoint'].unique()
    sample_pre = min(sample_times)
    sample_post = max(sample_times)
    sample_time_dict = {sample_pre: 'Pre', sample_post: 'Post'}
    pi_df['time_label'] = pi_df['timepoint'].map(sample_time_dict)

    #collapse all windows to a single overall diversity value per timepoint
    for time_label, time_group in pi_df.groupby('time_label'):
        overall_diversity = time_group['avg_hamming_dist'].mean()

        diversity_rows.append({'mu': param_dict['mu'], 'rho': param_dict['rho'],
                                'sel': param_dict['sel'], 'origins': param_dict['origins'],
                                'rep': param_dict['rep'], 'time_label': time_label,
                                'overall_diversity': overall_diversity})

diversity_df = pd.DataFrame(diversity_rows)

#Cast the parameter columns to numeric types now that parsing is done
diversity_df['mu'] = diversity_df['mu'].astype(float)
diversity_df['rho'] = diversity_df['rho'].astype(float)
diversity_df['sel'] = diversity_df['sel'].astype(float)
diversity_df['origins'] = diversity_df['origins'].astype(int)

#Exclude the 30 origin data from the summary, matching param_sweep_summary.py
diversity_df = diversity_df[diversity_df['origins'] != 30]

rho_list = sorted(diversity_df['rho'].unique())
s_list = sorted(diversity_df['sel'].unique())
mu_list = sorted(diversity_df['mu'].unique())

origin_palette = sns.color_palette('coolwarm', len(diversity_df['origins'].unique()))
origin_palette_dict = {origin: origin_palette[i] for i, origin in enumerate(sorted(diversity_df['origins'].unique()))}

#Fix rho, mu, and s in turn, matching param_sweep_summary.py
fixed_rho = 1e-07
fixed_mu = 1e-07
fixed_s = 1

fig, ax = plt.subplots(2, 3, sharey=False)

time_label_list = ['Pre', 'Post']
for row_ind, time_label in enumerate(time_label_list):
    row_data = diversity_df[diversity_df['time_label'] == time_label]

    #Varying s
    curr_ax = ax[row_ind, 0]
    curr_data = row_data[(row_data['rho'] == fixed_rho) & (row_data['mu'] == fixed_mu)]
    sns.stripplot(data=curr_data, x='sel', y='overall_diversity', hue='origins', palette=origin_palette_dict,
                  ax=curr_ax, jitter=True, dodge=True, marker='.', alpha=0.3)
    curr_ax.set_xlabel(r'Selection coefficient ($s$)')
    curr_ax.set_ylabel(f'{time_label}\nOverall diversity\n(mean pairwise\nHamming distance)')
    curr_ax.legend().set_visible(False)

    #Varying rho
    curr_ax = ax[row_ind, 1]
    curr_data = row_data[(row_data['sel'] == fixed_s) & (row_data['mu'] == fixed_mu)]
    sns.stripplot(data=curr_data, x='rho', y='overall_diversity', hue='origins', palette=origin_palette_dict,
                  ax=curr_ax, jitter=True, dodge=True, marker='.', alpha=0.3)
    curr_ax.set_xlabel(r'Recombination rate ($\rho$)')
    curr_ax.set_ylabel('')
    curr_ax.legend().set_visible(False)

    #Varying mu
    curr_ax = ax[row_ind, 2]
    curr_data = row_data[(row_data['rho'] == fixed_rho) & (row_data['sel'] == fixed_s)]
    sns.stripplot(data=curr_data, x='mu', y='overall_diversity', hue='origins', palette=origin_palette_dict,
                  ax=curr_ax, jitter=True, dodge=True, marker='.', alpha=0.3)
    curr_ax.set_xlabel(r'Mutation rate ($\mu$)')
    curr_ax.set_ylabel('')

    if row_ind == 0:
        curr_ax.legend(ncol=1, title='Number of\norigins', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        curr_ax.legend().set_visible(False)

ax[0, 0].set_title(r'Varying $s$' + f'\n' + r'$\mu$=' + f'{fixed_mu} and ' + r'$\rho$=' + f'{fixed_rho}')
ax[0, 1].set_title(r'Varying $\rho$' + f'\n' + r'$\mu$=' + f'{fixed_mu} and ' + r'$s$=' + f'{fixed_s}')
ax[0, 2].set_title(r'Varying $\mu$' + f'\n' + r'$\rho$=' + f'{fixed_rho} and ' + r'$s$=' + f'{fixed_s}')

#Save the figure
#adjust the subplots to make room for the legend
plt.subplots_adjust(bottom=0.1, top=0.88, left=0.15, right=0.85, wspace=0.2, hspace=0.4)
plt.savefig(f'{out_dir}diversity_param_sweep_summary.png', dpi=300)
plt.close()
