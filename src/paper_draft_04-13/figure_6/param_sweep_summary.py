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

import area_between_curves

params = {'figure.figsize': (4.5, 2), 'axes.labelsize': 6, 'axes.titlesize': 6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
linewidth = 0.5


#In this file I want to go through the parameter sweep runs
#I want to make a plot varying s, a plot varying rho, and a plot varying mu

fig, ax = plt.subplots(1, 3, sharey=True)
ax = ax.flatten()

out_dir = '../../../results/paper_draft_04-13/figure_6/'

#I want to plot the area between the two curves over various parameters to see how it changes. 
ihh_data_file = '../../../results/02-19-2026_exploratory/04-23-2026/standardized_ihh_binned.csv'
ihh_data = pd.read_csv(ihh_data_file)


#Exclude the 30 origin data from the summary
ihh_data = ihh_data[ihh_data['origins'] != 30]

rho_list = [float(x) for x in ihh_data['rho'].unique()]
s_list = [float(x) for x in ihh_data['sel'].unique()]
mu_list = [float(x) for x in ihh_data['mu'].unique()]

print(mu_list)


s_list = [0.1, 1, 5]

rho_list = sorted(rho_list)
s_list = sorted(s_list)
mu_list = sorted(mu_list)

#Fix rho and mu
fixed_rho = 1e-07
fixed_mu = 1e-07
fixed_s = 1

#sort all the lists
rho_list.sort()
s_list.sort()
mu_list.sort()

origin_palette = sns.color_palette('coolwarm', len(ihh_data['origins'].unique()))
origin_palette_dict = {origin: origin_palette[i] for i, origin in enumerate(sorted(ihh_data['origins'].unique()))}

#First calculate the area between curves for each combination of parameters and store it in a dataframe

param_combos = [(mu, fixed_rho, fixed_s) for mu in mu_list] +\
                 [(fixed_mu, rho, fixed_s) for rho in rho_list] +\
                    [(fixed_mu, fixed_rho, s) for s in s_list]
print(param_combos)


area_between_curves_df = []
for mu, rho, s in param_combos:
    curr_data = ihh_data[(ihh_data['mu'] == mu) & (ihh_data['rho'] == rho) & (ihh_data['sel'] == s)]
    origin_area_dict = {}

    print(f'Processing mu={mu}, rho={rho}, sel={s}')
    print('-----------------')

    for curr_origin_num in curr_data['origins'].unique():
        curr_origin_data = curr_data[curr_data['origins'] == curr_origin_num]
        if curr_origin_data.empty:
            continue
        
        for rep in curr_origin_data['rep'].unique():
            curr_rep_data = curr_origin_data[curr_origin_data['rep'] == rep]
            if curr_rep_data.empty:
                continue

            #plot the data
            pre_curve = curr_rep_data[curr_rep_data['time_label'] == 'Pre'].sort_values(by='bin_start')
            post_curve = curr_rep_data[curr_rep_data['time_label'] == 'Post'].sort_values(by='bin_start')
            
            #Merge the two curves on the bin start and end values to make sure they are aligned for the area calculation
            merged_curves = pd.merge(pre_curve, post_curve, on=['bin_start', 'bin_end'], suffixes=('_d0', '_post'))
            
            #Remove any rows with NaN values in the binned_average columns
            merged_curves = merged_curves.dropna(subset=['binned_average_d0', 'binned_average_post',
                                                            'adj_iHH_std_d0', 'adj_iHH_std_post'])
            bin_start = merged_curves['bin_start'].values
            binned_averages_d0 = merged_curves['binned_average_d0'].values
            binned_averages = merged_curves['binned_average_post'].values

            #calculate the area between the two curves
            calculated_area = area_between_curves.area_between_curves(bin_start,
                                                                binned_averages_d0,
                                                                binned_averages)
            
            #Normalize the area by the length of the fragment (2500) to make it easier to compare across different parameters
            calculated_area = calculated_area / 2500
            origin_area_dict[curr_origin_num] = calculated_area

            area_between_curves_df.append({'mu': mu, 'rho': rho, 'sel': s, 'origins': curr_origin_num, 'area_between_curves': calculated_area,
                                                'rep': rep})
            print(calculated_area)

area_between_curves_df = pd.DataFrame(area_between_curves_df)

print(area_between_curves_df['mu'].unique())

#Now I want to plot the area between curves varying s
curr_ax = 0
curr_data = area_between_curves_df[(area_between_curves_df['rho'] == fixed_rho) &\
                                    (area_between_curves_df['mu'] == fixed_mu)]
sns.stripplot(data = curr_data, x='sel', y='area_between_curves', hue='origins', palette=origin_palette_dict, ax=ax[curr_ax],
                    jitter=True, dodge=True, marker = '.', alpha = 0.3)
ax[curr_ax].axhline(0, color='black', linestyle='--', linewidth=linewidth, zorder=0)
ax[curr_ax].set_title(r'Varying $s$' + f'\n' + r'$\mu$=' + f'{fixed_mu} and ' + r'$\rho$=' + f'{fixed_rho}')
ax[curr_ax].set_xlabel(r'Selection coefficient ($s$)')
ax[curr_ax].set_ylabel('Genetic linkage\n (integrated iHH)')
ax[curr_ax].legend().set_visible(False)
curr_ax += 1


#Varying rho
curr_data = area_between_curves_df[(area_between_curves_df['sel'] == fixed_s) &\
                                    (area_between_curves_df['mu'] == fixed_mu)]
sns.stripplot(data = curr_data, x='rho', y='area_between_curves', hue='origins', palette=origin_palette_dict, ax=ax[curr_ax],
                    jitter=True, dodge=True, marker = '.', alpha = 0.3)
ax[curr_ax].axhline(0, color='black', linestyle='--', linewidth=linewidth, zorder=0)
ax[curr_ax].set_title(r'Varying $\rho$' + f'\n' + r'$\mu$=' + f'{fixed_mu} and ' + r'$s$=' + f'{fixed_s}')
ax[curr_ax].set_xlabel(r'Recombination rate ($\rho$)')
ax[curr_ax].set_ylabel('Genetic linkage (integrated iHH)')
ax[curr_ax].legend().set_visible(False)
curr_ax += 1

#Now plot the area between curves varying mu
curr_data = area_between_curves_df[(area_between_curves_df['rho'] == fixed_rho) &\
                                    (area_between_curves_df['sel'] == fixed_s)]
print(curr_data)
sns.stripplot(data = curr_data, x='mu', y='area_between_curves', hue='origins', palette=origin_palette_dict, ax=ax[curr_ax],
                    jitter=True, dodge=True, marker = '.', alpha = 0.3)
ax[curr_ax].set_title(r'Varying $\mu$' + f'\n' + r'$\rho$=' + f'{fixed_rho} and ' + r'$s$=' + f'{fixed_s}')
ax[curr_ax].axhline(0, color='black', linestyle='--', linewidth=linewidth, zorder=0)
ax[curr_ax].set_xlabel(r'Mutation rate ($\mu$)')
ax[curr_ax].set_ylabel('Genetic linkage\n (integrated iHH)')
#align the legend label on the center
ax[curr_ax].legend(ncol=1, title='Number of\norigins', bbox_to_anchor=(1.05, 1), loc='upper left')


#Save the figure

#adjust the subplots to make room for the legend
plt.subplots_adjust(bottom = 0.2, top = 0.85, left = 0.12, right = 0.85, wspace = 0.2, hspace = 0.3)
plt.savefig(f'{out_dir}area_between_curves_summary_lower_fixed.png', dpi = 300)

#Also try plotting it with logged y axis
ax[0].set_ylabel('Genetic linkage\n (integrated iHH)', fontsize=6)
plt.ylim(-10, 10e3)
plt.yscale('symlog')
plt.subplots_adjust(bottom = 0.2, top = 0.85, left = 0.12, right = 0.85, wspace = 0.2, hspace = 0.3)
plt.savefig(f'{out_dir}area_between_curves_summary_symlog_lower_fixed.png', dpi = 300)
plt.close() 



