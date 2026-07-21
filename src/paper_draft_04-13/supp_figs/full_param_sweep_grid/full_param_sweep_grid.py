import sys
sys.path.append('../../../../bin/')
sys.path.append('../../../../bin/wrappers/')
sys.path.append('../../../../data/clyde_westfall_2024_final/10-1074/')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch

import area_between_curves

params = {'figure.figsize': (7, 7), 'axes.labelsize': 8, 'axes.titlesize': 8,
          'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
          'legend.title_fontsize': 8, 'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
linewidth = 0.5


out_dir = '../../../../results/paper_draft_04-13/supp_figs/full_param_sweep_grid/'

#I want to plot the area between the two curves over various parameters to see how it changes.
ihh_data_file = '../../../../results/02-19-2026_exploratory/04-23-2026/standardized_ihh_binned.csv'
ihh_data = pd.read_csv(ihh_data_file)

#Exclude the 30 origin data from the summary
ihh_data = ihh_data[ihh_data['origins'] != 30]

rho_list = sorted(float(x) for x in ihh_data['rho'].unique())
mu_list = sorted(float(x) for x in ihh_data['mu'].unique())
s_list = sorted([0.1, 1, 5])

#Check which combinations of parameters we have data for
for mu in mu_list:
    for rho in rho_list:
        for s in s_list:
            curr_data = ihh_data[(ihh_data['mu'] == mu) & (ihh_data['rho'] == rho) & (ihh_data['sel'] == s)]
            if curr_data.empty:
                print(f'No data for mu={mu}, rho={rho}, sel={s}')

origin_palette = sns.color_palette('viridis', len(ihh_data['origins'].unique()))
origin_palette_dict = {origin: origin_palette[i] for i, origin in enumerate(sorted(ihh_data['origins'].unique()))}

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)

#Loop through each of the parameters
for mu_ind, mu in enumerate(mu_list):
    curr_mu_data = ihh_data[ihh_data['mu'] == mu]

    for rho_ind, rho in enumerate(rho_list):
        curr_rho_data = curr_mu_data[curr_mu_data['rho'] == rho]
        if curr_rho_data.empty:
            print(f'No data for mu={mu}, rho={rho}')
            continue

        #Calculate the area between the Pre/Post curves for each selection
        #coefficient, origin number and replicate.
        area_records = []
        for s in s_list:
            curr_s_data = curr_rho_data[curr_rho_data['sel'] == s]

            for curr_origin_num in curr_s_data['origins'].unique():
                curr_origin_data = curr_s_data[curr_s_data['origins'] == curr_origin_num]

                for rep in curr_origin_data['rep'].unique():
                    curr_rep_data = curr_origin_data[curr_origin_data['rep'] == rep]

                    calculated_area = area_between_curves.area_between_binned_curves(curr_rep_data)
                    area_records.append({'mu': mu, 'rho': rho, 'sel': s,
                                         'origins': curr_origin_num,
                                         'area_between_curves': calculated_area,
                                         'rep': rep})

        area_between_curves_df = pd.DataFrame(area_records)

        sns.stripplot(data=area_between_curves_df, x='sel', y='area_between_curves', hue='origins',
                      palette=origin_palette_dict, ax=ax[mu_ind, rho_ind],
                      jitter=True, dodge=True, marker='.', alpha=0.3)

        ax[mu_ind, rho_ind].axhline(0, color='black', linestyle='--', linewidth=linewidth, zorder=0)
        ax[mu_ind, rho_ind].set_title(r'$\mu$=' + f'{mu}, ' + r'$\rho$=' + f'{rho}')
        ax[mu_ind, rho_ind].set_xlabel('Selection coefficient (s)')
        ax[mu_ind, rho_ind].set_ylabel('iiHH')


#Add a legend to the last subplot
ax[-1, -1].legend(title='Number of origins', bbox_to_anchor=(0.75, 2), loc='upper center', ncol=2)

#Turn off the subplots without data
for i in range(len(mu_list)):
    for j in range(len(rho_list)):
        if j > i:
            ax[i, j].axis('off')

#Set the other legends to not show
ax = ax.flatten()
for i in range(len(ax)-1):
    ax[i].legend().set_visible(False)

#Make room on the left/bottom for the overall axis labels and arrows
fig.subplots_adjust(left=0.17, bottom=0.15)

#Overall mu label on the left (mu increases down the rows), arrow pointing down
mu_arrow = FancyArrowPatch((0.06, 0.6), (0.06, 0.4), transform=fig.transFigure,
                           arrowstyle='-|>', mutation_scale=20, lw=1.5,
                           color='black', clip_on=False)
fig.patches.append(mu_arrow)
fig.text(0.025, 0.5, r'Increasing $\mu$', rotation=90, va='center', ha='center',
         fontsize=10)

#Overall rho label on the bottom (rho increases across the columns), arrow pointing right
rho_arrow = FancyArrowPatch((0.43, 0.05), (0.63, 0.05), transform=fig.transFigure,
                            arrowstyle='-|>', mutation_scale=20, lw=1.5,
                            color='black', clip_on=False)
fig.patches.append(rho_arrow)
fig.text(0.53, 0.015, r'Increasing $\rho$', va='center', ha='center',
         fontsize=10)

plt.savefig(out_dir + 'area_between_curves_sel.png', dpi=300)

plt.yscale('symlog')

plt.savefig(out_dir + 'area_between_curves_sel_symlog.png', dpi=300)
