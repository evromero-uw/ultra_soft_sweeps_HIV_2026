import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.stats import gamma

# For each number of simulated origins, fit a gamma distribution to the
# Integrated iHH values at W4, then plot the likelihood of each in-vivo
# participant's own Integrated iHH value under each fit.

params = {'axes.labelsize': 6, 'axes.titlesize': 6,
          'legend.fontsize': 6, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
          'legend.title_fontsize': 6, 'figure.titlesize': 6, 
          'font.family': 'sans-serif',
          'font.sans-serif': 'Arial'}
rcParams.update(params)
linewidth = 1

###############################################################################
# Paths
###############################################################################
out_dir = '../../../results/paper_draft_04-13/figure_5/'
comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'

###############################################################################
# Analysis parameters
###############################################################################
ORIGIN_NUM_LIST = [1, 2, 10, 20, 30, 50, 100]
COMP_TIME = 'W4'
BIN_COUNT = 30

FIG_SIZE_1 = (6, 3)
FIG_SIZE_2 = (3.5, 2)


# 10-1074 participants. 1HD9K has no entry in closest_post_nadir_timepoints.csv,
# so it falls back to W4 (see NADIR_FALLBACK below) rather than being dropped.
PAR_LIST_1074 = ['1HB3', '1HC2', '1HC3', '1HD1', '1HD4K', '1HD5K', '1HD6K',
                 '1HD7K', '1HD10K', '1HD11K', '1HD9K']

NADIR_FALLBACK = 4

# Same participant color palette used in summary_ihh_one_panel.py, extended
# with the next color in the ColorBrewer "Paired" sequence for 1HD9K
PAR_PALETTE = {'1HB3': '#A6CEE3', '1HC2': '#1F78B4', '1HC3': '#B2DF8A',
               '1HD1': '#33A02C', '1HD4K': '#FB9A99', '1HD5K': '#E31A1C',
               '1HD6K': '#FDBF6F', '1HD7K': '#FF7F00', '1HD10K': '#6A3D9A',
               '1HD11K': '#B15928', '1HD9K': '#CAB2D6'}

###############################################################################
# Fit a gamma distribution to the simulated Integrated iHH values per origins
###############################################################################
sim_area_df = pd.read_csv(out_dir + 'sim_area_df.csv')
subset_df = sim_area_df[sim_area_df['time_label'] == COMP_TIME]

gamma_params = []
for curr_origin in ORIGIN_NUM_LIST:
    curr_data = subset_df[subset_df['origins'] == curr_origin]['area']
    shape, loc, scale = gamma.fit(curr_data)
    gamma_params.append({'origins': curr_origin, 'shape': shape, 'loc': loc, 'scale': scale})

gamma_params_df = pd.DataFrame(gamma_params)

#Round the gamma fit parameters to 3 decimal places for easier reading in the CSV
gamma_params_df_save = gamma_params_df.copy()
gamma_params_df_save[['shape', 'loc', 'scale']] = gamma_params_df_save[['shape', 'loc', 'scale']].round(3)
gamma_params_df_save.to_csv(out_dir + 'gamma_fit_params.csv', index=False)

###############################################################################
# Plot grid: one histogram panel per number of origins
###############################################################################
fig, ax = plt.subplots(2, 4, sharex=False, sharey=False, figsize=FIG_SIZE_1)
ax = ax.flatten()

origin_colors = plt.cm.viridis(np.linspace(0, 1, len(ORIGIN_NUM_LIST)))
fit_curves = []
gamma_params = []

for i, curr_origin in enumerate(ORIGIN_NUM_LIST):
    curr_ax = ax[i]
    curr_data = subset_df[subset_df['origins'] == curr_origin]['area']
    bin_edges = np.histogram_bin_edges(curr_data, bins=BIN_COUNT)

    curr_ax.hist(curr_data, bins=bin_edges, density=True, color='steelblue',
                 edgecolor='black', linewidth=0.5)

    # Fit a gamma distribution (with a free location shift to accommodate
    # negative area values) and overlay its density.
    shape, loc, scale = gamma.fit(curr_data)
    gamma_params.append({'origins': curr_origin, 'shape': shape, 'loc': loc, 'scale': scale})
    x_vals = np.linspace(bin_edges[0], bin_edges[-1], 200)
    fitted_pdf = gamma.pdf(x_vals, shape, loc=loc, scale=scale)
    curr_ax.plot(x_vals, fitted_pdf, color='darkorange', linewidth=1)
    fit_curves.append((x_vals, fitted_pdf))

    curr_ax.set_title(f'{curr_origin} origins')
    curr_ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)

# Summary panel: all gamma fit curves overlaid, colored by number of origins
summary_ax = ax[len(ORIGIN_NUM_LIST)]
for (x_vals, fitted_pdf), curr_origin, color in zip(fit_curves, ORIGIN_NUM_LIST, origin_colors):
    summary_ax.plot(x_vals, fitted_pdf, color=color, linewidth=1,
                     label=f'{curr_origin}')
summary_ax.axvline(0, color='gray', linestyle='--', linewidth=0.5)
summary_ax.set_title('Fits')
summary_ax.legend(title='Origins', loc='upper left', bbox_to_anchor=(1.02, 1),
                   frameon=False, borderaxespad=0)
summary_ax.set_ylim(0, summary_ax.get_ylim()[1])

fig.supxlabel('Integrated iHH', y=0.02, fontsize=6)
fig.supylabel('Density', x=0.02, ha='center', va='center', fontsize=6)
plt.subplots_adjust(hspace=0.5, wspace=0.3, top=0.92, bottom=0.13, left=0.09, right=0.85)
plt.savefig(out_dir + 'iihh_histogram_grid.png', dpi=300)
plt.close()

###############################################################################
# Restrict each in-vivo participant to their own nearest post-nadir timepoint
###############################################################################
comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'], comp_time_df['closest_post_nadir']))

inv_area_df = pd.read_csv(out_dir + 'inv_area_df.csv')
inv_area_df = inv_area_df[inv_area_df['dataset'] == '10-1074']
inv_area_df = inv_area_df[inv_area_df['participant'].isin(PAR_LIST_1074)]
inv_area_df['closest_post_nadir'] = inv_area_df['participant'].map(nadir_time_dict).fillna(NADIR_FALLBACK)
inv_area_df['closest_post_nadir'] = ['W' + str(int(x)) for x in inv_area_df['closest_post_nadir']]
inv_area_df = inv_area_df[inv_area_df['time_label'] == inv_area_df['closest_post_nadir']]

###############################################################################
# Density of each in-vivo Integrated iHH value under each fitted distribution
###############################################################################
likelihood_rows = []
for _, inv_row in inv_area_df.iterrows():
    for _, gamma_row in gamma_params_df.iterrows():
        density = gamma.pdf(inv_row['area'], gamma_row['shape'],
                             loc=gamma_row['loc'], scale=gamma_row['scale'])
        likelihood_rows.append({'participant': inv_row['participant'],
                                 'in_vivo_area': inv_row['area'],
                                 'origins': gamma_row['origins'],
                                 'gamma_density': density})

likelihood_df = pd.DataFrame(likelihood_rows)
likelihood_df.to_csv(out_dir + 'in_vivo_gamma_density.csv', index=False)

###############################################################################
# Plot: likelihood vs. number of origins, one line per participant
###############################################################################
fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE_2)

for curr_par in PAR_LIST_1074:
    curr_df = likelihood_df[likelihood_df['participant'] == curr_par].sort_values('origins')
    # Floor exact-zero densities (float underflow in the extreme tail) so
    # every participant stays visible on the log-scaled y-axis.
    plotted_density = curr_df['gamma_density'].clip(lower=1e-10)
    ax.plot(curr_df['origins'].astype(int).astype(str), plotted_density, marker='o',
            markersize=3, linewidth=linewidth, color=PAR_PALETTE[curr_par], label=curr_par)

ax.set_yscale('log')
ax.set_xlabel('Number of origins')
ax.set_ylabel('Likelihood')
ax.legend(title='Participant', loc='upper left', bbox_to_anchor=(1.05, 1),
          frameon=False, borderaxespad=0)
handles, labels = ax.get_legend_handles_labels()
labels = [x + '\n(Non-responder)' if x == '1HD9K' else x for x in labels]
ax.legend(handles, labels, title='Participant', loc='upper left', bbox_to_anchor=(1.05, 1),
          frameon=False, borderaxespad=0)


plt.subplots_adjust(wspace=0.2, top =0.9, bottom=0.3, left=0.15, right=0.70)
plt.savefig(out_dir + 'iihh_likelihood_plot.png', dpi=300)
plt.close()
