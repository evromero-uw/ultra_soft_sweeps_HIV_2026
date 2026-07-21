import os
import re
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# All simulations from the 07-11-2026 origin-counts pipeline. The SLiM
# script's establishment-check block runs every tick from burn_in to
# last_sample (or until establishment/loss triggers early deregistration) and
# prints:
#   "Mutation count"
#   <size(sim.mutationsOfType(m4))>
# This is a full-population census of every currently-circulating m4 origin
# (not a 100-individual sample, which can miss rare/low-frequency lineages).
# Taking the last such printed count gives the number of surviving origins for
# a simulation.
SIMS_DIR = '../../../../data/slim_simulations/07-11-2026_sims/'
OUT_DIR = '../../../../results/paper_draft_04-13/supp_figs/matched_sims_origin_counts/'

os.makedirs(OUT_DIR, exist_ok=True)

# Directory names look like:
#   origins_100_min_80_rep_0
DIR_RE = re.compile(
    r'origins_(?P<origins>\d+)_min_(?P<min>\d+)_rep_(?P<rep>\d+)'
)


def count_unique_m4(slim_output_file):
    """Return the last full-population m4 mutation count printed by the SLiM
    script's establishment-check block (i.e. the last "Mutation count" /
    <n> pair in the raw output)."""
    last_count = None
    with open(slim_output_file) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip() == '"Mutation count"':
            last_count = int(lines[i + 1].strip())
    return last_count if last_count is not None else 0


# Collect one row per simulation directory.
records = []
for sim_dir in sorted(glob.glob(os.path.join(SIMS_DIR, 'origins_*'))):
    match = DIR_RE.search(os.path.basename(sim_dir))
    if match is None:
        continue
    slim_output = os.path.join(sim_dir, 'slim_output.txt')
    if not os.path.exists(slim_output):
        print(f'Skipping (no slim_output.txt): {sim_dir}')
        continue
    records.append({
        'origins': int(match.group('origins')),
        'n_m4': count_unique_m4(slim_output),
    })

record_df = pd.DataFrame(records)
print(record_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot grid: single row, one subplot per selection coefficient. Within each
# subplot, a stripplot of the number of unique m4 mutations against the number
# of origins introduced (one point per simulation, i.e. per mu/rho combo).
# ---------------------------------------------------------------------------
origin_order = sorted(record_df['origins'].unique())

fig, ax = plt.subplots(1, 1, sharey=True)


sns.stripplot(data= record_df, x='origins', y='n_m4', order=origin_order,
                hue='origins', palette='viridis', legend=False,
                jitter=True, size=7, alpha=0.3, edgecolor='black',
                linewidth=0.5, ax=ax, marker = '.')

# Reference lines, plotted against the categorical x positions (0..N-1).
positions = range(len(origin_order))
# Number of origins introduced (the ceiling: y == x).
ax.plot(positions, origin_order, color='black', linestyle='--',
        marker='o', label='Intended number of escape mutations')
# Mean number of surviving unique escape mutations per origin group.
means = [record_df[record_df['origins'] == o]['n_m4'].mean() for o in origin_order]
ax.plot(positions, means, color='crimson', linestyle='-',
        marker='s', label='Mean number of unique escape mutations')

ax.set_title('Surviving escape origins\nin HIV matched simulations')
ax.set_xlabel('Number of escape mutations introduced')
ax.legend(loc='upper left', frameon=False)

ax.set_ylabel('Number of unique escape mutations (final tick)')

plt.tight_layout()
out_path = OUT_DIR + 'matched_sims_origin_counts.png'
fig.savefig(out_path, dpi=300)
print('Saved figure to', out_path)