import os
import re
import glob

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# All simulations from the 06-25-2026 origin-counts pipeline. The SLiM script
# emits every m4 (escape) mutation in the full population via outputMutations()
# at the final tick, so each distinct m4 mutation is one line of the form:
#   #OUT: <tick> <cycle> T <subpop> <mutid> m4 <pos> <s> <h> <orig_subpop> <orig_tick> <count> <nuc>
# Counting the distinct m4 mutation ids gives the number of surviving origins
# for a simulation.
SIMS_DIR = '../../../../data/slim_simulations/06-25-2026_sims/'
OUT_DIR = '../../../../results/paper_draft_04-13/supp_figs/param_sweep_origin_counts/'

os.makedirs(OUT_DIR, exist_ok=True)

# Directory names look like:
#   origins_100_min_80_rep_0_mu_1e-05_sel_0.1_rho_1e-05
DIR_RE = re.compile(
    r'origins_(?P<origins>\d+)_min_(?P<min>\d+)_rep_(?P<rep>\d+)'
    r'_mu_(?P<mu>[\de.+-]+)_sel_(?P<sel>[\d.]+)_rho_(?P<rho>[\de.+-]+)'
)


def count_unique_m4(slim_output_file):
    """Return the number of distinct m4 mutation ids at the final tick."""
    mut_ids_by_tick = {}
    with open(slim_output_file) as f:
        for line in f:
            if not line.startswith('#OUT:'):
                continue
            fields = line.split()
            # fields: '#OUT:' tick cycle marker subpop mutid type pos s h
            #          orig_subpop orig_tick count nuc
            if len(fields) < 14 or fields[3] != 'T' or fields[6] != 'm4':
                continue
            tick = int(fields[1])
            mut_ids_by_tick.setdefault(tick, set()).add(int(fields[5]))
    if not mut_ids_by_tick:
        return 0
    final_tick = max(mut_ids_by_tick)
    return len(mut_ids_by_tick[final_tick])


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
        'mu': match.group('mu'),
        'sel': float(match.group('sel')),
        'rho': match.group('rho'),
        'n_m4': count_unique_m4(slim_output),
    })

df = pd.DataFrame(records)
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Plot grid: single row, one subplot per selection coefficient. Within each
# subplot, a stripplot of the number of unique m4 mutations against the number
# of origins introduced (one point per simulation, i.e. per mu/rho combo).
# ---------------------------------------------------------------------------
sel_values = sorted(df['sel'].unique())
origin_order = sorted(df['origins'].unique())

fig, axes = plt.subplots(1, len(sel_values),
                         figsize=(5 * len(sel_values), 4.5),
                         sharey=True, squeeze=False)
axes = axes[0]

for ax, sel in zip(axes, sel_values):
    sub = df[df['sel'] == sel]
    sns.stripplot(data=sub, x='origins', y='n_m4', order=origin_order,
                  hue='origins', palette='viridis', legend=False,
                  jitter=True, size=7, alpha=0.8, edgecolor='black',
                  linewidth=0.5, ax=ax)

    # Reference lines, plotted against the categorical x positions (0..N-1).
    positions = range(len(origin_order))
    # Number of origins introduced (the ceiling: y == x).
    ax.plot(positions, origin_order, color='black', linestyle='--',
            marker='o', label='Intended origin number')
    # Mean number of surviving unique escape mutations per origin group.
    means = [sub[sub['origins'] == o]['n_m4'].mean() for o in origin_order]
    ax.plot(positions, means, color='crimson', linestyle='-',
            marker='s', label='Mean # unique escape mutations')

    ax.set_title(f's = {sel}')
    ax.set_xlabel('Number of origins introduced')
    ax.legend(loc='upper left', frameon=False)

axes[0].set_ylabel('Number of unique escape mutations (final tick)')

fig.suptitle('Surviving escape origins by selection coefficient')
plt.tight_layout()
out_path = OUT_DIR + 'param_sweep_escape_origins.png'
fig.savefig(out_path, dpi=300)
print('Saved figure to', out_path)
