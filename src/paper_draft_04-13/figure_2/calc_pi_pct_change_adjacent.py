import os
import sys
sys.path.append('../../../bin/')
sys.path.append('../../../bin/wrappers/')
sys.path.append('../../../data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import dataset_metadata


# In this file I start from the windowed diversity data produced by
# calc_diversity.py (10-1074_pi_vs_time.csv). For each participant / window I
# calculate the percent change in diversity relative to D0 at every timepoint.
# I then subset the result to the two windows directly adjacent to (flanking)
# the hxb2 resistance sites.

WINDOW_SIZE = 60

# Here is the input/output folder
ioDir = '../../../results/paper_draft_04-13/figure_2/'

# The first timepoint post viral-load nadir for each participant
comp_time_file = '../../../results/paper_draft_04-13/figure_1/closest_post_nadir_timepoints.csv'

# Rows with a percent decrease in diversity larger than this magnitude get
# flagged in the output
PCT_DECREASE_THRESH = 10

# The resistance sites in hxb2 nucleotide coordinates. We take the full span
# covered by the resistance sites so we can find the windows on either side.
hxb2_res_positions = dataset_metadata.RESISTANCE_POS_NT_HXB2
res_min = min(pos[0] for pos in hxb2_res_positions)
res_max = max(pos[1] for pos in hxb2_res_positions)

# Load the windowed diversity data made by calc_diversity.py
all_pi_df = pd.read_csv(ioDir + '10-1074_pi_vs_time.csv')

# Calculate the percent change in diversity relative to D0 for each
# participant / window / timepoint. A window is identified by its arr_start.
pct_change_df = []
for (curr_par, arr_start), group in all_pi_df.groupby(['participant', 'arr_start']):
    d0_rows = group[group['time'] == 'D0']
    if d0_rows.empty:
        continue
    d0_pi = d0_rows['pi'].values[0]

    for _, row in group.iterrows():
        pct_change = (row['pi'] - d0_pi) / d0_pi * 100 if d0_pi != 0 else np.nan
        pct_change_df.append([curr_par, row['time'], row['hxb2_start'],
                              row['hxb2_end'], arr_start, row['pi'], d0_pi,
                              pct_change])

pct_change_df = pd.DataFrame(pct_change_df,
                             columns=['participant', 'time', 'hxb2_start',
                                      'hxb2_end', 'arr_start', 'pi', 'd0_pi',
                                      'pi_change_percent'])


# Now subset to the two windows directly adjacent to the resistance sites: the
# non-overlapping window ending just before the resistance region (left) and
# the non-overlapping window starting just after it (right). The hxb2
# coordinates of each window vary slightly between participants, so we pick
# these windows per participant.
adjacent_rows = []
for curr_par, group in pct_change_df.groupby('participant'):
    windows = group[['arr_start', 'hxb2_start', 'hxb2_end']].drop_duplicates()

    # Left flanking window: the window with the largest hxb2_end that still
    # falls at or before the start of the resistance region.
    left = windows[windows['hxb2_end'] <= res_min]
    if not left.empty:
        left_arr = left.loc[left['hxb2_end'].idxmax(), 'arr_start']
        left_rows = group[group['arr_start'] == left_arr].copy()
        left_rows['window'] = 'left'
        adjacent_rows.append(left_rows)

    # Right flanking window: the window with the smallest hxb2_start that falls
    # at or after the end of the resistance region.
    right = windows[windows['hxb2_start'] >= res_max]
    if not right.empty:
        right_arr = right.loc[right['hxb2_start'].idxmin(), 'arr_start']
        right_rows = group[group['arr_start'] == right_arr].copy()
        right_rows['window'] = 'right'
        adjacent_rows.append(right_rows)

adjacent_df = pd.concat(adjacent_rows, ignore_index=True)

# Subset to the first timepoint post viral-load nadir for each participant.
# The closest_post_nadir column gives the week (0 -> D0, otherwise W<week>).
comp_time_df = pd.read_csv(comp_time_file)
nadir_time_dict = dict(zip(comp_time_df['participant'],
                           comp_time_df['closest_post_nadir']))
# Only keep participants for which we have a post-nadir timepoint
adjacent_df = adjacent_df[adjacent_df['participant'].isin(nadir_time_dict)]
adjacent_df['nadir_time'] = adjacent_df['participant'].map(nadir_time_dict)
adjacent_df['nadir_time'] = ['W' + str(int(x)) if x != 0 else 'D0'
                             for x in adjacent_df['nadir_time']]
adjacent_df = adjacent_df[adjacent_df['time'] == adjacent_df['nadir_time']]

# Flag rows with a percent decrease in diversity greater than the threshold
adjacent_df['pct_decrease_over_thresh'] = \
    adjacent_df['pi_change_percent'] < -PCT_DECREASE_THRESH

# Sort and save the results
adjacent_df = adjacent_df.sort_values(['pct_decrease_over_thresh', 'participant', 'window', 'time'])
adjacent_df.to_csv(ioDir + '10-1074_pi_pct_change_adjacent_windows.csv',
                   index=False)


