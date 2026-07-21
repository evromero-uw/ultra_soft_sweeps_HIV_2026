import numpy as np
import pandas as pd

#In this file I am calculating rebound times (first VL >= 75% of OG load)
vl_file = '../../../data/clyde_westfall_2024_final/vl_metadata.csv'
out_file = '../../../data/clyde_westfall_2024_final/vl_filtered_rebound.csv'
vl_data = pd.read_csv(vl_file)

#The nadir should be before week 16, since that's during the trial
#Participant 1HC2 initiated ART at week 8, so we will drop any time points after week 8 for this participant




for curr_column in vl_data.columns:
    if curr_column != 'Study_ID':
        converted_vals = []
        for curr_val in vl_data[curr_column]:
            print(curr_val)
            if curr_val[0] == '<':
                converted_vals.append(np.nan)
            else:
                curr_val = ''.join(c for c in curr_val if c.isdigit() or c == '.')
                converted_vals.append(float(curr_val))
        vl_data[curr_column] = converted_vals

#Get the number of columns
num_columns = len(vl_data.columns)
nadir_max_col = vl_data.columns.get_loc('W16') 


#Day 0
rebound_times = []
nadir_times = []
for index, row in vl_data.iterrows():
    day_0 = row['D0']
    rebound = 'None'

    if row['Study_ID'] == '1HC2':
        nadir_max_col = vl_data.columns.get_loc('W8')

    #The nadir should be before week 16, since that's during the trial
    possible_nadir_times = row[7:nadir_max_col]


    nadir_ind = np.argmin(row[7:nadir_max_col])
    nadir = vl_data.columns[nadir_ind + 7]
    if nadir == 'D7':
        nadir = 'W1'

    for i in range(7, num_columns):
        curr_column = vl_data.columns[i]
        if row[curr_column] >= 0.75*day_0:
            rebound = curr_column
            if rebound == 'D7':
                rebound = 'W1'
            break
    rebound_times.append(rebound)
    nadir_times.append(nadir)

vl_data['Rebound'] = rebound_times
vl_data['Nadir'] = nadir_times

vl_data.to_csv(out_file, index=False)