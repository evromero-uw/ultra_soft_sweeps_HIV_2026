import glob
import sys
all_paths = glob.glob('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/*')
sys.path.extend(all_paths)
sys.path.append('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin/')
sys.path.append('/net/feder/vol1/home/evromero/2025_hiv_linkage/data/clyde_westfall_2024_final/10-1074/')
import numpy as np
import pandas as pd
import seaborn as sns
import data_util as du
import dataset_metadata
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import colors

import networkx as nx
from networkx.algorithms import bipartite



def heatmap_mask(seqArr, gap_cols = {}):
    """This function helps make highlighter plots using seaborn.heatmap. It 
    takes a sequence array and a set of columns to be filtered out. Then, it 
    looks at each position in the sequences and replaces the majority 
    nucleotide at that position with a '.' character. 
    This way the majority nucleotide will be white on the heatmap.
    ---------------------------------------------------------------------------
    """
    #loop through the columns to get the segregating sites and their positions
    outArr = []

    for i in range(seqArr.shape[1]):
        #if it is a gap column, replace the whole column with '.'
        if i in gap_cols:
            curr_col = np.full(seqArr.shape[0], '-')
            outArr.append(curr_col)
            continue

        #get the column
        curr_col = seqArr[:,i]

        #find the majority nucleotide
        majority_nuc = pd.Series(curr_col)
        majority_nuc = majority_nuc.mode()[0]

        #replace the majority nucleotide with a '.'
        np.place(curr_col, curr_col == majority_nuc , '.')

        outArr.append(curr_col)
       
    outArr = np.array(outArr)
    outArr = np.transpose(outArr)

    return outArr

def make_highliter_plot_unclustered(outFile, title, seqArr, arr_res_pos,
                                    arr_to_hxb2, hxb2_to_arr,
                                    cut_coords = None,
                                    horizontal_divider = None,
                                    gap_cols = {}):
    """ Given a sequence array and the positions of the resistance mutations,
    makes a highliter plot with vertical lines demarcating the resistance 
    mutation positions.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    outFile: string, the path to the output file
    title: string, the title of the plot
    seqArr: numpy array, the sequence array where each row is a sequence and
            each column is a nucleotide position
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    arr_to_hxb2: dict, a dictionary where the key is the index in the sequence
                array and the value is the corresponding hxb2 position
    hxb2_to_arr: dict, a dictionary where the key is the hxb2 position and the
                value is the corresponding index in the sequence array
    cut_coords: tuple, a tuple containing two integers that demarcate the start
                and end positions (start, end) of the region to be plotted in
                HXB2 coordinates. Under the default None setting, the full
                array will be plotted.
    horizontal_divider: list, if True, will put a horizontal line in the
                    heatmap at each of the specified positions.
    gap_cols: set, a set containing the column indices that represent loci where
                the sequences contain more gaps than is tolerable (as computed by
                du.make_gap_col_set). This set is empty by default.
    ax: matplotlib axis object, if specified, the plot will be made on this
                axis. Otherwise, a new axis will be created.
    """
    
    heatmap_arr = heatmap_mask(seqArr.copy(), gap_cols=gap_cols)
    column_indices = list(range(0, heatmap_arr.shape[1]))
   

    column_indices = [round(arr_to_hxb2[i], 4) for i in column_indices]
    heatmap_arr = pd.DataFrame(heatmap_arr, columns=column_indices)
   
    loop_cols = heatmap_arr.columns
    
    #Get only the region of interest if coordinates are given
    if cut_coords:
        for curr_col in loop_cols:
            if curr_col < cut_coords[0] or curr_col > cut_coords[1]:
                heatmap_arr = heatmap_arr.drop(columns = curr_col)
    

    value_to_int = {'.': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5} 
    cmap = sns.color_palette("Dark2", 5, as_cmap=True)




    ax = sns.heatmap(heatmap_arr.replace(value_to_int), cmap=cmap, 
                    mask=heatmap_arr.replace(value_to_int) <= 0)
    
    resistance_heatmap_color(ax, seqArr, arr_res_pos, arr_to_hxb2,
                            hxb2_to_arr, cut_coords = cut_coords)

    #Make a window around all the resistance sites
    start_pos_1 = min(arr_res_pos[0][0], arr_res_pos[1][0])
    end_pos_2 = max(arr_res_pos[0][1], arr_res_pos[1][1])


    if cut_coords:
        first_cut = hxb2_to_arr[cut_coords[0]]
        start_pos_1 = start_pos_1 - first_cut
        end_pos_2 = end_pos_2 - first_cut + 1
  

    #Put vertical lines in to mark the resistance window
    ax.axvline(start_pos_1, color = 'black', linewidth = 0.5)
    ax.axvline(end_pos_2, color = 'black', linewidth = 0.5)

    #Put a horizontal line if specified
    for curr_divider in horizontal_divider:
        ax.axhline(curr_divider, color = 'black', linewidth = 1)


    #modify colorbar:
    colorbar = ax.collections[0].colorbar 
    r = colorbar.vmax - 0 
    colorbar.set_ticks([0 + r / len(value_to_int) * (0.5 + i) for i in range(len(value_to_int))])
    colorbar.set_ticklabels(list(value_to_int.keys())) 
    plt.title(title)
    plt.ylabel('Sequence')
    plt.xlabel('HXB2 Position')
    # plt.tight_layout()
    plt.savefig(outFile)
    plt.close()


def make_highliter_plot_clustered(outFile, title, seqArr, arr_res_pos,
                                    arr_to_hxb2, hxb2_to_arr, seq_info_df,
                                    cut_coords = None,
                                    horizontal_divider = None,
                                    gap_cols = {},
                                    res_color = False):
    """ Given a sequence array and the positions of the resistance mutations,
    makes a highliter plot with vertical lines demarcating the resistance 
    mutation positions. This version of the function facets the final plot 
    by cluster label and timepoint
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    outFile: string, the path to the output file
    title: string, the title of the plot
    seqArr: numpy array, the sequence array where each row is a sequence and
            each column is a nucleotide position
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    arr_to_hxb2: dict, a dictionary where the key is the index in the sequence
                array and the value is the corresponding hxb2 position
    hxb2_to_arr: dict, a dictionary where the key is the hxb2 position and the
                value is the corresponding index in the sequence array
    seq_info_df: pd.DataFrame, a dataframe containing the sequence metadata,
                including the cluster label and timepoint.
    cut_coords: tuple, a tuple containing two integers that demarcate the start
                and end positions (start, end) of the region to be plotted in
                HXB2 coordinates. Under the default None setting, the full
                array will be plotted.
    horizontal_divider: list, if True, will put a horizontal line in the
                    heatmap at each of the specified positions.
    gap_cols: set, a set containing the column indices that represent loci where
                the sequences contain more gaps than is tolerable (as computed by
                du.make_gap_col_set). This set is empty by default.
    res_color: bool, if True, the sequences will be colored by the resistance
                mutations. Default is False.
    """
    value_to_int = {'.': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5} 
    cmap = colors.Colormap("Dark2", 5)
    cmap = colors.ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:gray'], 'Meh', N=5)

    heatmap_arr = heatmap_mask(seqArr.copy(), gap_cols=gap_cols)
    
    column_indices = list(range(0, heatmap_arr.shape[1]))
    column_indices = [round(arr_to_hxb2[i], 4) for i in column_indices]

    heatmap_arr = pd.DataFrame(heatmap_arr, columns=column_indices)
   
    loop_cols = heatmap_arr.columns
    
    #Get only the region of interest if coordinates are given
    if cut_coords:
        for curr_col in loop_cols:
            if curr_col < cut_coords[0] or curr_col > cut_coords[1]:
                heatmap_arr = heatmap_arr.drop(columns = curr_col)

    #Now, loop through the dataframe and make a plot for each cluster label with more than
    #one sequence
    cluster_labels = seq_info_df['cluster_label'].unique()
    print("# Clusters before Filtering: " + str(len(cluster_labels)))


    filtered_cluster_labels = []
    for curr_cluster in cluster_labels:
        if len(seq_info_df[seq_info_df['cluster_label'] == curr_cluster]) > 2:
            filtered_cluster_labels.append(curr_cluster)
    cluster_labels = filtered_cluster_labels
    print("# Clusters after Filtering: " + str(len(cluster_labels)))


    time_labels = seq_info_df['time_label'].unique().tolist()
    time_labels.sort(key = lambda x: int(x[1:]))



    fig, axs = plt.subplots(np.min([4, len(cluster_labels)]), len(time_labels),
                            figsize = (20, 10))

    for i in range(0, len(cluster_labels)):
        curr_cluster = cluster_labels[i]
        
        curr_cluster_df = seq_info_df[seq_info_df['cluster_label'] == curr_cluster]

        for j in range(len(time_labels)):
            curr_time = time_labels[j]
            
            #Get the sequences for the current cluster and timepoint
            curr_time_df = curr_cluster_df[curr_cluster_df['time_label'] == curr_time]

            if curr_time_df.shape[0] == 0:
                continue

            #Sort the sequences by the resistance mutations
            if res_color:
                mut_order = drm_order(curr_time_df['res_muts'].values)
                new_df_list = []
                for curr_item in mut_order:
                    tuple_vals = curr_time_df['res_muts'].apply(tuple)
                    new_df_list.append(curr_time_df[tuple_vals == curr_item])
                curr_time_df = pd.concat(new_df_list)


            
    
            if len(cluster_labels) == 1:
                curr_ax = axs[j]
            else:
                curr_ax = axs[i%4, j]
            curr_inds = curr_time_df['seq_index'].values.flatten()    
 
    
            curr_heat_arr = heatmap_arr.iloc[curr_inds]
            curr_heat_arr.reset_index(drop = True, inplace = True)
            curr_seq_arr = seqArr[curr_inds, :]

            sns.heatmap(curr_heat_arr.replace(value_to_int), cmap = cmap,
                        mask=curr_heat_arr.replace(value_to_int) <= 0, ax = curr_ax)

            #Color the heatmap by resistance mutations
            if res_color:
                resistance_heatmap_color(curr_ax, curr_seq_arr, arr_res_pos,
                                          arr_to_hxb2, hxb2_to_arr, 
                                          cut_coords = cut_coords)
            



            #Make a window around all the resistance sites
            start_pos_1 = min(arr_res_pos[0][0], arr_res_pos[1][0])
            end_pos_2 = max(arr_res_pos[0][1], arr_res_pos[1][1])


            if cut_coords:
                first_cut = hxb2_to_arr[cut_coords[0]]
                start_pos_1 = start_pos_1 - first_cut
                end_pos_2 = end_pos_2 - first_cut + 1
    
            #Put a horizontal line if specified
            if horizontal_divider:
                for curr_divider in horizontal_divider:
                    curr_ax.axhline(curr_divider, color = 'black', linewidth = 1)
            
            for curr_ind in range(1, len(curr_heat_arr)):
                if curr_ind % 5 == 0:
                    curr_ax.axhline(curr_ind, color = 'black', linewidth = 0.1)

            if i % 4 == 0:
                curr_ax.set_title(curr_time) 
            if i % 4 != 3:
                curr_ax.set_xticks([])
            if len(curr_heat_arr) > 5:
                curr_ax.set_yticks([])
                

            # # #modify colorbar:
            colorbar = curr_ax.collections[0].colorbar 
            
            # if i == 0 and j == 0:
            #     r = colorbar.vmax + 1
            #     colorbar.set_ticks([0 + r / len(value_to_int) * (1 + i) for i in range(5)])
            #     colorbar.set_ticklabels(['A', 'C', 'G', 'T', '-']) 
            # else:
            #     colorbar.remove()
            colorbar.remove()
            

            curr_ax.set_ylabel('Sequence')
            curr_ax.set_xlabel('HXB2 Position')
            curr_ax.set_yticklabels([])

        if i % 4 == 3 and i != 0:
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(outFile + '_' + str(i) + '.png')
            plt.close()

            if i != len(cluster_labels) - 1:
                fig, axs = plt.subplots(4, len(time_labels),
                                figsize = (20, 10))
    if len(cluster_labels) % 4 != 0:
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(outFile + '_' + str(len(cluster_labels)-1) + '.png')
        plt.close()
    return

def make_highliter_plot_collapsed(outFile, title, seqArr, arr_res_pos,
                                    arr_to_hxb2, hxb2_to_arr, seq_info_df,
                                    cut_coords = None,
                                    horizontal_divider = None,
                                    gap_cols = {},
                                    res_color = False,
                                    amino_acid = False,
                                    verbose = False,
                                    simulation_times = False):
    """ Given a sequence array and the positions of the resistance mutations,
    makes a highliter plot with vertical lines demarcating the resistance 
    mutation positions. This version of the function facets the final plot 
    by timepoint only, rather than both cluster label and
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    outFile: string, the path to the output file
    title: string, the title of the plot
    seqArr: numpy array, the sequence array where each row is a sequence and
            each column is a nucleotide position
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    arr_to_hxb2: dict, a dictionary where the key is the index in the sequence
                array and the value is the corresponding hxb2 position
    hxb2_to_arr: dict, a dictionary where the key is the hxb2 position and the
                value is the corresponding index in the sequence array
    seq_info_df: pd.DataFrame, a dataframe containing the sequence metadata,
                including the cluster label and timepoint.
    cut_coords: tuple, a tuple containing two integers that demarcate the start
                and end positions (start, end) of the region to be plotted in
                HXB2 coordinates. Under the default None setting, the full
                array will be plotted.
    horizontal_divider: list, if True, will put a horizontal line in the
                    heatmap at each of the specified positions.
    gap_cols: set, a set containing the column indices that represent loci where
                the sequences contain more gaps than is tolerable (as computed by
                du.make_gap_col_set). This set is empty by default.
    res_color: bool, if True, the sequences will be colored by the resistance
                mutations. Default is False.
    amino_acid: bool, if True, the sequences will be colored by the amino acid
    simulation_times: bool, if True, the time labels will integers indicating
                simulationg generations rather than the timepoint label.
                Default is False.
    """
    if amino_acid:
        palette = [
        '#C8C8C8', '#145AFF', '#00DCDC', '#E60A0A', '#E6E600',
        '#00DCDC', '#E60A0A', '#EBEBEB', '#8282D2', '#0F820F', 
        '#0F820F', '#145AFF', '#E6E600', '#3232AA', '#DC9682', 
        '#FA9600', '#FA9600', '#B45AB4', '#3232AA', '#0F820F', 
        '#FFFFFF']
        cmap = colors.ListedColormap(palette, 'Meh', N=21)
        value_to_int = {'.':0, 'A':1, 'R':2, 'N':3, 'D':4,
                        'C':5, 'Q':6, 'E':7, 'G':8, 'H':9,
                        'I':10, 'L':11, 'K':12, 'M':13, 'F':14,
                        'P':15, 'S':16, 'T':17, 'W':18, 'Y':19,
                        'V':20, '-':21}
        hue_order = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                        'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '-']
        vmax = 21
    else:
        value_to_int = {'.': 0, 'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5} 
        hue_order = ['A', 'C', 'G', 'T', '-']
        vmax = 5
        cmap = colors.ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:pink', 'tab:gray'], 'Meh', N=5)



    heatmap_arr = heatmap_mask(seqArr.copy(), gap_cols=gap_cols)
    if verbose:
        print("Heatmap Arr " + heatmap_arr)
    
    column_indices = list(range(0, heatmap_arr.shape[1]))
    column_indices = [round(arr_to_hxb2[i], 4) for i in column_indices]

    heatmap_arr = pd.DataFrame(heatmap_arr, columns=column_indices)
   
    loop_cols = heatmap_arr.columns
    
    #Get only the region of interest if coordinates are given
    if cut_coords:
        for curr_col in loop_cols:
            if curr_col < cut_coords[0] or curr_col > cut_coords[1]:
                heatmap_arr = heatmap_arr.drop(columns = curr_col)

    #Now, loop through the dataframe and make a plot for each time label
    seq_info_df = seq_info_df.drop(axis = 0, index = 0)
    cluster_labels = seq_info_df['cluster_label'].unique()

    time_labels = seq_info_df['time_label'].unique().tolist()
    if simulation_times:
        time_labels.sort()
    else:
        time_labels.sort(key = lambda x: int(x[1:]))



    fig, axs = plt.subplots(1, len(time_labels),
                            figsize = (20, 10))
    axs = axs.flatten()


    for i in range(len(time_labels)):
        curr_ax = axs[i]
        curr_time = time_labels[i]
        
        #Get the sequences for the current cluster and timepoint
        curr_time_df = seq_info_df[seq_info_df['time_label'] == curr_time]

        if curr_time_df.shape[0] == 0:
            continue

        #Sort the sequences by cluster label
        curr_seq_arr = []
        curr_heat_arr = []

        #Sort the sequences by the cluster label
        curr_time_df = curr_time_df.sort_values(by = 'cluster_label')
        curr_inds = curr_time_df['seq_index'].values.flatten()    
        curr_heat_arr = heatmap_arr.iloc[curr_inds]

        for index, row in curr_time_df.iterrows():
            curr_seq_arr.append(seqArr[row['seq_index'], :])
        curr_seq_arr = np.array(curr_seq_arr)

        

        sns.heatmap(curr_heat_arr.replace(value_to_int), cmap = cmap,
                    mask=curr_heat_arr.replace(value_to_int) <= 0,
                    vmin = 1, vmax = vmax,
                    ax = curr_ax)

        #Color the heatmap by resistance mutations
        if res_color:
            resistance_heatmap_color(curr_ax, curr_seq_arr, arr_res_pos,
                                        arr_to_hxb2, hxb2_to_arr, 
                                        cut_coords = cut_coords)
        


        #Make a window around all the resistance sites
        if res_color:
            start_pos_1 = min(arr_res_pos[0][0], arr_res_pos[1][0])
            end_pos_2 = max(arr_res_pos[0][1], arr_res_pos[1][1])
            curr_ax.axvline(start_pos_1, color = 'black')
            curr_ax.axvline(end_pos_2, color = 'black')


        if cut_coords:
            first_cut = hxb2_to_arr[cut_coords[0]]
            start_pos_1 = start_pos_1 - first_cut
            end_pos_2 = end_pos_2 - first_cut + 1

        #Put a horizontal line if specified
        if horizontal_divider:
            for curr_divider in horizontal_divider:
                curr_ax.axhline(curr_divider, color = 'black', linewidth = 1)
        
        for curr_ind in range(1, len(curr_heat_arr)):
            if curr_ind % 5 == 0:
                curr_ax.axhline(curr_ind, color = 'black', linewidth = 0.1)

        curr_ax.set_title(curr_time) 
            

        # # # #modify colorbar:
        # Get the colorbar object from the Seaborn heatmap
        colorbar = curr_ax.collections[0].colorbar
        # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
        r = colorbar.vmax - colorbar.vmin
        n = len(hue_order)
        colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n) + r * i / (n) for i in range(n)])
        colorbar.set_ticklabels(hue_order)

        curr_ax.set_ylabel('Sequence')
        curr_ax.set_xlabel('HXB2 Position')
        curr_ax.set_yticklabels([])

        if i % 4 == 3 and i != 0:
            plt.suptitle(title)
            plt.tight_layout()
            plt.savefig(outFile + '_' + str(i) + '.png')
            plt.close()

            if i != len(cluster_labels) - 1:
                fig, axs = plt.subplots(4, len(time_labels),
                                figsize = (20, 10))
    if len(cluster_labels) % 4 != 0:
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(outFile + '.png')
        plt.close()
    return



                                    
###############################################################################
################################ Helper Functions #############################
###############################################################################
def organize_by_resistance(seqArr, ):
    pass

def sort_helper(seq):
    """ Code from stack overflow https://stackoverflow.com/questions/480214/how-do-i-remove-duplicates-from-a-list-while-preserving-order
    to remove duplicates from a list while preserving order"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def drm_order(drm_list):
    """Orders a list of DRMs by site and amino acid. It will return a list of
    tuples, rather than a list of lists."""
    sublists_sorted = []
    #Define the sorting functions
    position = lambda x: int(x[-3:-1])
    amino_acid = lambda x: x[-1]
    
    #Loop through the list and sort the sublists
    for curr_drm in drm_list:
        if len(curr_drm) == 1:
            sublists_sorted.append(curr_drm)
        else:
            curr_drm = sorted(curr_drm, key = lambda x: (position(x), amino_acid(x)))
            sublists_sorted.append(curr_drm)
    

    #Now sort across the lists in the sublist
    none_list = [x for x in sublists_sorted if x[0] == None]
    sublists_sorted = [x for x in sublists_sorted if x[0] != None]
    sublists_sorted = sorted(sublists_sorted, key = lambda x: (position(x[0]), amino_acid(x[0])))
    none_list.extend(sublists_sorted)

    #Finally, remove duplicates
    none_list = [tuple(x) for x in none_list]
    none_list = list(dict.fromkeys(none_list))
    none_list = sort_helper(none_list)

    return none_list

def resistance_heatmap_color(plot_ax, seqArr, arr_res_pos, arr_to_hxb2, 
                            hxb2_to_arr, cut_coords = None):
    """Colors a heatmap based on the resistance mutations that are present in
    the sequences.
    """
    arr_res_pos = [arr_res_pos[i] for i in range(len(arr_res_pos))]
    arr_res_pos.sort(key = lambda x: x[0])

    #Loop through and slice out the resistance mutations
    removed_arr_list = []
    for curr_region in arr_res_pos:
        region_start = curr_region[0] 
        region_end = curr_region[1] 
        removed_arr_list.append(seqArr[:, region_start:region_end+1])

    x_pos_min = min(arr_res_pos[0][0], arr_res_pos[1][0])
    x_pos_max = max(arr_res_pos[0][1], arr_res_pos[1][1])
    box_width = x_pos_max - x_pos_min + 1
    x_pos = x_pos_min

    if cut_coords:
        first_cut = hxb2_to_arr[cut_coords[0]]
        x_pos = x_pos - first_cut
    

    
    #Now for each sequence in the array, check if it has a DRM
    for i in range(seqArr.shape[0]):
        first_region = removed_arr_list[0]
        first_region = first_region[i,:]
        second_region = removed_arr_list[1]
        second_region = second_region[i,:]
        ypos = i
   
        res_muts = du.identify_resistance_10_1074(first_region, second_region)

        #If the sequence has a DRM, color it on the heatmap
        if res_muts[0] != None:
            if len(res_muts) == 1:
                curr_color = dataset_metadata.RESISTANCE_HUE_DICT[res_muts[0]]
                plot_ax.add_patch(Rectangle((x_pos, ypos), box_width,
                                                         1, alpha = 0.3, 
                                                         color = curr_color))
            elif len(res_muts) == 2:
                color_1 = dataset_metadata.RESISTANCE_HUE_DICT[res_muts[0]]
                color_2 = dataset_metadata.RESISTANCE_HUE_DICT[res_muts[1]]
                plot_ax.add_patch(Rectangle((x_pos, ypos), 
                                                        box_width/2,
                                                        1, alpha = 0.3, 
                                                        color = color_1))
                plot_ax.add_patch(Rectangle((x_pos + box_width/2, ypos), 
                                                        box_width/2,
                                                        1, alpha = 0.3, 
                                                        color = color_2))
    return

    



