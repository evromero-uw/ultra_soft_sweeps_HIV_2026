import sys
sys.path.append('../bin/')
import numpy as np
import pandas as pd
import data_util as du
from scipy.spatial.distance import hamming

#This file will contain utilities for calculating various linkage statistics
#describing the data set.

def count_drms_per_hap(seqArr, hxb2_res_positions, arr_res_pos,
                       allele_freq_thresh):
    """Given a sequence array of either simulated or in vivo data and a list
    of resistance sites, this function steps out from the resistance site one
    nucleotide at a time and counts the number of resistance mutations on each
    existing haplotype.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    hxb2_res_positions: tuple, a tuple where each entry is itself a tuple
                containing the start and end positions of the resistance
                mutation in hxb2 coordinates ((start, end) ,(start2, end2)).
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    allele_freq_thresh: float, the minimum minor allele frequency for a site to
                be considered segregating.
    Returns:
    --------
    res_mut_df: pandas.DataFrame, a dataframe containing the number of
                resistance mutations on each haplotype stepping out one snp at
                a time.
    """
    #Split the array into the reference sequence and the rest of the sequences
    refSeq = seqArr[0,:]
    seqArr = seqArr[1:,:]

    #Make a set spanning the range of array resistance positions
    arr_res_set = set(range(arr_res_pos[0][0], arr_res_pos[0][1] + 1))\
                .union(set(range(arr_res_pos[1][0],\
                                arr_res_pos[1][1] + 1)))

    sim_seg_dict = du.get_seg_sites(np.vstack((refSeq, seqArr)),
                                        arr_res_set,
                                        allele_freq_thresh=allele_freq_thresh)

    #Get the closest resistance mutations to each site and sort by distance 
    #from the resistance mutations
    sim_seg_dist_list = du.get_closest_res(sim_seg_dict, arr_res_pos)

    #Make places to store the number of resistance mutations and the number
    #of haplotypes
    hap_counts = []
    res_count_dict = {}

    #Now build haplotypes out from the central sites
    haplotype_list = [seqArr]

    #Get the number of DRMs in the original array
    all_drm_rem = du.mask_hxb2_coords(np.vstack((refSeq,seqArr)),
                                        start_pos = hxb2_res_positions[0][0],
                                        end_pos = hxb2_res_positions[0][1],
                                        arr_start_pos = 1,
                                        second_site = (hxb2_res_positions[1][0],
                                                    hxb2_res_positions[1][1]))[1]
    all_res_muts = du.count_unique_resistance_10_1074(all_drm_rem)
    #record the number of resistance mutations
    res_count_dict[1] = [len(all_res_muts)]

    for curr_site in sim_seg_dist_list:
        updated_hap_list = []

        #Record the number of haplotypes
        hap_counts.append(len(haplotype_list))
        curr_res_mut_counts = []

        #Split any haplotypes
        for curr_hap in haplotype_list:
            site_pos = curr_site[0]
            site_allele = curr_site[3]
        
            #Find the rows where the allele matches the site allele
            site_index = np.where(curr_hap[:,site_pos] == site_allele)[0]
            other_index = np.where(curr_hap[:,site_pos] != site_allele)[0]
            first_hap = curr_hap[site_index, :]
            second_hap = curr_hap[other_index, :]

            if first_hap.shape[0] > 0:
                #Count unique resistance mutations observed on the haplotype
                #Get the array slices with the resistance mutations
                first_rem=du.mask_hxb2_coords(np.vstack((refSeq,first_hap)),
                                        start_pos = hxb2_res_positions[0][0],
                                        end_pos = hxb2_res_positions[0][1],
                                        arr_start_pos = 1,
                                        second_site = (hxb2_res_positions[1][0],
                                                    hxb2_res_positions[1][1]))[1]
                        
                #Next we need to store the counts of resistance mutations so we can
                #plot them later
                first_res_muts = du.count_unique_resistance_10_1074(first_rem)
                curr_res_mut_counts.append(len(first_res_muts))
                #Add the haplotypes to the updated list
                updated_hap_list.append(first_hap)

            #Also count and store res muts for the second haplotype
            if second_hap.shape[0] > 0:
                second_rem=du.mask_hxb2_coords(np.vstack((refSeq,second_hap)),
                                        start_pos = hxb2_res_positions[0][0],
                                        end_pos = hxb2_res_positions[0][1],
                                        arr_start_pos = 1,
                                        second_site = (hxb2_res_positions[1][0],
                                                    hxb2_res_positions[1][1]))[1]
                sec_res_muts = du.count_unique_resistance_10_1074(second_rem)
                curr_res_mut_counts.append(len(sec_res_muts))
                updated_hap_list.append(second_hap)

        #record the number of resistance mutations
        if len(updated_hap_list) not in res_count_dict.keys():
            res_count_dict[len(updated_hap_list)] = curr_res_mut_counts
        
        #Update the haplotype list
        haplotype_list = updated_hap_list

    res_mut_df = pd.DataFrame.from_dict(res_count_dict, orient='index').T
    res_mut_df = pd.melt(res_mut_df, value_vars=res_mut_df.columns.tolist(),
                        var_name='num_haplotypes', value_name='num_res_muts')
    res_mut_df.dropna(inplace=True)

    return res_mut_df


def calculate_EHH(seqArr, core_hap_labels, seg_dict, arr_res_pos = None,
                  focal_pos = None):
    """ Given a sequence array, a set of core haplotypes deemed extremely
    unlikely to recombine, and a list of the segregating sites this function
    calculates the extended haplotype homozygosity. It then returns the EHH
    values for each segregating position, and a datastructure used for 
    creating a haplotype bifurcation diagram.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    core_hap_labels: np.array, an array where each index contains a number
                indicating the core haplotype of the sequence at that row index
                in seqArr
    segDict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele (position 0) and its frequency (position 1).
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)). These positions
                are used as the focal positions for the EHH calculation.
    focal_pos: tuple, (start, end) the positions in the array where the EHH 
                should be calculated from. If this is not None, the function
                will calculate the EHH from this position rather than the
                resistance mutations.
    """
    #Sort the segregating sites based on their distance from the core haps
    if focal_pos is not None:
        seg_dist_list = du.get_dist_from_region(seg_dict, focal_pos, 
                                                freq_sorted = False)  
    elif arr_res_pos is not None:
        seg_dist_list = du.get_closest_res(seg_dict, arr_res_pos, 
                                        freq_sorted = False)
    else:
        raise ValueError('Either arr_res_pos or focal_pos must be provided')
    

    #Make a dataframe of haplotype labels at each position
    hap_df = [core_hap_labels]
    hap_df_columns = ['core']
    
    #Here is a list of labels where each index corresponds to the row index of
    #a sequence in seqArr and the value is the current haplotype label for that
    #sequence.
    curr_hap_labels = core_hap_labels.copy()
    curr_hap_number = np.max(curr_hap_labels) + 1

    #Now we will make a dataframe where each column is a segregating site and
    #each row is a haplotype indicator for the sequence with the corresponding
    #index in seqArr
    for curr_seg_site in seg_dist_list:
        curr_seg_pos = curr_seg_site[0]
        site_nucs = seqArr[:,curr_seg_pos]

        for curr_hap in np.unique(curr_hap_labels):
            #Get the row indices of the sequences with the current haplotype
            curr_hap_inds = np.argwhere(curr_hap_labels == curr_hap)

            #Check all of the sequences at the current site and see if they
            #match
            curr_seqs = site_nucs[curr_hap_inds].flatten()

            #Otherwise we need to split the current hap
            if len(np.unique(curr_seqs)) > 1:

                for curr_nuc in np.unique(curr_seqs):
                    #Get the row indices of the sequences with the current 
                    #haplotype
                    curr_nuc_inds = np.argwhere(curr_seqs == curr_nuc)
                    curr_label_inds = curr_hap_inds[curr_nuc_inds].flatten()

                    #use numpy where to update the hap labels
                    np.put(curr_hap_labels, curr_label_inds, curr_hap_number)
                    curr_hap_number += 1

        #Now we need to add a column to the dataframe with the 
        #haplotype labels at the position
        hap_df.append(curr_hap_labels.copy())
        hap_df_columns.append(curr_seg_pos)
        
    #Convert the lists we've been making into a dataframe
    hap_df = np.array(hap_df).T
    hap_df = pd.DataFrame(hap_df, columns=hap_df_columns)
    
    #Now we need to calculate the EHH for each core haplotype at each
    #segregating site
    core_hap_labels = np.array(hap_df['core'])

    #Make a dataframe where we will save all EHH calculations
    all_results_df = []

    for curr_seg_site in seg_dist_list:
        seg_site_labels = np.array(hap_df[curr_seg_site[0]])
        
        my_results = ehh_equation(core_hap_labels, seg_site_labels)
        my_results['seg_site'] = curr_seg_site[0]
        all_results_df.append(my_results)
    
    all_results_df = pd.concat(all_results_df, ignore_index=True)
    return all_results_df
    
def get_closest_seq_hamming(seqArr, target_index, candidate_info_df):
    """Given a sequence array, an array index of the sequence of interest,
    and a dataframe containing information about each of the candidates to test
    for similarity, this function returns array index (what row in seqArr) of 
    the candidate sequence(s) with the lowest hamming distance compared to the
    target sequence.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. Traditionally, the resistance 
                regions will be masked, but the function can be called on full
                sequences too.
    target_index: int, the index of the sequence of interest in seqArr
    candidate_info_df: pandas.DataFrame, a dataframe containing information
                about each of the sequences in seqArr that should be tested 
                against the target.
    Returns:
    --------
    closest_seq_list: list, a list of the array indices of the sequences in
                seqArr that are closest to the target sequence.
    min_hamming_dist: int, the minimum hamming distance between the target
                sequence and any of the candidate sequences.
    """
    #Get the target sequence
    target_seq = seqArr[target_index,:]
    #Make a place to store the hamming distances
    hamming_dist_list = []
    curr_seq_inds = []
    #Loop through the candidate sequences
    for index, row in candidate_info_df.iterrows():
        candidate_ind = row['seq_index']
        if isinstance(candidate_ind, list):
            candidate_ind = candidate_ind[0]

        #Get the candidate sequence
        curr_seq = seqArr[candidate_ind,:]

        #Calculate the hamming distance
        curr_hamming_dist = hamming(target_seq, curr_seq)
        hamming_dist_list.append(curr_hamming_dist)
        curr_seq_inds.append(row['seq_index'])

    hamming_dist_list = np.array(hamming_dist_list)
    min_hamming_dist = np.min(hamming_dist_list)

    #Now get the indices of the sequences with the minimum hamming distance
    closest_seq_list = list(np.where(hamming_dist_list == \
                                     min_hamming_dist)[0])

    output = []
    for curr_seq in closest_seq_list:
        output.append(curr_seq_inds[curr_seq])
    closest_seq_list = output
    return closest_seq_list, min_hamming_dist


            
###############################################################################
############################# Helper Functions ################################
###############################################################################   
def ehh_equation(core_haps, site_haps):
    """ Given the list of haplotypes at a segregating site and the list of core
    haplotypes assignments for those same sequences, this function calculates
    the extended haplotype homozygosity for that site as described in Sabeti et
    al. 2002 and Klassman & Gautier 2022.
    """
    core_list = []
    ehh_list = []

    #for each core haplotype
    for curr_core_hap in np.unique(core_haps):
        #get the row indices of the sequences with the current core hap
        curr_core_hap_inds = np.argwhere(core_haps == curr_core_hap)
        assigned_haps = site_haps[curr_core_hap_inds].flatten()

        #get the number of sequences with the core allele
        n_a = curr_core_hap_inds.flatten().shape[0]

        #make a place to store the values in the sum
        hap_sum = 0

        #for each hapotype corresponding with the core
        for curr_assigned_hap in np.unique(assigned_haps):
            #get the row indices of the sequences with the current hap
            curr_assigned_hap_inds = np.argwhere(site_haps == curr_assigned_hap)
            #get the number of sequences with the current hap
            n_k = curr_assigned_hap_inds.flatten().shape[0]
            hap_sum += n_k * (n_k - 1)
        
        curr_core_EHH = n_a * (n_a -1)
        curr_core_EHH = 1/curr_core_EHH
        curr_core_EHH *= hap_sum
        core_list.append(curr_core_hap)
        ehh_list.append(curr_core_EHH)

    return pd.DataFrame(list(zip(core_list, ehh_list)), columns=['core', 'ehh'])

def label_ehh_core_general(seqArr, seq_info_df, region_tuple):
    """ Given a sequence array and a tuple of a core region (start, end) in
    array coordinates, this function labels the core haplotypes and returns an
    array of labels where each index holds the label for the corresponding row
    index in seqArr.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seq_info_df: pandas.DataFrame, a dataframe containing information about the
                sequences in seqArr. 
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    Returns:
    --------
    core_hap_labels: np.array, an array where each index contains a number
                indicating the core haplotype of the sequence at that row index
                in seqArr
    label_conv_dict_rev: dict, a dictionary where the key is the index in the
                core_hap_labels array and the value is the corresponding core
                haplotype.
    """
    #First get the region of interest
    focal_region = seqArr[:, region_tuple[0]:region_tuple[1] + 1]
    unique_haps = np.unique(focal_region, axis=0)

    #Loop through the sequences and make an array of core haplotype labels
    core_hap_labels = np.zeros(seqArr.shape[0])

    #First make a dictionary to convert haplotypes to numbers
    label_conv_dict = {}
    label_conv_dict_rev = {}
    for curr_hap in unique_haps:
        hap_label = len(label_conv_dict)
        label_conv_dict[tuple(curr_hap)] = hap_label
        label_conv_dict_rev[hap_label] = tuple(curr_hap)
    
    #Now loop through the sequences and label them
    for index, row in seq_info_df.iterrows():
        curr_region = focal_region[row['seq_index'], :]
        curr_hap = tuple(curr_region)
        core_hap_labels[row['seq_index']] = label_conv_dict[curr_hap]

    return core_hap_labels, label_conv_dict_rev

def label_ehh_core_10_1074(seqArr, seq_info_df,  arr_res_pos):
    """Given a sequence array and a tuple of resistance positions, labels the 
    core haplotypes (in our case the 10-1074 resistance sites) and returns a 
    an array of labels where each index holds the label for the corresponding
    row index in seqArr.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seq_info_df: pandas.DataFrame, a dataframe containing information about the
                sequences in seqArr. 
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    Returns:
    --------
    core_hap_labels: np.array, an array where each index contains a number
                indicating the core haplotype of the sequence at that row index
                in seqArr
    label_conv_dict_rev: dict, a dictionary where the key is the index in the
                core_hap_labels array and the value is the corresponding core
                haplotype.
    """
    #First we'll get label the resistance sites using existing functionality
    res_region_2 = seqArr[:, arr_res_pos[0][0]:arr_res_pos[0][1] + 1]
    res_region_1 = seqArr[:, arr_res_pos[1][0]:arr_res_pos[1][1] + 1]

    seq_info_df = du.label_resistance_10_1074([res_region_1, res_region_2],
                                                    seq_info_df)
    
    #Loop through the sequences and make an array of core haplotype labels
    core_hap_labels = np.zeros(seqArr.shape[0])    

    #First make a dictionary to convert haplotypes to numbers
    label_conv_dict = {}
    label_conv_dict_rev = {}
    seq_info_df['res_muts'] = [tuple(x) for x in seq_info_df['res_muts']]
    for curr_hap in seq_info_df['res_muts'].unique():
        hap_label = len(label_conv_dict)
        label_conv_dict[curr_hap] = hap_label
        label_conv_dict_rev[hap_label] = curr_hap

    #Now loop through the sequences and label them
    for index, row in seq_info_df.iterrows():
        curr_ind = row['seq_index']
        core_hap_labels[curr_ind] = label_conv_dict[row['res_muts']]

    return core_hap_labels, label_conv_dict_rev



            
        
     






