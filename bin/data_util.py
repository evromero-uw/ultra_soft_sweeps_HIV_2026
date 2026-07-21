import os
import sys
import math
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq

###############################################################################
#################### Loading data and making structs ##########################
###############################################################################
def fasta_to_dataStructs(inFile, caskey2017 = False, clyde2024 = False,
                         amino_acid = False):
    """ This function takes a fasta file of sequences from various timepoints
    and for each timepoint. Then, it makes an array and a dataframe which serve
    as the primary datastructures for the analysis. In the array, each row is a
    sequence and each column is a position in that sequence. This array should
    not be modified. The dataframe has all of the info about the sequences and
    has a column that gives the index of the sequence in the array.

    WARNING: Do not touch the array indexing or the column of the dataframe
    that gives the array indexing information
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    inFile: str, the path to the file with the Fasta sequences
    caskey2017: bool, True if the input file is from the original Caskey 2017
                study this argument is just made to deal with the sequence name
                formatting differing.
    clyde2024: bool, True if the input file is from the Clyde 2024 study this
                argument is just made to deal with the sequence name formatting
                differing.

    Returns:
    --------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    """
    all_seqs = SeqIO.parse(open(inFile),'fasta')

    #We will make a dataframe of the sequence metadata
    seq_info_df = []

    #We will also make an array of sequences
    seq_arr = []

    #Loop through each of the sequences and put them in a dataframe
    for fasta in all_seqs:

        name = fasta.id
        
        #Get the timepoint and participant from the sequence name
        if caskey2017:
            seq_info = name.split('.')
            orig_name = name.split('-')[0]
            newlabel = seq_info[3]
            time_label = newlabel.split('-')[1]
            participant = newlabel.split('-')[0]
        elif clyde2024:
            name = name.replace('-', '_')
            seq_info = name.split('_')
            orig_name = name
            time_label = seq_info[2]
            participant = seq_info[1]
        else: 
            seq_info = name.split('-')
            orig_name = seq_info[0]
            newlabel = seq_info[1]
            time_label = newlabel.split('_')[0]
            participant = newlabel.split('_')[1]
            
        #Make the time label uppercase
        time_label = time_label.upper()

        #Append the sequence to the array
        curr_seq = str(fasta.seq)
        curr_seq = curr_seq.upper()
        if not amino_acid:
            curr_seq = curr_seq.replace('U', 'T')
        seq_arr += [list(curr_seq)]
        seq_index = len(seq_arr) - 1

        #Append the sequence info to the dataframe
        seq_info_df.append([orig_name, time_label, 
                             participant, seq_index])

    #Make the dataframe
    seq_info_df = pd.DataFrame(seq_info_df, columns=['orig_name', 'time_label',
                                                     'participant', 'seq_index'])

    #Make the array
    seq_arr = np.array(seq_arr)

    return seq_arr, seq_info_df

def seqArr_to_oneHot(seqArr, seqDF, amino_acid = False):
    """ This function takes a fasta file of sequences from various timepoints
    and for each timepoint, makes a one hot encoded array of sequences. It then
    collapses the array into two dimension so that it fits the input format for
    sklearn algorithms. Also adds a column to the sequence info dataframe that
    gives the index of the sequence in the encoded array.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    amino_acid: bool, True if the input sequences are amino acid sequences.

    Returns:
    --------
    d2_encoded_arr: np.array, an array where each row is a one hot encoded 
                sequence.
    seqDF: the input dataframe modified to include an 'enc_index' column which
                gives the index of the sequence in the encoded array.
    
    """
    #Get rid of HXB2 so we don't cluster it 
    seqArr = seqArr[1:]
    seqDF['enc_index'] = seqDF['seq_index'] - 1

    #For each time point, make an array where each row is a sequence
    if amino_acid:
        encoded_arr = np.array([one_hot_encode_AA(seq) for seq in seqArr])
    else:
        encoded_arr = np.array([one_hot_encode(seq) for seq in seqArr])

    #Collapse the array into two dimensions
    nsamples, nx, ny = encoded_arr.shape
    d2_encoded_arr = encoded_arr.reshape((nsamples,nx*ny))



    return d2_encoded_arr, seqDF

###############################################################################
############################## HXB2 Mapping ###################################
###############################################################################

def get_hxb2_coords(seqArr, start_pos, end_pos, arr_start_pos):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. THE FIRST SEQUENCE IN THE ARRAY
    MUST BE THE HXB2 REFERENCE SEQUENCE. It also takes in hxb2 coordinates for
    positions of interest. Then, it returns the sequence array sliced to 
    include only the positions of interest.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    start_pos: int, the starting position for the region of interest in the
                HXB2 reference
    end_pos: int, the ending position for the region of interest in HXB2

    Returns:
    --------
    sliced_seqArr: np.array, the input array sliced to include only the region
                    of interest.
    sliced_indices_rev: dict, a dictionary where the key is the hxb2 position
                        and the value is the corresponding index in the
                        sequence array. This dictionary contains the correct
                        mapping for the sliced array (not the original array).
    arr_start_pos: int, the starting position of seqArr in hxb2 coordinates
                this argument is necessary so that the coordinate mapping is
                correct when the array is masked.
    """
    hxb2_to_arr, arr_to_hxb2= hxb2_mapping_dict(seqArr, arr_start_pos)

    #Isolate the region surrounding the resistance mutations
    #resistance indices 
    start_hxb2_pos = hxb2_to_arr[start_pos]
    end_hxb2_pos = hxb2_to_arr[end_pos] + 1
    sliced_seqArr = seqArr[:, start_hxb2_pos:end_hxb2_pos]

    hxb2_to_arr, arr_to_hxb2 = hxb2_mapping_dict(sliced_seqArr, start_pos)


    return sliced_seqArr, hxb2_to_arr, arr_to_hxb2

def mask_hxb2_coords(seqArr, start_pos, end_pos, arr_start_pos,
                     second_site = None):
    """
    This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. NOTE: THE FIRST SEQUENCE IN THE
    ARRAY MUST BE THE HXB2 REFERENCE SEQUENCE. It also takes in hxb2 coordinates
    for the starting position of the array and the positions to mask.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    start_pos: int, the starting position (in HXB2 coords) for the region of
                interest in the HXB2 reference
    end_pos: int, the ending position (in HXB2 coords) for the region to mask
    arr_start_pos: int, the starting position of seqArr in hxb2 coordinates
                this argument is necessary so that the coordinate mapping is
                correct when the array is masked.
    second_site: tuple, the start and end position of a second site to mask

    Returns:
    --------
    maskedArr: np.array, the input array with the region between start and end
                positions sliced out. 
    removedArrList: list of np.arrays, an list containing the arrays of the 
                sequences at the sites that were removed from the input array. 
                The arrays are organized in the list by starting position so 
                the zero element in the list corresponds to the lowest start
                position. NOTE: the first row in each array is the sequence
                of HXB2 in the resistance region.

    """
    #Get the indices of the positions to mask
    hxb2_to_arr, arr_to_hxb2 = hxb2_mapping_dict(seqArr, arr_start_pos)
    start_arr_pos_1= hxb2_to_arr[start_pos]
    end_arr_pos_1 = hxb2_to_arr[end_pos] + 1

    if second_site:
        start_arr_pos_2 = hxb2_to_arr[second_site[0]]
        end_arr_pos_2 = hxb2_to_arr[second_site[1]] + 1

        #Delete the further slice first to preserve the indexing
        if start_arr_pos_1 < start_arr_pos_2:
            maskedArr = np.delete(seqArr, np.s_[start_arr_pos_2:end_arr_pos_2],
                                    axis=1)
            maskedArr = np.delete(maskedArr, np.s_[start_arr_pos_1:end_arr_pos_1],
                                    axis=1)
            removedArrList = [seqArr[:, start_arr_pos_1:end_arr_pos_1],
                            seqArr[:, start_arr_pos_2:end_arr_pos_2]]
            
        else:
            maskedArr = np.delete(seqArr, np.s_[start_arr_pos_1:end_arr_pos_1],
                                    axis=1)
            maskedArr = np.delete(maskedArr, np.s_[start_arr_pos_2:end_arr_pos_2],
                                    axis=1)
            removedArrList = [seqArr[:, start_arr_pos_2:end_arr_pos_2],
                            seqArr[:, start_arr_pos_1:end_arr_pos_1]]

    else:
        #Mask the array
        maskedArr = np.delete(seqArr, np.s_[start_arr_pos_1:end_arr_pos_1],
                               axis=1)
        removedArrList = seqArr[:, start_arr_pos_1:end_arr_pos_1]

    return maskedArr, removedArrList

###############################################################################
############################## Resistance Mutations ###########################
###############################################################################

def label_resistance_10_1074(removedArrList, seqDF):
    """ Takes an array where each row is a sequence and each column is a
    resistance site (such as the one produced by mask_hxb2_coords) and
    identifies whether each sequence contains resistance mutations. This
    function is specific to the 10-1074 bnab. The first array in the list
    must contain the 3 nucleotides coding for hxb2 site 325 while the second 
    array in the list must contain the 9 nucleotides coding for sites 332-334.
    Then it labels the sequences in the dataframe with the resistance mutations
    that they contain.  
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    removedArr: list of two np.arrays, a list where each array codes for the 
                amino acids at the resistance sites mentioned in the docstring
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    
    Returns:
    --------
    seqDF: pd.DataFrame, the input dataframe with a column added that indicates
                whether the sequence contains resistance mutations.
    """
    if not removedArrList[1].shape[-1] == 9:
        raise ValueError('The second array in the input list must have 9 \
                         columns, representing the 3 amino acids at\
                         positions 332-334')
    if not removedArrList[0].shape[-1] == 3:
        raise ValueError('The first array in the input list must have 3\
                         columns, representing the amino acid at position 325')

    
    #Make a list to store the resistance mutations
    res_muts = []

    #Loop through each sequence and check it individually
    for i in range(removedArrList[0].shape[0]):
        first_seq = removedArrList[0][i]
        second_seq = removedArrList[1][i]
        curr_muts = identify_resistance_10_1074(first_seq, second_seq)

        res_muts.append(curr_muts)
    
    #Next, add the sequence labels to the sequence info dataframe
    all_labels = []
    for index, row in seqDF.iterrows():
        curr_ind = row['seq_index']
        curr_res = res_muts[curr_ind]
        all_labels.append(curr_res)

    seqDF['res_muts'] = all_labels
    return seqDF

def count_unique_resistance_10_1074(removedArrList):
    """
    Takes an array where each row is a sequence and each column is a
    resistance site (such as the one produced by mask_hxb2_coords) and
    identifies whether each sequence contains resistance mutations. This
    function is specific to the 10-1074 bnab. The first array in the list
    must contain the 3 nucleotides coding for hxb2 site 325 while the second 
    array in the list must contain the 9 nucleotides coding for sites 332-334.
    Then, it returns a list of all the unique resistances present in the array.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    removedArr: list of two np.arrays, a list where each array codes for the 
                amino acids at the resistance sites mentioned in the docstring

    Returns:
    --------
    unique_res: list, a list of all the unique resistance mutations present in
                the array
    """
    if not removedArrList[1].shape[-1] == 9:
        print(removedArrList[1].shape, file = sys.stderr)
        raise ValueError('The second array in the input list must have 9 \
                         columns, representing the 3 amino acids at\
                         positions 332-334')
    if not removedArrList[0].shape[-1] == 3:
        print(removedArrList[0].shape, file = sys.stderr)
        raise ValueError('The first array in the input list must have 3\
                         columns, representing the amino acid at position 325')
    
    #Make a list to store the resistance mutations we have observed
    unique_res = []
    
    #Loop through each of the rows in the array
    for i in range(removedArrList[0].shape[0]):

        first_seq = removedArrList[0][i]
        second_seq = removedArrList[1][i]
        curr_muts = identify_resistance_10_1074(first_seq, second_seq)

        unique_res += curr_muts
    
    #Get rid of duplicates
    unique_res = set(unique_res)

    if None in unique_res:
        unique_res.remove(None)

    return unique_res

def get_closest_res(seg_dict, arr_res_pos, freq_sorted = False):
    """ Takes in the dictionary of segregating sites and the indices of the 
    resistance mutations in the array. Returns a list of three tuples where
    the first element is the array position of the segregating site, the 
    second element is the resistance site it's closest to, and the third
    element is the distance between the segregating site and the resistance
    site. Note, this function assumes that there are only two resistance
    sites in the array and that the loci are not in these resistance sites.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_dict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele in position 0 and its frequency in position 1.
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    freq_sorted: bool, True if the output list should be sorted by frequency
                first (with ties broken by distance). False if the output list
                should only be sorted by distance.
    
    Returns:
    --------
    seg_dist_list: list, a list of tuples where the zero element is the array 
                position of the segregating site, the first element is the 
                resistance site it's closest to, and the second element is the
                distance between the segregating site and the resistance site.
                The third and fourth elements are the minor allele and its
                frequency, respectively.
    
    """
    #Make a list to store the closest loci and distances in
    seg_dist_list = []

    #Sort the resistance positions by their start position
    arr_res_pos = list(arr_res_pos)
    arr_res_pos.sort(key = lambda x: x[0])

    #Loop through the segregating sites and find the closest resistance site
    for curr_key in seg_dict.keys():

        #Before the first block
        if curr_key < arr_res_pos[0][0]:
            closest_dist = arr_res_pos[0][0] - curr_key
            closest_site = arr_res_pos[0][0]
        
        #Between the two blocks
        elif curr_key > arr_res_pos[0][1] and curr_key < arr_res_pos[1][0]:
            closest_dist = min(curr_key - arr_res_pos[0][1],
                                 arr_res_pos[1][0] - curr_key)
            closest_site = arr_res_pos[0][1] if closest_dist == \
                            curr_key - arr_res_pos[0][1] else arr_res_pos[1][0]
        
        #After the second block
        else:
            closest_dist = curr_key - arr_res_pos[1][1]
            closest_site = arr_res_pos[1][1]

        
        #Add the closest site and distance to the list
        seg_dist_list.append((curr_key, closest_site, closest_dist,
                               seg_dict[curr_key][0], seg_dict[curr_key][1]))

    #Sort the list by either frequency or distance depending on the input args
    if freq_sorted:
        seg_dist_list.sort(key = lambda x: (x[4], -x[2]), reverse = True)
    else:
        seg_dist_list.sort(key = lambda x: (x[2]))

    return seg_dist_list

def filter_susceptible_seqs(seqArr, seqDF, arr_res_pos):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata. Then, it labels and then removes sequences
    without resistance mutations from the array.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    Returns:
    --------
    seqArr: np.array, the input array with the sequences with susceptible 
                sequences removed.
    """
    #Save the reference sequence for use later
    HXB2_SEQ= seqArr[0, :]

    # Remove any susceptible sequences from the sequence array before we 
    # perform the analysis
    res_region_2 = seqArr[:, arr_res_pos[0][0]:arr_res_pos[0][1] + 1]
    res_region_1 = seqArr[:, arr_res_pos[1][0]:arr_res_pos[1][1] + 1]

    seqDF = label_resistance_10_1074([res_region_1, res_region_2], seqDF)

    new_seqArr = [[HXB2_SEQ]]

    for i in range(1, seqArr.shape[0]):
        curr_seq = seqArr[i, :]
        curr_seq_info = seqDF[seqDF['seq_index'] == i]
        if curr_seq_info['res_muts'].values[0][0] != None:
            new_seqArr.append([curr_seq])
    
    seqArr = np.vstack(new_seqArr)

    return seqArr

###############################################################################
############################## Segregating Sites ##############################
###############################################################################

def calc_bp_dist(pos1, pos2, seqArr, gap_col_set = None):
    """This function takes in two loci in array coordinate and a sequence array
    where each column is a locus and each row is a sequence. Then, it returns
    the number of base pairs between the two loci, disregarding intervening
    loci that are majority gaps.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    pos1: int, the first locus in array coordinates
    pos2: int, the second locus in array coordinates
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    gap_col_set: set, a set containing the array positions of the columns in
                the sequence array that are majority gaps. If this argument is
                None, the function will calculate the set itself. 
    """
    #Make sure the positions are in the right order
    my_pos_list = [pos1, pos2]
    my_pos_list.sort()
    pos1 = my_pos_list[0]
    pos2 = my_pos_list[1]

    #Get the number of columns between the two positions
    bp_dist = pos2 - pos1

    #Now, correct for any intervening positions where more than half of the
    #sequences are gaps
    inter_start = pos1 + 1
    #If there are no intervening sites
    if inter_start == pos2:
        return bp_dist

    #Otherwise, check the intervening sites
    inter_seq = seqArr[:, inter_start:pos2]

    for i in range(inter_seq.shape[1]):
        if gap_col_set:
            if i + inter_start in gap_col_set:
                bp_dist -= 1
        else:
            curr_col = inter_seq[:, i]
            gap_count = np.sum(curr_col == '-')
            if gap_count > inter_seq.shape[0] / 2:
                bp_dist -= 1
    
    return bp_dist

def get_seg_sites(seqArr, arr_res_set, allele_freq_thresh, return_multi = False,
                  include_drm_seg = False, verbose = False):
    """Takes in a sequence array and returns a dictionary where each key is the
    array position of a segregating SNP and the value is a tuple containing the
    minor allele and its frequency. By default, this dictionary only contains
    segregating sites without resistance mutations. NOTE: we are ignoring 
    additional minor alleles beyond the second highest frequency allele.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned.
    arr_res_set: set, a set containing the array positions of the resistance
                mutations.
    allele_freq_thresh: float, the minimum frequency of the second highest
                frequency allele to be considered a segregating site.
    return_multi: bool, True if the output dictionary should include all
                alleles and their frequencies above the threshold (including
                the major allele). False if the output dictionary should only
                include the second highest frequency allele and its frequency.
    include_drm_seg: bool, True if the output dictionary should include 
                resistance mutations. False if the output dictionary should
                exclude them.

    Returns:
    --------
    segDict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele (position 0) and its frequency (position 1).
    """
    seg_freq_dict = {}

    #Now loop through the sequence array and get the snp frequencies at each 
    #position
    for i in range(seqArr.shape[1]):

        
        #If it's a resistance mutation check if it should be included
        if not include_drm_seg and i in arr_res_set:
            continue

        #If it's not a resistance mutation
        #Check if the locus is segregating, disregarding the first sequence
        #which is the reference sequence
        curr_loc = seqArr[1:, i]
        curr_loc = np.unique(curr_loc, return_counts = True)

        if len(curr_loc[0]) > 1:
            nuc_counts = []

            #make sorted lists of the nucleotides and their counts
            for nuc_ind in range(len(curr_loc[0])):
                nuc = curr_loc[0][nuc_ind]
                count = curr_loc[1][nuc_ind]
                nuc_counts.append((nuc, count))

            #Sort from highest to lowest frequency
            nuc_counts.sort(key = lambda x: x[1], reverse = True)
            og_nuc_counts = nuc_counts.copy()


            #For now, don't count sites that are majority gaps
            if nuc_counts[0][0] == '-':
                continue

            

            #Also don't count gaps as segregating alleles
            nuc_counts = [x for x in nuc_counts if x[0] != '-']
            
            if len(nuc_counts) < 2:
                if verbose:
                    print('Only one allele at position', i, file = sys.stderr)
                    print(og_nuc_counts, file = sys.stderr)
                continue

            total_count = sum([x[1] for x in nuc_counts])

            #Output any minor alleles above the frequency thresholds
            if return_multi:
                out_list = []
                for j in range(0, len(nuc_counts)):
                    allele = nuc_counts[j][0]
                    freq = nuc_counts[j][1] / total_count
                    if freq >= allele_freq_thresh:
                        out_list.append((allele, freq))
                if len(out_list) > 1:
                    seg_freq_dict[i] = out_list

            #Otherwise, only output the second highest frequency allele
            #if it's above the threshold
            else:
                second_freq = nuc_counts[1][1] / total_count


                #If the second highest frequency allele is above the threshold
                #add it to the dictionary as a segregating site
                if second_freq >= allele_freq_thresh:
                    seg_freq_dict[i] = (nuc_counts[1][0], second_freq)
    
    return seg_freq_dict

def seg_freq_by_timepoint(seg_freq_dict, seqArr, seq_info_df, return_co_res = False):
    """Takes in a dictionary of segregating sites, the array of sequences,
    and the dataframe with the sequences' metadata. Then, it returns one
    dataframe with the frequency of each segregating site at each timepoint and
    a second dataframe with the allele's average frequency across all 
    timepoints.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_freq_dict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele in position 0 and its frequency in position 1.
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seq_info_df: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    return_co_res: bool, True if the output dataframe should include the resistance
                mutations that are present in the same sequences as the segregating
                alleles.
    Returns:
    --------
    allele_freq_df: pd.DataFrame, a dataframe with the frequency of each allele
                at each timepoint.
    all_time_freqs: pd.DataFrame, a dataframe with the average frequency of each
                allele across all timepoints.
    time_sample_sizes: dict, a dictionary where each key is a timepoint and the
                value is the number of sequences sampled at that timepoint.
    """
    #Separate out sequences by timepoint and label their new indices in the 
    #dataframe
    time_seq_dict = {}
    time_info_dict = {}

    #Loop through the timepoints and add the timepoint sequence arrays to a
    #dictionary
    for curr_timepoint in seq_info_df['time_label'].values:
        #Get the sequences for the current timepoint
        curr_timepoint_df = seq_info_df[seq_info_df['time_label'] == \
                                curr_timepoint].copy()

        #Sort the sequences by their index
        curr_timepoint_df = curr_timepoint_df.sort_values('seq_index')
        curr_timepoint_df = curr_timepoint_df.reset_index(drop=True)

        #add the sequences 
        curr_time_arr = seqArr[curr_timepoint_df['seq_index'].to_numpy(), :]
        curr_timepoint_df['time_seq_index'] = np.arange(curr_time_arr.shape[0])
        time_seq_dict[curr_timepoint] = curr_time_arr
        time_info_dict[curr_timepoint] = curr_timepoint_df

    allele_freq_df = []
    #The number of sequences sampled at each timepoint
    time_sample_sizes = {}
    all_time_freqs = []

    #Loop through the segregating sites and count the number of alleles that 
    #persist across timepoints. 
    for curr_entry in seg_freq_dict:
        alleles = seg_freq_dict[curr_entry]

        #Get the alleles at the given position across timepoints
        time_slices = {}
        for curr_timepoint in time_seq_dict:
            curr_time_arr = time_seq_dict[curr_timepoint]
            time_sample_sizes[curr_timepoint] = curr_time_arr.shape[0]
            time_slices[curr_timepoint] = curr_time_arr[:, curr_entry]

        #Check what timepoints the alleles are present in and also at what
        #frequencies
        for curr_allele in alleles:
            filter_out = False

            allele_freq_all = np.sum(seqArr[:, curr_entry] == curr_allele[0])
            allele_freq_all /= seqArr.shape[0]

            for curr_timepoint in np.sort(seq_info_df['time_label'].unique()):
                curr_timepoint_df = time_info_dict[curr_timepoint]
                curr_time_arr = time_slices[curr_timepoint]

                #Get the sequences that contain the allele
                if curr_allele[0] in curr_time_arr:
                    #calculate the frequencies
                    curr_allele_freq = np.sum(curr_time_arr == curr_allele[0])
                    curr_allele_freq /= curr_time_arr.shape[0]

                    #mark any resistance mutations they are seen with
                    curr_allele_indices = np.argwhere(curr_time_arr ==\
                                                         curr_allele[0])
                    curr_allele_indices = curr_allele_indices.flatten()

                    #now get the resistance mutations and count only the unique
                    #ones
                    curr_allele_info = curr_timepoint_df[curr_timepoint_df[
                                    'time_seq_index'].isin(curr_allele_indices)]
                    if return_co_res:
                        co_res_muts = curr_allele_info['res_muts'].values
                        co_res_muts = list(set(np.concatenate(co_res_muts)))
                        
                        allele_freq_df.append([curr_entry, curr_allele[0],
                                                curr_timepoint, curr_allele_freq,
                                                co_res_muts])
                    else:
                        allele_freq_df.append([curr_entry, curr_allele[0],
                                                curr_timepoint, curr_allele_freq])
                else:
                    if return_co_res:
                        allele_freq_df.append([curr_entry, curr_allele[0],
                                             curr_timepoint, 0, []])
                    else:
                        allele_freq_df.append([curr_entry, curr_allele[0],
                                             curr_timepoint, 0])
                    #Save the frequency accross all timepoints for calculating
                    #our nulls    
            if not filter_out:
                all_time_freqs.append([curr_entry, curr_allele[0],
                                        allele_freq_all])

    if return_co_res:
        allele_freq_df = pd.DataFrame(allele_freq_df, 
                                    columns = ['position', 'allele', 'time',
                                                'freqs','res_muts'])
    else:
        allele_freq_df = pd.DataFrame(allele_freq_df, 
                                    columns = ['position', 'allele', 'time',
                                                'freqs'])
    all_time_freqs = pd.DataFrame(all_time_freqs, 
                                  columns = ['position', 'allele', 'freqs'])
    
    return allele_freq_df, all_time_freqs, time_sample_sizes

def filter_seg_by_time(seg_freq_dict, seqArr, seq_info_df, time_seg_filter):
    """Takes in a dictionary of segregating sites (with no allele frequency
    filtering applied so far), the array of sequences, and the dataframe with
    the sequences' metadata. Then, it filters the segregating site dictionary
    to include only sequences that are above the time_seg_filter threshold at
    at least one timepoint.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_freq_dict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele in position 0 and its frequency in position 1. (or
                if multiseg is true, it contains a list of tuples where the
                zero element is the allele and the first element is the
                frequency)
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seq_info_df: pd.DataFrame, a dataframe with all of the info about the
                sequences

    Returns:
    --------
    filtered_seg_dict: dict, the input segregating site dictionary filtered to
                include only sites that are above the time_seg_filter threshold
                at at least one timepoint.
    """
    #First we'll get the frequencies of segregating sites across timepoints
    allele_freq_df, all_time_freqs, time_sample_sizes = seg_freq_by_timepoint(
        seg_freq_dict, seqArr, seq_info_df)
    
    #Now we'll filter the segregating sites to include only those that are
    #above the time_seg_filter threshold at at least one timepoint
    filtered_seg_dict = {}
    for curr_site in seg_freq_dict:
        curr_site_freqs = allele_freq_df[allele_freq_df['position'] == curr_site]
        curr_site_freqs = curr_site_freqs[curr_site_freqs['time'] != 'HXB2']

        curr_site_freqs = curr_site_freqs[curr_site_freqs['freqs'] >= \
                                        time_seg_filter]
        unique_alleles = curr_site_freqs['allele'].unique()

        if len(unique_alleles) > 1:
            filtered_seg_dict[curr_site] = seg_freq_dict[curr_site]
        
    return filtered_seg_dict

def make_seg_sets(seg_freq_dict, seqArr, seq_info_df, arr_res_window_min,
                  arr_res_window_max, arr_res_set):
    """This function takes in a dictionary of segregating sites, the sequence
    array, and the dataframe with the sequences' metadata. It returns a 
    dictionary where each key is the name of a sequence and each value is a 
    set containing all the minor alleles in that sequence in the tuple format
    (position, allele). Only alleles at sites between arr_res_window_min and
    arr_res_window_max (but not in arr_res_set) are included in the sets.

    NOTE: the mapping between the sequence dataframe and
    the sequence array must be intact for this function to work (if you have
    removed hxb2 from the top of the sequence array the seq_info_df indices
    must be adjusted by one to compensate).
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_freq_dict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele in position 0 and its frequency in position 1.
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. 
    seq_info_df: pd.DataFrame, a dataframe with all of the sequence metadata
    arr_res_window_min: int, the minimum position of the resistance window in
                array coordinates. Only sites in the window will be included
                in the sets.
    arr_res_window_max: int, the maximum position of the resistance window in
                array coordinates. Only sites in the window will be included
                in the sets.
    arr_res_set: set, a set containing the array positions of the resistance
                mutations.

    Returns:
    --------
    seg_set_dict, a dictionary where each key is the name of a sequence
                and each value is a set containing the minor alleles in that
                sequence in tuple format (position, allele).
    """
    seq_info_df = seq_info_df[seq_info_df['time_label'] != 'HXB2']

    #Make a dictionary to return
    seg_set_dict = {}

    #Make a dictionary mapping sequence names to their array indices
    index_to_name = dict(zip(seq_info_df['seq_index'], seq_info_df['orig_name']))
    
    #Populate the dictionary with the sequence names
    for curr_name in seq_info_df['orig_name'].values:
        seg_set_dict[curr_name] = set()

    #Loop through the segregating sites and add the minor alleles to the
    #appropriate set
    for curr_site in seg_freq_dict:
        if curr_site < arr_res_window_min or curr_site > arr_res_window_max:
            continue
        if curr_site in arr_res_set:
            continue
        curr_column = seqArr[:, curr_site]

        #Get the minor alleles at the site
        minor_alleles = seg_freq_dict[curr_site]
        minor_alleles = [x[0] for x in minor_alleles[1:]]
        
        #get indices that contain minor alleles
        minor_inds = np.argwhere(np.isin(curr_column, minor_alleles)).flatten()
        if 0 in minor_inds:
            minor_inds = [x for x in minor_inds if x != 0]
        
        #Add the minor alleles to the sets of all the sequences containing them
        for curr_ind in minor_inds:
            allele = curr_column[curr_ind]
            if allele != '-':
                name = index_to_name[curr_ind]
                seg_set_dict[name].add((curr_site, allele))

    
    return seg_set_dict

###############################################################################
############################# General Utilities ###############################
###############################################################################
def arrange_arr_by_time(seqArr, seqDF, time_order, cluster = False):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata. Lastly, it takes in a list of timepoints in
    the order that they should be arranged in the output array. Then, it
    returns the array and dataframe with the sequences in the array
    arranged in the order specified by time_order and the dataframe index
    column updated. If cluster is true, each timepoint group will be sorted
    by cluster assignment.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    time_order: list, a list of the timepoints in the order that they should
                be arranged in the output array.
    Returns:
    --------
    new_seqArr: np.array, the input array with the sequences arranged in the
                order specified by time_order.
    seqDF: pd.DataFrame, the input dataframe with the sequences arranged in the
                order specified by time_order.
    time_divides: list, a list of the indices in the array where the timepoints
                change.
    """
    time_divides = []
    new_seqArr = []
    for curr_time in time_order:
        curr_time_df = seqDF[seqDF['time_label'] == curr_time]
        if cluster:
            curr_time_df = curr_time_df.sort_values('cluster_label')
        for index, row in curr_time_df.iterrows():
            curr_ind = row['seq_index']
            new_seqArr.append(seqArr[curr_ind])
            new_index = len(new_seqArr) - 1
            seqDF.at[index, 'seq_index'] = new_index
        time_divides += [len(new_seqArr)]
    
    new_seqArr = np.stack(new_seqArr)
    return new_seqArr, seqDF, time_divides

def remove_gap_seqs(seqArr, seqDF, percent_gap):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata. Then, it removes sequences that have above a
    given percentage of gaps from both the array and the sequence dataframe
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    percent_gap: float, the percentage of gaps above which sequences will be
                filtered out of the analysis
    Return:
    -------
    outArr: np.array, the input array with the gap sequences removed
    outDF: pd.DataFrame, the input dataframe with the gap sequences removed

    """
    outArr = []
    outDF = []
    new_index = 0

    #Loop through the rows of the sequence array and check each
    for i in range(seqArr.shape[0]):
        curr_seq = seqArr[i]

        #Count the number of gaps in the sequence
        gap_count = np.count_nonzero(curr_seq == '-')

        #If the percentage of gaps is below the threshold, add the sequence
        #to the output array
        if gap_count/len(curr_seq) < percent_gap:
            outArr.append(curr_seq)

            #We need to get the corresponding array index and also update the
            #sequence dataframe
            curr_df_entry = seqDF[seqDF['seq_index'] == i].copy()
            curr_df_entry['seq_index'] = new_index
            outDF.append(curr_df_entry)
            new_index += 1
    
    outArr = np.array(outArr)
    outDF = pd.concat(outDF, ignore_index=True)

    return outArr, outDF

def generate_consensus_seq(seqArr):
    """ This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It returns an array of shape 
    (1, seqArr.shape[1]) where each column is the majority nucleotide at that
    position.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                the nucleotide at that position. The sequences are aligned and
                the first row is the reference sequence (HXB2).

    Returns:
    --------
    consensus_arr: np.array, a one dimensional array where each column is the
                majority nucleotide at that position.    
    """
    consensus_seq = []

    #loop through the columns to get consensus nucleotide at each site
    for i in range(seqArr.shape[1]):
        #get the column
        curr_col = seqArr[:,i]

        #find the majority nucleotide
        majority_nuc = pd.Series(curr_col)
        majority_nuc = majority_nuc.mode()[0]

        consensus_seq.append(majority_nuc)
    
    consensus_arr = np.array(consensus_seq)
    return consensus_arr

def filter_timepoints(seqArr, seqDF, time_set):
    """ This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata. Then, it removes sequences that are not from
    the timepoints specified in the time_list from both the array and the
    sequence dataframe
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    time_set: set, a set of the timepoints to keep in the analysis. 

    Returns:
    --------
    outArr: np.array, the input array with the sequences from the timepoints
                not in the time_list removed
    outDF: pd.DataFrame, the input dataframe with the sequences from the
                timepoints not in the time_list removed
    """
    outArr = []
    outDF = []

    #Get only sequences in the timepoints of interest
    time_set.add('HXB2')
    time_filtered_DF = seqDF[seqDF['time_label'].isin(time_set)]


    #Add the sequences from the filtered array into the output
    for index, row in time_filtered_DF.iterrows():
        curr_ind = row['seq_index']
        row_copy = row.copy()
        row_copy['seq_index'] = len(outArr)

        outArr.append(seqArr[curr_ind])
        outDF.append(row_copy)

    outArr = np.stack(outArr)
    outDF = pd.DataFrame(outDF)


    return outArr, outDF


def label_identical_seqs(seqArr, seqDF, ignore_time = False):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata. Next, it finds groups of identical sequences
    from the same sampling timepoint. It collapses the rows of the info 
    dataframe that correspond to identical sequences and then returns the
    dataframe with the collapsed rows.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    ignore_time: bool, True if the function should ignore the timepoints and
                collapse all identical sequences. False if the function should
                only collapse sequences that are identical and from the same
                timepoint.

    Returns:
    --------
    collapsedSeqDF: pd.DataFrame, the input dataframe with the rows 
                corresponding to identical sequences collapsed. In the new 
                dataframe, the 'seq_index' and 'orig_name' columns are now 
                lists with the info for all of the identical sequences.
    """
    if ignore_time:
        #make a dataframe which will hold the most recent identical sequences
        latest_collapsed_info_df = []
        collapsed_info_df = collapse_helper(seqArr, seqDF)

        #label each sequence with the day it was collected to make comparisons
        #easier
        seqDF = convert_times_to_days(seqDF)
        all_identical_seqs = []

        #For each sequence, choose a representative from the latest timepoint
        for index, row in collapsed_info_df.iterrows():
            other_seq_inds = row['identical_seqs']
            all_seq_inds = [row['seq_index']] + list(other_seq_inds)

            #put all of the sequences in the same dataframe and then get the
            #latest timepoint from there
            all_seq_info = seqDF[seqDF['seq_index'].isin(all_seq_inds)]
            all_seq_info = all_seq_info[all_seq_info.study_day == all_seq_info.study_day.max()]
            all_seq_info = all_seq_info.reset_index(drop = True)
            all_identical_seqs.append(tuple(set(all_seq_inds) - set(all_seq_info['seq_index'])))
            latest_collapsed_info_df.append(all_seq_info.iloc[0])
        
        latest_collapsed_info_df = pd.concat(latest_collapsed_info_df, axis = 1,
                                              ignore_index=True).T
        latest_collapsed_info_df['identical_seqs'] = all_identical_seqs

        return latest_collapsed_info_df


    else:
        all_collapsed_info = []
        #Loop through the time points since we will only count sequences as
        #identical if they are from the same time point
        for curr_time in seqDF['time_label'].unique():
            curr_time_info = seqDF[seqDF['time_label'] == curr_time].copy()
            #We need to reset the index because we are using it to loop
            curr_time_info = curr_time_info.reset_index(drop = True)

            collapsed_info_df = collapse_helper(seqArr, curr_time_info)
            curr_time_info = collapsed_info_df
            all_collapsed_info.append(curr_time_info)
            
        all_collapsed_info = pd.concat(all_collapsed_info, ignore_index=True)
        return all_collapsed_info


def convert_times_to_days(seqDF, rebound = False):
    """This function takes in a dataframe with the sequences' metadata and 
    converts the timepoint labels to days since study start. It returns the
    array with a 'study day' column added. The timepoints should be of the 
    format 'D0' or 'W1' where the number is the number of days or weeks since
    the study start.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array.
    rebound: bool, True if the rebound column also needs to be converted
    """
    my_timepoints = seqDF['time_label'].unique()
    mySeqDF = seqDF.copy()

    #Make a dictionary to convert the timepoints to days
    time_dict = {}
    for curr_time in my_timepoints:
        if curr_time[0] == 'D':
            time_dict[curr_time] = int(curr_time[1:])
        else:
            time_dict[curr_time] = int(curr_time[1:]) * 7


    #Now map the timepoints to days
    mySeqDF['study_day'] = mySeqDF['time_label'].map(time_dict)


    return mySeqDF

def array_to_fasta(seqArr, seqDF, out_file):
    """This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It also takes in a dataframe
    with the sequences' metadata (including the orig_name column which will
    provide sequence names for the new alignment). Then, it writes the 
    sequences to a fasta file with path specified by out_file.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned.
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
                and has a column that gives the index of the sequence in the
                array. It also has a column 'orig_name' that gives the name of
                the sequence.
    out_file: str, the path to the output fasta file.

    Returns:
    --------
    None    Writes a fasta file to the path specified by out_file.
    """

    out_text = ''
    #Loop through each of the sequences and append them to the string
    for index, row in seqDF.iterrows():
        curr_seq = seqArr[row['seq_index']]
        curr_name = row['orig_name']
        out_text += '>' + curr_name + '\n'
        out_text += ''.join(curr_seq) + '\n'
    
    #Write the string to the output file
    with open(out_file, 'w') as out_handle:
        out_handle.write(out_text)
    
    return

def get_dist_from_region(seg_freq_dict, region_tup, freq_sorted = False):
    """ Takes in the dictionary of segregating sites and the indices of a region
    of interest in the array (start, end). Returns a list of three tuples where
    the first element is the array position of the segregating site, the 
    second element is the region site it's closest to, and the third
    element is the distance between the segregating site and the region.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_dict: dict, a dictionary where each key is the array position of a 
                segregating SNP and the value is a tuple containing the minor
                allele in position 0 and its frequency in position 1.
    arr_res_pos: tuple, a tuple where each entry is itself a tuple containing
                the start and end positions of the resistance mutation in array
                coordinates ((start, end) ,(start2, end2)).
    freq_sorted: bool, True if the output list should be sorted by frequency
                first (with ties broken by distance). False if the output list
                should only be sorted by distance.
    
    Returns:
    --------
    seg_dist_list: list, a list of tuples where the zero element is the array 
                position of the segregating site, the first element is the 
                region site it's closest to, and the second element is the
                distance between the segregating site and the region.
                The third and fourth elements are the minor allele and its
                frequency, respectively.
    
    """
    seg_dist_list = []
    
    #Loop through the segregating sites and get the distance to the region
    for curr_seg_site in seg_freq_dict.keys():
        #If it's in the region the distance is 0
        if curr_seg_site >= region_tup[0] and curr_seg_site <= region_tup[1]:
            curr_distance = 0
            closest_site = curr_seg_site
        #Otherwise, calculate the distance
        else:
            if curr_seg_site < region_tup[0]:
                curr_distance = region_tup[0] - curr_seg_site
                closest_site = region_tup[0]
            else:
                curr_distance = curr_seg_site - region_tup[1]
                closest_site = region_tup[1]
        seg_dist_list.append((curr_seg_site, closest_site, curr_distance))

    #Sort the list by either frequency or distance depending on the input args
    if freq_sorted:
        seg_dist_list.sort(key = lambda x: (x[4], -x[2]), reverse = True)
    else:
        seg_dist_list.sort(key = lambda x: (x[2]))

    return seg_dist_list


###############################################################################
############################# Helper Functions ################################
###############################################################################

def make_gap_col_set(seqArr, gap_thresh = 0.5):
    """ This function takes in an array of aligned sequences where every row is
    a sequence and every column is a position. It returns a set of columns with
    a higher proportion of gaps than gap_thresh. The default is that columns
    with majority gaps are included in the set.
    """
    gap_col_set = set()
    for i in range(seqArr.shape[1]):
        curr_col = seqArr[:,i]
        gap_count = np.sum(curr_col == '-')
        if gap_count > (seqArr.shape[0] * gap_thresh):
            gap_col_set.add(i)
    return gap_col_set

def one_hot_encode(seq):
    """ This function takes a DNA sequence and returns a one hot encoded array 
    it is adapted from 
    https://numpy.org/devdocs/reference/generated/numpy.eye.html"""
    mapping = dict(zip("ACGT-", range(5)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(5)[seq2]

def one_hot_encode_AA(seq):
    """ This function takes an amino acid sequence and returns a one hot encoded
    array"""
    mapping = dict(zip("ACDEFGHIKLMNPQRSTVWY-X", range(22)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(22)[seq2]

def hxb2_mapping_dict(seqArr, start_pos):
    """This function takes in a sequence array where the first sequence is hxb2
    and returns two dictionaries that provide both directions of the mapping.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    start_pos: int, the starting position for the array in the HXB2 reference

    Returns:
    --------
    hxb2_to_arr: dict, a dictionary where the key is the hxb2 position and
                        the value is the corresponding index in the sequence
                        array
    arr_to_hxb2: dict, a dictionary where the key is the index in the sequence
                        array and the value is the corresponding hxb2 position
    """
    hxb2_to_arr = {}
    arr_to_hxb2 = {}
    index_start = -1
    myhxb2 = seqArr[0]

    #Now we will generate the indices to hxb2
    for i in range(len(myhxb2)):
        curr_AA = myhxb2[i]

        #Index by one unless there is a gap
        if curr_AA != '-':
            index_start = math.floor(index_start+1)
            
            #Update the dictionaries
            hxb2_to_arr[math.floor(start_pos) + index_start] = i
            arr_to_hxb2[i] = index_start + math.floor(start_pos)

        #We can accommodate 100 gaps before the indexing is thrown off
        else:
            index_start += 0.001
        
            #Update the dictionaries
            hxb2_to_arr[start_pos + index_start] = i
            arr_to_hxb2[i] = index_start + start_pos
    
    return hxb2_to_arr, arr_to_hxb2

def translate_seq(seq, error = True):
    """ This function takes in a sequence (in the style of a 1D numpy array
    where each element is a base) and returns the translated amino acid
    sequence as a string
    """
    #First convert the array to a string
    seq_str = ''.join(seq)

    #If we want to avoid throwing errors return none
    if not error:
        if '-' in seq_str:
            return ' '
        
    #Now translate the string
    seq_aa = Seq(seq_str)

    seq_aa = seq_aa.translate()

    return seq_aa

def identify_resistance_10_1074(first_region, second_region):
    """This function takes in two 1D sequence arrays where each element is a 
    single nucleotide. The first_region array is the three nucleotides coding
    for the AA at position 325. The second_region array is the nine nucleotides
    coding for the AAs at positions 332-334. This function returns a list of 
    the resistance mutations present in the sequence.

    If there are no resistance mutations, the function returns a list with a
    single element, None.
    """
    #Make a place to store the resistance mutations
    curr_muts = []

    #Translate the sequence
    first_seq = translate_seq(first_region)
    second_seq = translate_seq(second_region)


    # print(first_region)
    # print(first_seq)
    # print(second_region)
    # print(second_seq)

    #Check if the sequence contains resistance mutations
    if first_seq[0] != 'D' and first_seq[0] != 'N':
        curr_muts.append('D/N325' + first_seq[0])

    if second_seq[0] != 'N':
        curr_muts.append('N332' + second_seq[0])
        
    if second_seq[2] != 'S' and second_seq[2] != 'T':
        curr_muts.append('S334' + second_seq[2])
    
    if len(curr_muts) == 0:
        curr_muts.append(None)

    return curr_muts

def collapse_helper(seqArr, seq_info_df):
    """This function is a helper function for the label identical sequences
    function. It is used to collapse the rows of the dataframe that correspond
    to identical sequences.
    """
    collapsed_info_df = []
    all_identical_seqs = []
    to_drop = set()
    for index, row in seq_info_df.iterrows():
        identical_seqs = []
        
        if row['seq_index'] in to_drop:
            continue
        
        for i in range(index + 1, len(seq_info_df)):
            seq1 = seqArr[row['seq_index'], :]
            seq2 = seqArr[seq_info_df.iloc[i]['seq_index'], :]

            if np.array_equal(seq1, seq2):
                if seq_info_df.iloc[i]['res_muts'] == row['res_muts']:
                    to_drop.add(seq_info_df.iloc[i]['seq_index'])
                    identical_seqs.append(seq_info_df.iloc[i]['seq_index'])

        collapsed_info_df.append(row)
        all_identical_seqs.append(tuple(identical_seqs))
    
    
    collapsed_info_df = pd.concat(collapsed_info_df, ignore_index=True, axis = 1).T
    collapsed_info_df['identical_seqs'] = all_identical_seqs
    return collapsed_info_df