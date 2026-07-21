import os
import sys
sys.path.append('../../bin/')
import data_util
import numpy as np

#Here I am defining a class which will hold all of the data for a single
#participant. This will make it easier to pass the data around between functions
class Pardata:
    def __init__(self, dirPath, dataset_name, participant):
        #The name of the dataset (caskey or clyde)
        self.dataset_name = dataset_name
        #The path to the directory containing the data
        self.dirPath = dirPath
        #The code for the participant
        self.participant = participant
        #The threshold for the number of gaps allowed in a sequence
        self.gap_thresh = None
        #A list of time points filtered out if there are any
        self.time_filter = None
        #A frequency threshold for sites to be including in the segregating 
        #site dictionary
        self.allele_freq_thresh = None
        #A boolean indicating whether susceptible sequences have been filtered
        #out
        self.susceptible_filter = False
        #A boolean indicating whether the seg_freq_dict includes multiple minor
        #variants or only the second most frequent variant
        self.multi_seg = False
        #A boolean indicating whether the seg_freq_dict includes data from
        #DRM sites that are segregating
        self.include_drm_seg = False
        #A boolean indicating whether the seg_freq_dict includes sites that are
        #above the frequency threshold at any timepoint (rather than the standard
        #condition which is surpassing the threshold in the aggregate data)
        self.time_seg_filter = None
        #The sequence array where each row is a sequence and each column is a
        #nucleotide position
        self.seq_arr = None
        #A dataframe containing information about each sequence
        self.seq_info_df = None
        #The positions of the resistance mutations in array coordinates
        self.arr_res_positions = None
        #A set containing the positions of the array resistance mutations
        self.arr_res_set = None
        #A dictionary containing the segregating sites and their frequencies
        self.seg_freq_dict = None
        #Dictionaries mapping between HXB2 and array coordinates
        self.hxb2_nuc_coords_hta = None
        self.hxb2_nuc_coords_ath = None
    
    def load_data_3BNC117(self, allele_freq_thresh, multi_seg):
        """This function populates the objects' attributes with sequence data.
        -----------------------------------------------------------------------
        """
        #Check which data set to load since each has different naming
        #conventions
        if self.dataset_name == 'caskey2017':
            caskey2017_bool = True
            clyde2024_bool = False
        elif self.dataset_name == 'clyde2024':
            clyde2024_bool = True
            caskey2017_bool = False
        else: 
            caskey2017_bool = False
            clyde2024_bool = False

        #Load the data sets
        seq_arr, seq_info_df = data_util.fasta_to_dataStructs(self.dirPath,
                                                               caskey2017_bool,
                                                               clyde2024_bool)
        
        #Make the mapping dictionaries
        hxb2_nuc_coords_hta, hxb2_nuc_coords_ath = \
                                        data_util.hxb2_mapping_dict(seq_arr, 1)

        #Get the segregating sites
        seg_freq_dict = data_util.get_seg_sites(seq_arr, {}, 
                                    allele_freq_thresh= allele_freq_thresh,
                                    return_multi= multi_seg)
        
        
        #Now assign the attributes
        self.allele_freq_thresh = allele_freq_thresh
        self.multi_seg = multi_seg
        self.seq_arr = seq_arr
        self.seq_info_df = seq_info_df
        self.seg_freq_dict = seg_freq_dict
        self.hxb2_nuc_coords_hta = hxb2_nuc_coords_hta
        self.hxb2_nuc_coords_ath = hxb2_nuc_coords_ath

    def load_data_10_1074(self, hxb2_res_positions, allele_freq_thresh, 
                            multi_seg, drm_seg = False):
        """This function populates the objects' attributes with sequence data.
        -----------------------------------------------------------------------
        """
        #Check which data set to load since each has different naming
        #conventions
        if self.dataset_name == 'caskey2017':
            caskey2017_bool = True
            clyde2024_bool = False
        elif self.dataset_name == 'clyde2024':
            clyde2024_bool = True
            caskey2017_bool = False
        else: 
            caskey2017_bool = False
            clyde2024_bool = False

        #Load the data sets
        seq_arr, seq_info_df = data_util.fasta_to_dataStructs(self.dirPath,
                                                               caskey2017_bool,
                                                               clyde2024_bool)
        
        #Get the resistance positions
        hxb2_nuc_coords_hta, hxb2_nuc_coords_ath = \
                                        data_util.hxb2_mapping_dict(seq_arr, 1)
        arr_res_positions = ((hxb2_nuc_coords_hta[hxb2_res_positions[0][0]],\
                            hxb2_nuc_coords_hta[hxb2_res_positions[0][1]]),
                            (hxb2_nuc_coords_hta[hxb2_res_positions[1][0]],\
                            hxb2_nuc_coords_hta[hxb2_res_positions[1][1]]))
        
        #Now make a set spanning the range of array resistance positions
        arr_res_set = set(range(arr_res_positions[0][0], arr_res_positions[0][1] + 1))\
                        .union(set(range(arr_res_positions[1][0],\
                                        arr_res_positions[1][1] + 1)))
        
        #Label all of the sequences with their resistance mutations (or none)
        res_region_2 = seq_arr[:, arr_res_positions[0][0]:arr_res_positions[0][1] + 1]
        res_region_1 = seq_arr[:, arr_res_positions[1][0]:arr_res_positions[1][1] + 1]

        
        seq_info_df = data_util.label_resistance_10_1074(
                                            [res_region_1, res_region_2],
                                            seq_info_df)
        
        seg_freq_dict = data_util.get_seg_sites(seq_arr, arr_res_set, 
                                    allele_freq_thresh= allele_freq_thresh,
                                    return_multi= multi_seg,
                                    include_drm_seg = drm_seg)
        
        
        #Now assign the attributes
        self.allele_freq_thresh = allele_freq_thresh
        self.multi_seg = multi_seg
        self.include_drm_seg = drm_seg
        self.seq_arr = seq_arr
        self.seq_info_df = seq_info_df
        self.arr_res_positions = arr_res_positions
        self.arr_res_set = arr_res_set
        self.seg_freq_dict = seg_freq_dict
        self.hxb2_nuc_coords_hta = hxb2_nuc_coords_hta
        self.hxb2_nuc_coords_ath = hxb2_nuc_coords_ath
        
    
    def time_filter_data(self, out_timepoints):
        """This function takes a timepoint list and filters out the given time
        points from the data.
        -----------------------------------------------------------------------
        """
        time_set = set(self.seq_info_df['time_label'].unique())
        for timepoint in out_timepoints:
            time_set.remove(timepoint)

        seq_arr, seq_info_df = data_util.filter_timepoints(self.seq_arr,
                                                           self.seq_info_df,
                                                            time_set)
        
        #Now assign the attributes
        self.seq_arr = seq_arr
        self.seq_info_df = seq_info_df
        self.time_filter = out_timepoints

    def filter_gaps(self, gap_thresh):
        """This function filters out sequences with more than a given
        percentage of gaps.
        -----------------------------------------------------------------------
        """
        seq_arr, seq_info_df = data_util.remove_gap_seqs(self.seq_arr, 
                                                    self.seq_info_df, 
                                                    gap_thresh)
        
        #Now assign the attributes
        self.seq_arr = seq_arr
        self.seq_info_df = seq_info_df
        self.gap_thresh = gap_thresh

    def filter_susceptible(self):
        """This function filters out susceptible sequences.
        -----------------------------------------------------------------------
        """
        seq_arr = data_util.filter_susceptible_seqs(self.seq_arr, 
                                                        self.seq_info_df,
                                                        self.arr_res_positions)
        
        #Now assign the attributes
        self.seq_arr = seq_arr
        self.susceptible_filter = True
    
    def filter_seg_time(self, time_seg_filter):
        """This function filters the seg_freq_dict to only include sites that
        are above the frequency threshold at any timepoint (rather than the 
        standard condition which is surpassing the threshold in the aggregate 
        data). If there was a previous allele frequency threshold associated
        with this object, this function will overwrite it.
        -----------------------------------------------------------------------
        Params:
        time_seg_filter: float, the frequency threshold for a site to be
            included in the seg_freq_dict.
        """
        #First check if we need to undo the segregating filtering by remaking
        #the seg_freq_dict
        if self.allele_freq_thresh and self.allele_freq_thresh != 0:
            self.seg_freq_dict = data_util.get_seg_sites(self.seq_arr,
                                    self.arr_res_set,
                                    allele_freq_thresh= self.allele_freq_thresh,
                                    return_multi= self.multi_seg,
                                    include_drm_seg = self.include_drm_seg)

        self.seg_freq_dict = data_util.filter_seg_by_time(self.seg_freq_dict,
                                                          self.seq_arr,
                                                         self.seq_info_df,
                                                         time_seg_filter)
        self.time_seg_filter = time_seg_filter



    






