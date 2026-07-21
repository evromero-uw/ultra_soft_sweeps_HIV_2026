# Documentation for Pardata class

## Attributes

### Basic Metadata
 - **dataset_name**: string, the name of the dataset (for now caskey2017 or clyde2023).
 - **dirPath**: string, the path to the directory containing the sequence alignments that will be loaded.
 - **participant**: string, the code for the participant the data is associated with.
 - **arr_res_positions**: tuple, a tuple where each entry is a tuple containing the start and end positions of a resistance site in array coordinates (for indexing into the seq_arr). E.g.
 ((start_1, end_1), (start_2, end_2)).
 - **arr_res_set**: set, a set containing the positions of resistance mutations in array coordinates. For example, if the resistance region spans 994-996, this set should contain {994, 995, 996}.

### Data Structures
 - **seq_arr**: np.array, the array holding all of the sampled sequences for the participant. Each row is a sequence and each column is a position in that sequence. The sequences are aligned and the first row is the reference sequence (HXB2).
 - **seq_info_df**: pd.DataFrame, a dataframe containing the metadata for the sequences. This dataframe includes 
    - **orig_name**: str, sequence name
    - **time_label**: str, a label indicating the sampling timepoint
    - **participant**: str, the individual the sequence is associated with
    - **seq_index**: int, the row index of the associated sequence in seq_arr
    - **res_muts**: list, the resistance mutations (if any) that the sequence carries
 - **seg_freq_dict**: dictionary, a dictionary where each key is the array position of a segregating SNP and the value is a tuple containing the minor allele (position 0) and its frequency (position 1). This dictionary only contains sites where the minor allele frequency is >allele_freq_thresh.
 - **hxb2_nuc_coords_hta**: dictionary, a dictionary where each key is a coordinate in hxb2 indexing and each value is its corresponding index in seq_arr.
 - **hxb2_nuc_coords_ath**: dictionary, a dictionary where each key is an array coordinate specifying a row in the seq_arr. Each value is the corresponding coordinate in relation to the hxb2 reference.


### Filtering Information
 - **allele_freq_thresh**: float, the lower threshold for minor allele frequency at site. Sites are only considered segregating if the second most common allele reaches this frequency.
 - **gap_thresh**: float, the maximum fraction of gaps allowed per sequence (sequences with larger percentages of gaps are filtered out).
 - **time_filter**: list, a list of timepoint labels for timepoints that have been filtered out from both the object's sequence array and sequence dataframe.
 - **susceptible_filter**: boolean, True if susceptible sequences have been filtered out of the object's sequence array. False otherwise.
 - **multi_seg**: boolean, True if the segregating sites dictionary contains all of the minor variants above allele_freq_thresh. False if only the second highest frequency allele is included.
 - **include_drm_seg**: boolean, True if the segregating sites dictionary includes data from drug resistance sites that are segregating. False, if only non DRM sites are included.

## Methods