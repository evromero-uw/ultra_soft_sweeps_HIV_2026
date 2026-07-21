import os
import sys
#for running on cluster
sys.path.append('/net/feder/vol1/home/evromero/2025_hiv_linkage/bin')
sys.path.append('/Volumes/feder-vol1/home/evromero/2025_hiv_linkage/bin')
import numpy as np
import pandas as pd
import slim_out_parse as sop
import zaniniUtil as zu
from snakemake.script import snakemake

#The cutoffs for processing the segregating loci
SEG_CUTOFF = 0.01

# This script is going to take in the output from a slim run. It will then make
# a segregating loci dataframe and a haplotype dataframe. These dataframes
# can be used as input for the recombination estimation snakemake pipeline.
outdataDir = snakemake.params.out_sim_dir
if not os.path.exists(outdataDir + '/analysis'):
    os.makedirs(outdataDir + '/analysis')

# # Get the slim output we need to parse
inputFile = snakemake.input[0]
hxb2 = snakemake.params.hxb2_file

# Sample files for testing
# inputFile = '/net/feder/vol1/home/evromero/2025_hiv_linkage/data/slim_simulations/single_origin_sims/rep_9/slim_output.txt'
# hxb2 = '/net/feder/vol1/home/evromero/2025_hiv_linkage/data/hxb2_nuc_env.txt'
# outdataDir = '/net/feder/vol1/home/evromero/2025_hiv_linkage/data/slim_simulations/single_origin_sims/rep_9'

#Read in the hxb2 sequence we will use to get the relevant nucleotides
with open(hxb2) as f:
    hxb2_seq = f.readlines()[0]
    hxb2_seq = hxb2_seq.strip()

#Read in the slim output
with open(inputFile) as f:
    lines = f.read()
    #Separate each output section and get rid of the script text
    lines = lines.split('#OUT')[1:]

    
    #For each timepoint take the latest occurrence (since sometimes
    # the mutation is lost and the log file restarts)
    time_dict_seg = {}
    time_dict_fixed = {}
    for curr_section in lines:
        #Get the timepoint
        first_line = curr_section.split('\n')[0]
        first_line = first_line.split(' ')
        last_line = curr_section.split('\n')[-2]
        if 'RESTARTING' in last_line or 'ESTABLISHED' in last_line:
            curr_section = curr_section.split('\n')[:-2]
            curr_section = '\n'.join(curr_section)
        
        time = int(first_line[1])
        type = first_line[3]
        if type == 'SS':
            time_dict_seg[time] = curr_section
        else:
            time_dict_fixed[time] = curr_section
    #Now we can combine the sections back together
    lines = list(time_dict_seg.values()) + list(time_dict_fixed.values())



#Make dataframes to store the segregating and fixed loci output
seg_loci_slim_df = []
fixed_loci_slim_df = []

#Look at the number of reads in the data
samplenum = []

#Make a dictionary of the haplotype dictionaries keyed by timepoint
time_hap_dict = {}

#Now separately parse each section
for curr_section in lines:
    #Get the timepoint
    first_line = curr_section.split('\n', 1)[0]
    first_line = first_line.split(' ')

    #Isolate the timepoint and convert to an int
    time = int(first_line[1])
    segregating = (first_line[3] == 'SS')
    if segregating: samplenum.append(int(first_line[5]))

    #Check if there is this section includes genotype info
    contains_genomes = len(curr_section.split('Haplosomes:')) > 1
    
    #Separate out and process the genotype info if it exists
    if contains_genomes:
        #Separate out the genotype info
        split_results = curr_section.split('Haplosomes:')
        genome_data = split_results[1]
        mut_data = split_results[0]

        #Organize the genotype data
        time_hap_dict[time] = sop.parse_genome_data(genome_data)
    else: mut_data = curr_section
    
    #Put the mutation data into a nice dataframe format
    if segregating:
        seg_loci_slim_df.append(sop.parse_output_section(mut_data))
    else:
        fixed_loci_slim_df.append(sop.parse_output_section(mut_data))

#Combine the segregating and fixed loci dataframes
seg_loci_slim_df = pd.concat(seg_loci_slim_df, ignore_index=True)
fixed_loci_slim_df = pd.concat(fixed_loci_slim_df, ignore_index=True)

#Get the number of samples
samplenum = np.unique(samplenum)
if len(samplenum) > 1:
    raise ValueError('The number of sampled individuals doesn\'t\
        match across timepoints. Frequencies cannot be properly calculated.')
samplenum = samplenum[0]
    
#Initialize the dataframes we will output
all_seg_loci_df = []
all_haplotype_df = []


#Now loop through timepoints in order and make the dataframes we need to output
all_fixed_times = fixed_loci_slim_df['time'].unique()
all_seg_times = seg_loci_slim_df['time'].unique()
all_times = np.unique(np.concatenate((all_fixed_times, all_seg_times)))
all_times.sort()

for curr_time in all_times:
    #Get the fixed mutations at this timepoint
    curr_fixed_df = fixed_loci_slim_df[fixed_loci_slim_df['time'] == curr_time]

    #Loop through the mutations and update the hxb2 sequence
    for index, row in curr_fixed_df.iterrows():
        #Get the nucleotide that is mutated to
        allele = row['allele']
        #Get the position of the mutation
        mut_pos = row['position']
        #Update the hxb2 sequence
        hxb2_seq = hxb2_seq[:mut_pos] + allele + hxb2_seq[mut_pos+1:]

    #Get the segregating mutations at this timepoint
    curr_seg_df = seg_loci_slim_df[seg_loci_slim_df['time'] == curr_time]
    

    #Make the segregating loci dataframe
    print('Making segregating loci dataframe for timepoint ' + str(curr_time))
    seg_loci_df = sop.make_seg_loci_df(curr_seg_df, hxb2_seq, samplenum, SEG_CUTOFF)
    seg_loci_df['timepoint'] = curr_time
    seg_loci_df['frag_len'] = len(hxb2_seq)

    #Now we can make the haplotypes dataframe
    print('Making haplotype dataframe for timepoint ' + str(curr_time))
    relevant_genomes = time_hap_dict[curr_time]
    haplotype_df = sop.make_haplotype_df(relevant_genomes, curr_seg_df, hxb2_seq)
    haplotype_df['timepoint'] = curr_time

    

    #10-07-2022 Elena commented out since we are going to go with the less stringent filtering
    # #Lastly, We will filter the haplotype df and add it to the list
    # print('Filtering dataframes for timepoint ' + str(curr_time))
    # haplotype_df = zu.filter_genotype_df(haplotype_df, seg_loci_df, CUTOFF, SUCCESS)


    #Add our dataframes to the lists
    all_seg_loci_df.append(seg_loci_df)
    all_haplotype_df.append(haplotype_df)

#Write the dataframes to files
all_seg_loci_df = pd.concat(all_seg_loci_df, ignore_index=True)
all_seg_loci_df.to_pickle(outdataDir + '/analysis/FilteredLoci')

all_haplotype_df = pd.concat(all_haplotype_df, ignore_index=True)
#sort the haplotype df columns so they are in order
all_haplotype_df = all_haplotype_df.reindex(sorted(all_haplotype_df.columns, key=lambda x: (x not in ['individual', 'timepoint'], x)), axis=1)
all_haplotype_df.to_pickle(outdataDir + '/analysis/FilteredGenotypes')