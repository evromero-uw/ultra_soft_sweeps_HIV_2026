import numpy as np
import pandas as pd

def allele_persist_nulls(seg_freqs_all_df, time_sample_sizes, bootstraps):
    """Given a set of allele frequencies measured over multiple timepoints and
    a dictionary including the number of sequences sampled at each timepoint,
    this function calculates the null distribution of allele persistence using
    binomial draws.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seg_freqs_all_df: pd.DataFrame, a dataframe containing the frequencies of
                segregating alleles, their position, the time they were sampled,
                and any resistance mutations they were sampled alongside.
    time_sample_sizes: dict, a dictionary where each key is a timepoint and the
                value is the number of sequences sampled at that timepoint.
    bootstraps: int, the number of bootstraps to perform.

    Returns:
    --------
    all_persistence_df: pd.DataFrame, a dataframe containing the persistence of
                each allele at each timepoint for each bootstrap.
    
    """
    
    #Get the number of timepoints
    timepoints = time_sample_sizes.keys()
    timepoints = np.sort(list(timepoints))

    #Get the number of segregating sites
    positions = len(seg_freqs_all_df['position'].unique())
    pos_list = seg_freqs_all_df['position'].unique()

    allele_freqs = seg_freqs_all_df['freqs'].values
    print('The number of zero frequencies is:')
    print(len(allele_freqs[allele_freqs == 0]))
    all_persist_df = []

    #Loop through the bootstraps
    for curr_rep in range(bootstraps):
        persistence_df = []

        #Loop through the segregating sites.
        for i in range(positions):
            #Draw a random allele frequency
            curr_freq = np.random.choice(allele_freqs)
            curr_pos = pos_list[i]
            status = 'Null Unseen'
            comp_status = 'Null Unseen'

            #Loop through the timepoints
            for curr_timepoint in timepoints:
                num_samples = time_sample_sizes[curr_timepoint]

                #Sample sequences randomly
                rng = np.random.default_rng()
                num_alleles = np.sum(rng.binomial(num_samples, curr_freq))
                num_non_alleles = num_samples - num_alleles

                #If its time 0 only mark but don't save in dataframe
                if curr_timepoint == 0:
                    if num_alleles > 0:
                        status = 'Null Persisting'
                    if num_non_alleles > 0:
                        comp_status = 'Null Persisting'
                #Otherwise mark the allele status and save it in the dataframe
                else:
                    if num_alleles > 0:
                        if status == 'Null Unseen':
                            status = 'Null Gained'
                        elif status == 'Null Lost':
                            status = 'Null Regained'
                        else:
                            status = 'Null Persisting'
                    else:
                        if status != 'Null Unseen':
                            status = 'Null Lost'
                    if num_non_alleles > 0:
                        if comp_status == 'Null Unseen':
                            comp_status = 'Null Gained'
                        elif comp_status == 'Null Lost':
                            comp_status = 'Null Regained'
                        else:
                            comp_status = 'Null Persisting'
                    else:
                        if comp_status != 'Null Unseen':
                            comp_status = 'Null Lost'
                    persistence_df.append([curr_pos, 'NULL', curr_timepoint, status,
                                           curr_freq])
                    persistence_df.append([curr_pos, 'NULL', curr_timepoint,
                                            comp_status, 1 - curr_freq])
        persistence_df = pd.DataFrame(persistence_df,
                                        columns = ['position', 'allele', 
                                                'time', 'persistence', 'freq'])
        persistence_df['rep'] = curr_rep
        all_persist_df.append(persistence_df)
    
    #Loop through and summarize the bootstraps
    all_persistence_df = pd.concat(all_persist_df, ignore_index=True)
    all_persistence_df = all_persistence_df[
                            all_persistence_df['persistence'] != 'Null Unseen']
    all_persistence_df['time'] = ['Week ' + str(x) for x in all_persistence_df['time'].values]
    
    return all_persistence_df

def in_vivo_persistence(allele_freq_df):
    """Given a dataframe containing the frequency of each allele at each
    timepoint, this function calculates the persistence of each allele.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    allele_freq_df: pd.DataFrame, a dataframe containing columns with the 
                position, allele, timepoint, and frequency, as well as any
                resistance mutations the allele was sampled alongside.
    Returns:
    --------
    persistence_df: pd.DataFrame, a dataframe containing the persistence of
                each allele at each timepoint.
    """
    #Now for each allele mark it as persisting, gained, lost, or regained
    persistence_df = []
    timepoint_list = np.unique(allele_freq_df['time'].values)
    timepoint_list = np.sort(timepoint_list)

    #Loop through the alleles and mark their statuses at each timepoint
    for name, group in allele_freq_df.groupby(['position','allele']):
        status = 'Unseen'

        #Loop through the timepoints other than 0
        for i in range(len(timepoint_list)):
            curr_timepoint = timepoint_list[i]
            curr_freq = group[group['time'] == curr_timepoint]['freqs'].values

            #If its time 0 only mark but don't save in dataframe
            if curr_timepoint == 0:
                if curr_freq > 0:
                    status = 'Persisting'
            else:
                if curr_freq > 0:
                    if status == 'Unseen':
                        status = 'Gained'
                    elif status == 'Lost':
                        status = 'Regained'
                    else:
                        status = 'Persisting'
                else:
                    if status != 'Unseen':
                        status = 'Lost'
            persistence_df.append([name[0], name[1], curr_timepoint, status])

    persistence_df = pd.DataFrame(persistence_df, 
                                  columns = ['position', 'allele', 
                                             'time', 'persistence'])
    persistence_df = persistence_df[persistence_df['time'] != 0]
    persistence_df['time'] = ['Week ' + str(x) for x in \
                              persistence_df['time'].values]
    
    return persistence_df

############################ Summary Functions ################################
def summarize_persistence(all_persistence_df, sim_bool):
    """ Takes a dataframe of persistence information and counts the number of 
    alleles in each persistance group at each timepoint. If sim_bool is True,
    then the function assumes that the persistence dataframe is from a
    simulation and will calculate the 95% confidence intervals using 
    the bootstrapped reps in the dataframe. Otherwise, the function populates
    the 95% confidence intervals with the count in all fields.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    all_persistence_df: pd.DataFrame, a dataframe containing the persistence of
                    each allele at each timepoint for each bootstrap.
    sim_bool: bool, a boolean indicating whether the persistence dataframe is
                    from a simulation or not.
    
    Returns:
    --------
    persistence_summary_df: pd.DataFrame, a dataframe containing the average
                    number of times an allele was observed to persist at each 
                    timepoint and the 95% confidence intervals.
    """
    persistence_summary_df = []

    #Loop through the timepoints and calculate the average persistence
    for name, group in all_persistence_df.groupby(['rebound', 'persistence'],
                                                  observed=False):
        if sim_bool:
            rep_counts = np.unique(group['rep'], return_counts=True)[1]
            avg_count = np.median(rep_counts)
            lower_error = np.quantile(rep_counts, 0.025)
            upper_error = np.quantile(rep_counts, 0.975)
        else:
            avg_count = group.shape[0]
            lower_error = avg_count
            upper_error = avg_count

        persistence_summary_df.append([name[0], name[1], avg_count, lower_error,
                                      upper_error])
    persistence_summary_df = pd.DataFrame(persistence_summary_df,
                                        columns = ['time', 'persistence',
                                                    'avg_count', 'lower_error',
                                                    'upper_error'])
    return persistence_summary_df
    


    