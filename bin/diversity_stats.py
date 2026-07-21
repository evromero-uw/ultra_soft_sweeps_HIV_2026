import numpy as np
from scipy.spatial.distance import hamming


def calc_ave_pairwise_hamming(seqArr, seqDF, verbose = False):
    """
    Given a dataframe with sequence info, this function calculates the average 
    pairwise hamming distance between all sequences with indices listed in the
    dataframe. Note: the sequence array can contain additional sequences, but
    only the sequences with indices listed in the dataframe will be used.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
            and has a column that gives the index of the sequence in the
            array.
    Returns:
    --------
    ave_hamming: float, the average pairwise hamming distance between all
                sequences listed in the dataframe.        
    """
    #Get only the sequences that are in the dataframe
    index_vals = seqDF['seq_index'].values
    seqArr = seqArr[index_vals, :]
    
    #Calculate the pairwise hamming distance
    num_seqs = seqArr.shape[0]
    hamming_dists = []
    for i in range(num_seqs):
        for j in range(i, num_seqs):
            if i == j:
                continue

            #Get indices where either sequence has a gap
            gap_indices = np.logical_or(seqArr[i, :] == '-', seqArr[j, :] == '-')
            
            #Remove the gap indices from the sequences
            seq1 = seqArr[i, ~gap_indices]
            seq2 = seqArr[j, ~gap_indices]

            hamming_dists.append(hamming(seq1, seq2))
    if verbose:
        print(hamming_dists)
    hamming_dists = np.array(hamming_dists)
    hamming_dists = hamming_dists[~np.isnan(hamming_dists)]
    ave_hamming = np.mean(hamming_dists)
    return ave_hamming

def calc_tajimas_d(seqArr, seqDF):
    """
    Given a dataframe with sequence info, this function calculates the Tajima's
    D statistic for the sequences listed in the dataframe. Note: the sequence
    array can contain additional sequences, but only the sequences with indices
    listed in the dataframe will be used.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    seqArr: np.array, an array where each row is a sequence and each column is
                a position in that sequence. The sequences are aligned and the
                first row is the reference sequence (HXB2).
    seqDF: pd.DataFrame, a dataframe with all of the info about the sequences
            and has a column that gives the index of the sequence in the
            array.
    Returns:
    --------
    tajd: float, the Tajima's D statistic for the sequences listed in the
                dataframe.
    """
    #Calculate the average pairwise hamming distance
    ave_hamming = calc_ave_pairwise_hamming(seqArr.copy(), seqDF)

    #Get only the sequences that are in the dataframe
    index_vals = seqDF['seq_index'].values
    seqArr = seqArr[index_vals, :]

    num_segsites = 0
    #Convert the sequences to strings
    #Get the number of segregating sites
    for i in range(seqArr.shape[1]):
        curr_col = seqArr[:, i]
        curr_col = curr_col[curr_col != '-']
        curr_col = np.unique(curr_col)
        if len(curr_col) > 1:
            num_segsites += 1

    #Calculate the Tajima's D statistic
    n = seqArr.shape[0]
    a1 = taj_a1(n)
    e1 = taj_e1(n)
    e2 = taj_e2(n)

    tajD_numerator = ave_hamming - (num_segsites/a1)
    tajD_denom = np.sqrt((e1*num_segsites) + (e2*num_segsites*(num_segsites - 1)))
    tajd = tajD_numerator/tajD_denom

    return tajd

###############################################################################
############################## Taj D helper functions #########################
def taj_a1(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    a1 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    a1: float, the value of the a1 term in the Tajima's D statistic.
    """
    a1 = 0
    for i in range(1, n):
        a1 += 1/i
    return a1

def taj_a2(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    a2 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    a2: float, the value of the a2 term in the Tajima's D statistic.
    """
    a2 = 0
    for i in range(1, n):
        a2 += 1/(i**2)
    return a2

def taj_b1(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    b1 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    b1: float, the value of the b1 term in the Tajima's D statistic.
    """
    b1 = (n + 1)/(3*(n - 1))
    return b1

def taj_b2(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    b2 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    b2: float, the value of the b2 term in the Tajima's D statistic.
    """
    b2 = (2*(n**2 + n + 3))/(9*n*(n - 1))
    return b2

def taj_c1(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    c1 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    c1: float, the value of the c1 term in the Tajima's D statistic.
    """
    c1 = taj_b1(n) - (1/taj_a1(n))
    return c1

def taj_c2(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    c2 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    c2: float, the value of the c2 term in the Tajima's D statistic.
    """
    c2 = taj_b2(n) - ((n + 2)/(taj_a1(n)*n)) + (taj_a2(n)/(taj_a1(n)**2))
    return c2

def taj_e1(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    e1 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    e1: float, the value of the e1 term in the Tajima's D statistic.
    """
    e1 = taj_c1(n)/taj_a1(n)
    return e1

def taj_e2(n):
    """
    Helper function for Tajima's D statistic. This function calculates the
    e2 term in the Tajima's D statistic.
    ---------------------------------------------------------------------------
    Parameters:
    -----------
    n: int, the number of sequences in the sample.
    Returns:
    --------
    e2: float, the value of the e2 term in the Tajima's D statistic.
    """
    e2 = taj_c2(n)/(taj_a1(n)**2 + taj_a2(n))
    return e2

            

