import sys
import numpy as np
import pandas as pd
import math

def r2(AB_obs, Ab_obs, aB_obs, ab_obs):
    """ Takes in the numbers of observations of each haplotype and outputs
    the R^2 statistic
    ---------------------------------------------------------------------------
    returns
    -------
    r_squared: float, The R^2 statistic
    None if frequencies of alleles A or B are greater than 99%
    """
    #sum our haplotypes to get total number of observations
    allSum = AB_obs + aB_obs + Ab_obs + ab_obs

    #calculate the frequencies of our alleles
    p_A = (Ab_obs + AB_obs)/float(allSum)
    p_B = (AB_obs + aB_obs)/float(allSum)

    #make sure we wont get a value error
    if p_A > 0.99 or p_B > 0.99 or p_A < 0.01 or p_B < 0.01: 
        # print("Frequencies were too low")
        return None
    

    topFrac = AB_obs/ (AB_obs + aB_obs + Ab_obs + ab_obs)
    numerator = topFrac - (p_A * p_B)
    numerator = numerator ** 2
    denominator = p_A * p_B * (1-p_A) * (1-p_B)

    r_squared = numerator / denominator
    return r_squared

def calc_D_prime(AB_obs, Ab_obs, aB_obs, ab_obs):
    """ Takes in the numbers of observations of each haplotype and outputs
    the D' statistic
    ---------------------------------------------------------------------------
    returns
    -------
    d_stat: float, The D' statistic
    None if frequencies of alleles A or B are greater than 99%
    """
    #sum our haplotypes to get total number of observations
    allSum = AB_obs + aB_obs + Ab_obs + ab_obs

    #calculate the frequencies of our alleles
    p_A = (Ab_obs + AB_obs)/float(allSum)
    p_B = (AB_obs + aB_obs)/float(allSum)

    #make sure we wont get a value error
    if p_A > 0.99 or p_B > 0.99 or p_A < 0.01 or p_B < 0.01: 
        # print("Frequencies were too low")
        return None
    
    AB_freq = AB_obs/allSum
    Ab_freq = Ab_obs/allSum
    aB_freq = aB_obs/allSum
    ab_freq = ab_obs/allSum
    
    d_stat = AB_freq * ab_freq - aB_freq * Ab_freq
    D_max = None
    if d_stat > 0:
        D_max = min(p_A * (1-p_B), (1-p_A) * p_B)
    elif d_stat == 0:
        return 0
    else:
        D_max = min(p_A * p_B, (1-p_A) * (1-p_B))

    d_stat = d_stat / D_max 
    return abs(d_stat)

def calc_D(AB_obs, Ab_obs, aB_obs, ab_obs):
    """ Takes in the numbers of observations of each haplotype and outputs
    the D statistic. 
    ---------------------------------------------------------------------------
    returns
    -------
    d_stat: float, The D statistic
    None if frequencies of alleles A or B are greater than 99%
    """
    #sum our haplotypes to get total number of observations
    allSum = AB_obs + aB_obs + Ab_obs + ab_obs

    #calculate the frequencies of our alleles
    p_A = (Ab_obs + AB_obs)/float(allSum)
    p_B = (AB_obs + aB_obs)/float(allSum)

    #make sure we wont get a value error
    if p_A > 0.99 or p_B > 0.99 or p_A < 0.01 or p_B < 0.01: 
        # print("Frequencies were too low")
        return None
    
    AB_freq = AB_obs/allSum
    Ab_freq = Ab_obs/allSum
    aB_freq = aB_obs/allSum
    ab_freq = ab_obs/allSum
    
    d_stat = AB_freq * ab_freq - aB_freq * Ab_freq
    return d_stat