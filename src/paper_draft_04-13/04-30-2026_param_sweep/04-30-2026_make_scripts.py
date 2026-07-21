import os
import numpy as np
from snakemake.script import snakemake
from string import Template

#In this file I am going to be making directories for running the slim
#simulations on the cluster
OUT_DIR_PATH = snakemake.params.out_sim_dir
IN_SIM_TEMPLATE = snakemake.params.in_sim_template
NE = snakemake.params.NE
rep = snakemake.wildcards.rep
num_origins = snakemake.wildcards.num_origins
min_num_origins = snakemake.wildcards.min_num_origins
mu = snakemake.wildcards.mu
selection_coeff = snakemake.wildcards.selection_coeff
rho = snakemake.wildcards.rho

#Make the slim script
script_name = os.path.join(OUT_DIR_PATH, f'sim_origins_{num_origins}_min_{min_num_origins}_rep_{rep}_mu_{mu}_sel_{selection_coeff}_rho_{rho}.slim')

#For now just copy a template script
with open(IN_SIM_TEMPLATE, 'r') as f:
    script_content = f.read()

    #Calculate how many origins we need to introduce to get the desired
    #number of origins at the end of the simulation
    prob_fixation = (1 - np.exp(-2* float(selection_coeff))) / (1 - np.exp(-2*NE*float(selection_coeff)))
    num_origins_to_introduce = int(float(num_origins) / prob_fixation)

    #Get the script add the variables into the script
    script_content = Template(script_content).substitute(num_origins=num_origins_to_introduce,
                                                         min_num_origins=min_num_origins,
                                                         mu=mu,
                                                         selection_coeff=selection_coeff,
                                                         rho=rho)
with open(script_name, 'w') as f:
        f.write(script_content)

    
