import os
from snakemake.script import snakemake

#In this file I am going to be making directories for running the slim
#simulations on the cluster
OUT_DIR_PATH = snakemake.params.out_sim_dir
IN_SIM_TEMPLATE = snakemake.params.in_sim_template
rep = snakemake.wildcards.rep

#Make the slim script
script_name = os.path.join(OUT_DIR_PATH, f'sim_rep_{rep}.slim')

#For now just copy a template script
with open(IN_SIM_TEMPLATE, 'r') as f:
    script_content = f.read()
with open(script_name, 'w') as f:
        f.write(script_content)

    
