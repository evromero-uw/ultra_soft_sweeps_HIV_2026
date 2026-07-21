import os
from snakemake.script import snakemake
from string import Template

#In this file I am going to be making directories for running the slim
#simulations on the cluster
OUT_DIR_PATH = snakemake.params.out_sim_dir
IN_SIM_TEMPLATE = snakemake.params.in_sim_template
rep = snakemake.wildcards.rep
num_origins = snakemake.wildcards.num_origins
min_num_origins = snakemake.wildcards.min_num_origins

#Make the slim script
script_name = os.path.join(OUT_DIR_PATH, f'sim_origins_{num_origins}_min_{min_num_origins}_rep_{rep}.slim')

#For now just copy a template script
with open(IN_SIM_TEMPLATE, 'r') as f:
    script_content = f.read()

    #Get the script add the variables into the script
    script_content = Template(script_content).substitute(num_origins=num_origins,
                                                         min_num_origins=min_num_origins)
with open(script_name, 'w') as f:
        f.write(script_content)

    
