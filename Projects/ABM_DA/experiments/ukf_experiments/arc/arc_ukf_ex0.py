#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 14:16:25 2020

@author: medrclaa
"""

"""splitting up old arc.py file into individual experiment runs for clarity.

===========
ARC4 Version
===========

New version of the above for arc4 using a conda environment for my sanity.
Use the standard means of cloning the git but use this venv instead.

module load anaconda
conda create -p /nobackup/medrclaa/ukf_py python=3 numpy scipy matplotlib shapely imageio seaborn
source activate /nobackup/medrclaa/ukf_py


Extract files using usual scp commands
If we are accessing arc remotely we have two remote servers 
(one with virtually no storage) to go through so use proxy jump to avoid being
excommunicated by the arc team.

Shamelessly stolen from :

https://superuser.com/questions/276533/scp-files-via-intermediate-host

With the format:

scp -oProxyJump=user@remote-access.leeds.ac.uk
e.g.
scp -oProxyJump=medrclaa@remote-access.leeds.ac.uk medrclaa@arc4.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/results/agg* /Users/medrclaa/new_aggregate_results

"""
import logging
import os
import sys
import numpy as np
from arc import main as arc_main

sys.path.append("../modules")
from ukf_ex0 import benchmark_params, ex0_save

# %%
def ex0_input(model_params, ukf_params, test):
    """Update the model and ukf dictionaries so that experiment 0 runs on arc.

    
    - Define some lists of populations `num_age`, a list of proportions observed 
    `prop`. We also generate a list of unique experiment ids for each 
    population/proportion pair.
    - Construct a cartesian product list containing all unique combinations 
    of the above 3 parameters (sample_rate/noise/run_id).
    - Let arc's test array functionality choose some element of the above 
    product list as parameters for an individual experiment.
    - Using these parameters update model_params and ukf_params using 
    ex1_params.
    - Also add `run_id`, `file_name` to ukf_params required for arc run.
    - Output updated dictionaries.

    Parameters
    ------

    model_params, ukf_params : dict
        dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf.

    test : bool
        If we're testing this function change the file name slightly.
        
    Returns
    ------

    model_params, ukf_params : dict
        updated dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf. 

    """

    # Lists of parameters to vary over
    n = 10 # 10 to 30 agent population by 10
    sample_rate = [1, 2, 5, 10]  # assimilation rates 
    noise = [0, 0.25, 0.5, 1, 2, 5] #gaussian observation noise standard deviation
    run_id = np.arange(0, 30, 1)  # 30 repeats for each combination of the above parameters

    # Assemble lists into grand list of all combinations. 
    # Each experiment will use one item of this list.
    param_list = [(x, y, z)
                  for x in sample_rate for y in noise for z in run_id]

    # Let task array choose experiment parameters
    if not test:
        sample_rate = param_list[int(sys.argv[1])-1][0]
        noise = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]
        
    #If testing use some fast test parameters.
    else:
        sample_rate = 10
        noise = 5
        run_id = "test"

    # Update model and ukf parameter dictionaries so experiment 0 runs
    # See `uk_ex0.py` for more details
    model_params, ukf_params, base_model = benchmark_params(n, noise, sample_rate,
                                                      model_params, ukf_params)
    
    # These also need adding and aren't in the main experiment module
    # Unique run id
    ukf_params["run_id"] = run_id
    # Function saves required data from finished ukf instance. A numpy array in this case.
    ukf_params["save_function"] = ex0_save
    # File name and destination of numpy file saved in `ex0_save`
    ukf_params["file_destination"] = "../results"
    ukf_params["file_name"] = "config_agents_{}_rate_{}_noise_{}-{}".format(
        str(n).zfill(3),
        str(float(sample_rate)),
        str(float(noise)),
        str(run_id).zfill(3)) + ".npy"

    return model_params, ukf_params, base_model    

if __name__ == '__main__':
    test = False
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")
    arc_main(ex0_input, test)
