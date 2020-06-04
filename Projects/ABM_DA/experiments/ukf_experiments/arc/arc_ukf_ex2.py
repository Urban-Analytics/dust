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
from ukf_ex2 import aggregate_params

sys.path.append('../../../stationsim')
from ukf2 import pickler

# %%
def ex2_input(model_params, ukf_params, test):
    """Update the model and ukf parameter dictionaries so that experiment 2 runs.

    
    - Define some lists of populations `num_age`, a list of aggregate squares 
    `bin_size`. We also generate a list of unique experiment ids for each 
    population/grid_square pair.
    - Construct a cartesian product list containing all unique combinations 
    of the above 3 parameters (pop/bin_size/run_id).
    - Let arc's test array functionality choose some element of the above 
    product list as parameters for an individual experiment.
    - Using these parameters update model_params and ukf_params using 
    ex2_params.
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

    num_age = [10, 20, 30, 50]  # 10 to 30 agent population by 10
    # unitless grid square size (must be a factor of 100 and 200)
    bin_size = [5, 10, 25, 50]
    run_id = np.arange(0, 30, 1)  # 30 runs

    param_list = [(x, y, z) for x in num_age for y in bin_size for z in run_id]

    if not test:
        n = param_list[int(sys.argv[1])-1][0]
        bin_size = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]

    else:
        n = 5
        bin_size = 50
        run_id = "test2"

    model_params, ukf_params, base_model = aggregate_params(n, bin_size,
                                                            model_params, ukf_params)
    # One additional parameter outside of the experiment module for the run id
    ukf_params["run_id"] = run_id
    
    # Save function 
    # Function saves required data from finished ukf instance. 
    # For ukf_ex1 we pickle the whole ukf_ss class using ukf2.pickler.
    
    ukf_params["save_function"] = pickler
    # File name and destination of numpy file saved in `ex0_save`
    ukf_params["file_destination"] = "../results" 
    ukf_params["file_name"] = "agg_ukf_agents_{}_bin_{}-{}".format(
        str(n).zfill(3),
        str(bin_size),
        str(run_id).zfill(3)) + ".pkl"

    return model_params, ukf_params, base_model

if __name__ == '__main__':
    test = True
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")
    arc_main(ex2_input, test)

