#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from ukf_ex1 import omission_params

sys.path.append('../../../stationsim')
from ukf2 import pickler

# %%

def ex1_input(model_params, ukf_params, test):
    """Update the model and ukf parameter dictionaries so that experiment 1 runs.

    
    - Define some lists of populations `num_age`, a list of proportions observed 
    `props`. We also generate a list of unique experiment ids for each repeat of a 
    population/proportion pair.
    - Construct a cartesian product list containing all unique combinations 
    of the above 3 parameters (pop/prop/run_id).
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

    num_age = [10, 20, 30]  # 10 to 30 agent population by 10
    # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
    props = [0.25, 0.5, 0.75, 1]
    run_id = np.arange(0, 30, 1)  # 30 runs

    param_list = [(x, y, z) for x in num_age for y in props for z in run_id]

    if not test:
        "assign parameters according to task array"
        n = param_list[int(sys.argv[1])-1][0]
        prop = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]
    else:
        "if testing use these parameters for a single quick run."
        n = 5
        prop = 0.5
        run_id = "test"

    model_params, ukf_params, base_model = omission_params(n, prop,
                                                           model_params, ukf_params)
    # One additional parameter outside of the experiment module for the run id
    ukf_params["run_id"] = run_id
    
    # Save function 
    # Function saves required data from finished ukf instance. 
    # For ukf_ex1 we pickle the whole ukf_ss class using ukf2.pickler.
    
    ukf_params["save_function"] = pickler
    # File name and destination of numpy file saved in `ex0_save`
    ukf_params["file_destination"] = "../results" 
    ukf_params["file_name"] = "ukf_agents_{}_prop_{}-{}".format(
        str(n).zfill(3),
        str(prop),
        str(run_id).zfill(3)) + ".pkl"

    return model_params, ukf_params, base_model


if __name__ == '__main__':
    test = True
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")
    arc_main(ex1_input, test)
