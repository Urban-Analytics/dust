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

import sys
import numpy as np
from arc import arc

sys.path.append("..")
sys.path.append("../..")
import modules.default_ukf_gcs_configs as configs
from modules.ex0.ukf_gcs_ex0 import benchmark_params, ex0_save

sys.path.append('../../..')
sys.path.append('../../../..')
from stationsim.ukf2 import ukf_ss

# %%

def ex0_parameters(n, parameter_lists, test):
    """let the arc task array choose experiment parameters to run

    Parameters
    ----------
    test : bool
        if test is true we choose some simple parameters to run on arc.
        this is to test the file works and produces results before running 
        a larger batch of jobs to have none of them succeed and 400 abort
        emails.
        
    parameter_lists : list
        `parameter_lists` is a list of lists where each element of the list
        is some set of experiment parameters we wish to run. E.g. this may be
        [10, 1.0, 1] for the first experiment running with 10 agents 100% 
        observed.

    Returns
    -------
    sample_rate, run_id : int
        `sample_rate` how often we assimilate and unique `run_id` for each n and prop.
    noise : float
        `noise` standard deviation of gaussian noise added to observations
    """
    
    if not test:
        sample_rate = parameter_lists[int(sys.argv[1])-1][0]
        noise = parameter_lists[int(sys.argv[1])-1][1]
        run_id = parameter_lists[int(sys.argv[1])-1][2]
        
    #If testing use some fast test parameters.
    else:
        sample_rate = 1
        noise = 1
        run_id = "test"
        
    return sample_rate, noise, run_id


def arc_ex0_main(n, parameter_lists, test):
    """main function to run ukf experiment 0 on arc.
    
    - load in deault params
    - choose experiment params using taks array
    - update default parameters using benchmark_params and chosen parameters
    - generate filename to save to
    - initatiate arc class and run ukf
    - save results to numpy files
    
    Parameters
    ----------
    n : int
        `n` number of agents
    parameter_lists : list
        `parameter_lists` is a list of lists where each element of the list
        is some set of experiment parameters we wish to run. E.g. this may be
        [10, 1.0, 1] for the first experiment running with 10 agents 100% 
        observed.
    test : bool
        if test is true we choose some simple parameters to run on arc.
        this is to test the file works and produces results before running 
        a larger batch of jobs to have none of them succeed and 400 abort
        emails.
    """
    # load in default params
    ukf_params = configs.ukf_params
    model_params = configs.model_params
    
    if test:
        n = 5
        model_params["seed"] = 8
        model_params["step_limit"] = 100
        
    # load in experiment 1 parameters
    sample_rate, noise, run_id = ex0_parameters(n, parameter_lists, test)
    # update model and ukf parameters for given experiment and its' parameters
    model_params, ukf_params, base_model =  benchmark_params(n, 
                                                             noise, 
                                                             sample_rate, 
                                                             model_params, 
                                                             ukf_params)
    
        
    #file name to save results to
    file_name = "gcs_config_agents_{}_rate_{}_noise_{}-{}".format(
        str(n).zfill(3),
        str(float(sample_rate)),
        str(float(noise)),
        str(run_id).zfill(3)) + ".npy"
    destination = "../results/"
    
    # initiate arc class
    ex0_arc = arc(test)
    arc_args = [ukf_params, model_params, base_model]
    # run ukf_ss filter for arc class
    u = ex0_arc.arc_main(ukf_ss, file_name, *arc_args)
    # save entire ukf class as a pickle
    ex0_arc.arc_save(ex0_save, destination, file_name)

if __name__ == '__main__':
    test = True
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")
        
    # Lists of parameters to vary over
    n = 30 # 10 to 30 agent population by 10
    sample_rate = [1, 2, 5, 10]  # assimilation rates 
    noise = [0, 0.25, 0.5, 1, 2, 5] #gaussian observation noise standard deviation
    run_id = np.arange(0, 30, 1)  # 30 repeats for each combination of the above parameters

    # Assemble lists into grand list of all combinations. 
    # Each experiment will use one item of this list.
    parameter_lists = [(x, y, z)
                  for x in sample_rate for y in noise for z in run_id]
    arc_ex0_main(n, parameter_lists, test)
