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

sys.path.append("../modules")
from ukf_ex2 import aggregate_params
import default_ukf_configs as configs

sys.path.append('../../../stationsim')
from ukf2 import ukf_ss, pickler

# %%

def ex2_parameters(parameter_lists, test):
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
    n, run_id : int
        agent population `n` and unique `run_id` for each n and prop.
    prop : float
        proportion of agents observed `prop`
    """
    
    if not test:
        n = parameter_lists[int(sys.argv[1])-1][0]
        bin_size = parameter_lists[int(sys.argv[1])-1][1]
        run_id = parameter_lists[int(sys.argv[1])-1][2]

    else:
        n = 5
        bin_size = 50
        run_id = "test2"
        
    return n, bin_size, run_id

def arc_ex2_main(parameter_lists, test):
    """main function to run ukf experiment 1 on arc.
    
    Parameters
    ----------
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
    # load in experiment 1 parameters
    n, bin_size, run_id = ex2_parameters(parameter_lists, test)
    # update model and ukf parameters for given experiment and its' parameters
    model_params, ukf_params, base_model =  aggregate_params(n, 
                                                            bin_size, 
                                                            model_params, 
                                                            ukf_params)

    destination = "../results/" 
    file_name = "agg_ukf_agents_{}_bin_{}-{}".format(
        str(n).zfill(3),
        str(bin_size),
        str(run_id).zfill(3)) + ".pkl"

    # initiate arc class
    ex2_arc = arc(ukf_params, model_params, base_model, test)
    # run ukf_ss filter for arc class
    u = ex2_arc.arc_main(ukf_ss, file_name)
    # save entire ukf class as a pickle
    ex2_arc.arc_save(pickler, destination, file_name)
    
if __name__ == '__main__':
    test = True
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")
    num_age = [10, 20, 30, 50]  # 10 to 30 agent population by 10
    # unitless grid square size (must be a factor of 100 and 200)
    bin_size = [5, 10, 25, 50]
    run_id = np.arange(0, 30, 1)  # 30 runs

    parameter_lists = [(x, y, z) for x in num_age for y in bin_size for z in run_id]

    arc_ex2_main(parameter_lists, test)

