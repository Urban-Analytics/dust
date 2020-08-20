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
import sys
import numpy as np

from arc import arc

sys.path.append("../modules")
from ex3.ukf_ex3 import rj_params, get_gates, set_gates
from ex3.rjmcmc_ukf import rjmcmc_ukf

sys.path.append("../modules")
import default_ukf_configs as configs

sys.path.append('../../..')
#sys.path.append('../../../../stationsim')

from stationsim.ukf2 import pickler, ukf_ss

# %%

def ex3_parameters(parameter_lists, test):
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
        "assign parameters according to task array"
        n = parameter_lists[int(sys.argv[1])-1][0]
        jump_rate = parameter_lists[int(sys.argv[1])-1][1]
        theta = parameter_lists[int(sys.argv[1])-1][2]
        run_id = parameter_lists[int(sys.argv[1])-1][3]
    else:
        "if testing use these parameters for a single quick run."
        n = 5
        jump_rate = 5
        n_jumps = 3
        run_id = "test"
        
    return n, jump_rate, n_jumps, run_id


def arc_ex3_main(parameter_lists, test):
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
        
    Returns
    ------
    u : cls
        `u` instance of finished rjukf run
    """
    # load in default params
    ukf_params = configs.ukf_params
    model_params = configs.model_params
    if test:
        model_params["seed"] = 8
        
    # load in experiment 1 parameters
    n, jump_rate, n_jumps, run_id = ex3_parameters(parameter_lists, test)
    # update model and ukf parameters for given experiment and its' parameters
    model_params, ukf_params, base_model =  rj_params(n, jump_rate, n_jumps,
                                                            model_params, 
                                                            ukf_params)
    #file name to save results to
    file_name = "rjukf_agents_{}_rate_{}_jumps_{}-{}".format(
        str(n).zfill(3),
        str(jump_rate),
        str(n_jumps),
        str(run_id).zfill(3)) + ".pkl"
    #where to save the file
    destination = "../results/" 
    
    # initiate arc class
    ukf_args =  model_params, ukf_params, base_model, get_gates, set_gates
    ex3_arc = arc(test)
    # run ukf_ss filter for arc class
    u = ex3_arc.arc_main(rjmcmc_ukf, file_name, *ukf_args)
    # save entire ukf class as a pickle
    ex3_arc.arc_save(pickler, destination, file_name)
    
    return u

if __name__ == '__main__':
    
    #if testing set to True. if running batch experiments set to False
    test = True
    if test:
        print("Test set to true. If you're running an experiment, it wont go well.")

    # agent populations from 10 to 30
    num_age = [10, 20, 30]  
    # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
    props = [0.25, 0.5, 0.75, 1]
    # how many experiments per population and proportion pair. 30 by default.
    run_id = np.arange(0, 30, 1)
    #cartesian product list giving all combinations of experiment parameters.
    param_list = [(x, y, z) for x in num_age for y in props for z in run_id]
    
    u = arc_ex3_main(param_list, test)

