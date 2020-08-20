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

sys.path.append("../modules/ex1")
from ukf_ex1 import omission_params
sys.path.append("../modules")
import default_ukf_configs as configs

sys.path.append('../../../stationsim')
sys.path.append('../../../../stationsim')

from ukf2 import pickler, ukf_ss

# %%

def ex1_parameters(parameter_lists, test):
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
        prop = parameter_lists[int(sys.argv[1])-1][1]
        run_id = parameter_lists[int(sys.argv[1])-1][2]
    else:
        "if testing use these parameters for a single quick run."
        n = 5
        prop = 0.5
        run_id = "test"
        
    return n, prop, run_id


def arc_ex1_main(parameter_lists, test):
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
    n, prop, run_id = ex1_parameters(parameter_lists, test)
    # update model and ukf parameters for given experiment and its' parameters
    model_params, ukf_params, base_model =  omission_params(n, 
                                                            prop, 
                                                            model_params, 
                                                            ukf_params)
    #file name to save results to
    file_name = "ukf_agents_{}_prop_{}-{}".format(
        str(n).zfill(3),
        str(prop),
        str(run_id).zfill(3)) + ".pkl"
    #where to save the file
    destination = "../results/" 
    
    # initiate arc class
    ex1_arc = arc(ukf_params, model_params, base_model, test)
    # run ukf_ss filter for arc class
    u = ex1_arc.arc_main(ukf_ss, file_name)
    # save entire ukf class as a pickle
    ex1_arc.arc_save(pickler, destination, file_name)

if __name__ == '__main__':
    
    #if testing set to True. if running batch experiments set to False
    test = False
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
    
    arc_ex1_main(param_list, test)

