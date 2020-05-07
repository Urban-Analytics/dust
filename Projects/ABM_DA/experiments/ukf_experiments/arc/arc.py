#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHATEVER YOU DO MAKE SURE YOU USE SCIPY<1.21 
or you cant depickle the experiments.

This file allows you to repeat multiple experiments in paralllel 
of the ukf on stationsim using Leeds' arc3 HPC cluster.

NOTE: Import data from `arc.py` with the following in a bash terminal:
    
scp remote_file_source* local_file_destination.

Given the appropriate directories e.g.

scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.

===========
ARC4 Version
===========

New version of the above for arc4 using a conda environment for my sanity.
Use the standard means of cloning the git but use this venv instead.

module load anaconda
conda create -p /nobackup/medrclaa/ukf_py python=3 numpy scipy matplotlib shapely imageio seaborn
source activate /nobackup/medrclaa/ukf_py


Extract files using usual scp commands
If we are accessing arc remotely we have two remote servers to go through 
and so use proxy jump. 
https://superuser.com/questions/276533/scp-files-via-intermediate-host
With the format:

scp -oProxyJump=user@remote-access.leeds.ac.uk
e.g.
scp -oProxyJump=medrclaa@remote-access.leeds.ac.uk medrclaa@arc4.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/results/agg* /Users/medrclaa/new_aggregate_results

"""
import os
import sys
sys.path.append('../../../stationsim')
sys.path.append("../modules")

import numpy as np
import default_ukf_configs as configs
from ukf_ex2 import aggregate_params
from ukf_ex1 import omission_params
from ukf_ex0 import ex0_params, ex0_save
from stationsim_model import Model
from ukf2 import ukf_ss, pickler




# %%
def ex0_input(model_params, ukf_params, test):
    """Update the model and ukf parameter dictionaries so that experiment 0 runs.

    
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

    n = 5  # 10 to 30 agent population by 10

    sample_rate = [1, 2, 5, 10]  # how often to assimilate with ukf
    # list of gaussian noise standard deviations
    noise = [0, 0.25, 0.5, 1, 2, 5]
    run_id = np.arange(0, 30, 1)  # 30 runs

    param_list = [(x, y, z)
                  for x in sample_rate for y in noise for z in run_id]

    if not test:
        sample_rate = param_list[int(sys.argv[1])-1][0]
        noise = param_list[int(sys.argv[1])-1][1]
        run_id = param_list[int(sys.argv[1])-1][2]

    else:
        sample_rate = 10
        noise = 5
        run_id = "test"

    model_params, ukf_params, base_model = ex0_params(n, noise, sample_rate,
                                                      model_params, ukf_params)

    ukf_params["run_id"] = run_id

    ukf_params["f_name"] = "config_agents_{}_rate_{}_noise_{}-{}".format(
        str(n).zfill(3),
        str(float(sample_rate)),
        str(float(noise)),
        str(run_id).zfill(3))

    ukf_params["file_name"] = ukf_params["f_name"] + ".npy"
    ukf_params["do_pickle"] = False

    return model_params, ukf_params, base_model    

def ex1_input(model_params, ukf_params, test):
    """Update the model and ukf parameter dictionaries so that experiment 1 runs.

    
    - Define some lists of populations `num_age`, a list of proportions observed 
    `prop`. We also generate a list of unique experiment ids for each 
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

    ukf_params["run_id"] = run_id
    ukf_params["file_name"] = "ukf_agents_{}_prop_{}-{}".format(
        str(n).zfill(3),
        str(prop),
        str(run_id).zfill(3)) + ".pkl"

    return model_params, ukf_params, base_model


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

    ukf_params["run_id"] = run_id
    ukf_params["file_name"] = "agg_ukf_agents_{}_bin_{}-{}".format(
        str(n).zfill(3),
        str(bin_size),
        str(run_id).zfill(3)) + ".pkl"

    return model_params, ukf_params, base_model

def pickle_save(u, destination, f_name):
    """save ukf class instances as pickles for experiment 1 and 2
    
    Parameters
    ------
    u : cls
        Some finished ukf_ss class `u` from which we wish to save information.
    
    destination, f_name : str
        The `destination` where any files are saved and the file name `f_name`.
    """
    
    pickler(u, destination, f_name)


def main(ex_input, ex_save, test=False):
    """main function for running ukf experiments in arc.

    Runs an instance of ukf_ss for some experiment and associated parameters.
    Either pickles the run or saves some user defined aspect of said run.

    - define some input experiment for arc
    - update model_params and ukf_params dictionaries for said experiment
    - run ukf_ss with said parameters
    - pickle or save metrics when run finishes.

    Parameters
    ------
    ex_input : func
        `ex_input` some experiment input that updates the default model_params
        and ukf_params dicitonaries with items needed for filter to run given
        experiment.

    ex_save : func
        Function `ex_save` some user provided functions that saves aspects
        from the ukf run of interest. The function requires some ukf_ss class 
        `u`, some destination to save any files `destination`, and a file name
        to save `file_name. For example, for experiment 0 we save a numpy 
        array of 3 grand medians in a numpy array. For experiments 1 and 2 we 
        pickle the entire ukf class as a class dictionary.
        
    test : bool
        if true run a `test` for the given input and save functions.
        This does not use the task arrays of the arc system but ensures
        that the 2 functions work before running any experiment and wasting
        time.

    """

    if not test:

        if len(sys.argv) != 2:
            print("I need an integer to tell me which experiment to run. \n\t"
                  "Usage: python run_pf <N>")
            sys.exit(1)

    model_params = configs.model_params
    ukf_params = configs.ukf_params

    model_params, ukf_params, base_model = ex_input(
        model_params, ukf_params, test)

    verbose = True
    "more info on ukf parameters."
    if test:
        "no info when testing. easier to read."
        verbose = False
        
    if verbose:
        
        print("UKF params: " + str(ukf_params))
        print("Model params: " + str(model_params))

    # init and run ukf
    u = ukf_ss(model_params, ukf_params, base_model)
    u.main()

    #save data using specified save function
    #e.g. saves numpy array of grand medians for ex0 using `ex0_save`
    # or pickles the whole ukf class for experiment 1,2 using `pickle_save`
    
    f_name = ukf_params["file_name"]
    ex_save(u, "../results/", ukf_params["file_name"])

    #delete any test files that were saved for tidiness.
    if test:
        print(f"Test successful. Deleting the saved test file : {f_name}")
        os.remove("../results/" + ukf_params["file_name"])


if __name__ == '__main__':
    test = True
    print("warning test set to true. if youre running an experiment, it wont go well.")
    #main(ex0_input, ex0_save, test)
    #main(ex1_input, pickle_save, test)
    main(ex2_input, pickle_save, test)


