#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WHATEVER YOU DO MAKE SURE YOU USE SCIPY<1.21 
or you cant depickle the experiments.
!!todo solve this
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

sys.path.append("../modules")
import default_ukf_configs as configs

sys.path.append("../../../stationsim")
from ukf2 import ukf_ss

# %%

def main(ex_input, test=False):
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

    # If not testing abort the run if no set of parameters are specified
    if not test:

        if len(sys.argv) != 2:
            print("I need an integer to tell me which experiment to run. \n\t"
                  "Usage: python run_pf <N>")
            sys.exit(1)

    # Load in default parameters for stationsim and ukf
    model_params = configs.model_params
    ukf_params = configs.ukf_params

    # Update model parameters  for given experiment using ex_input
    model_params, ukf_params, base_model = ex_input(
        model_params, ukf_params, test)
    
    # Start logging incase run fails. Specify filename and logging level.
    logging.basicConfig(filename = ukf_params["file_name"]+ ".log", level = logging.INFO)
    
    print("UKF params: " + str(ukf_params))
    print("Model params: " + str(model_params))

    # init and run ukf
    u = ukf_ss(model_params, ukf_params, base_model)
    u.main()

    #save data using specified save function
    #e.g. saves numpy array of grand medians for ex0 using `ex0_save`
    # or pickles the whole ukf class for experiment 1,2 using `pickle_save`
    
    ex_save = ukf_params["save_function"]
    ex_save(u, ukf_params["file_destination"], ukf_params["file_name"])

    #delete any test files that were saved for tidiness.
    if test:
        f_name = ukf_params["file_name"]
        print(f"Test successful. Deleting the saved test file : {f_name}")
        os.remove(ukf_params["file_destination"] + ukf_params["file_name"])


