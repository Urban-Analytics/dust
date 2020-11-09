#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WORTH HAVING SCIPY<1.21 IF DEALING WITH OLD DATA.

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

# %%

class arc():
    
    def __init__(self, test = False):
        """load in updated parameters and model so DA algorithm runs
        
        Parameters
        ----------
        filter_params, model_params : dict
            dictionaries of `model_params` model parameters and `filter_params` 
            DA filter parameters
        base_model : class
            `base_model` some base model to run the DA algorithm on. 
            stationsim for now.
        test : bool
            if `test` is true we choose some simple parameters to run on arc.
            this is to test the file works and produces results before running 
            a larger batch of jobs to have none of them succeed and 400 abort
            emails.
        """
        self.test = test
        
    def arc_main(self, filter_function, file_name, *args):
        """ main function for running experiments on arc
        
        - load updated filter and model parameters and model.
        - init model with the above
        - run model  via .main() method.
        
        Parameters
        ----------
        filter_function : cls
            `filter_function` class that applies some DA algorithm. Needs an init
            that loads model_params, filter_params, base_model
        file_name : TYPE
            DESCRIPTION.
        """
        # If not testing abort the run if no set of parameters are specified        
        if not self.test:
            if len(sys.argv) != 2:
                print("I need an integer to tell me which experiment to run. \n\t"
                      "Usage: python run_pf <N>")
                sys.exit(1)
        
        # Start logging incase run fails. Specify filename and logging level.
        logging.basicConfig(filename = file_name + '.log', level = logging.INFO)
                
        print("Model params: " + str(args[0]))
        print("Filter params: " + str(args[1]))
    
        # init and run ukf
        self.u = filter_function(*args)
        self.u.main()
        
        return self.u
        #save data using specified save function
        #e.g. saves numpy array of grand medians for ex0 using `ex0_save`
        # or pickles the whole ukf class for experiment 1,2 using `pickle_save`
        
        
    def arc_save(self, ex_save, destination, file_name):
        """save some data from an experiment for analysis
        
        Parameters
        ----------
        ex_save : func
            `ex_save` function that takes finished DA algorithm class u and saves some 
            data from it.
        destination , file_name : str
            `destination` where the data is saved to and `file_name` the file name
        """
        ex_save(self.u, destination, file_name)
    
        #delete any test files that were saved for tidiness.
        if self.test:
            print(f"Test successful. Deleting the saved test file : {file_name}")
            os.remove(destination + file_name)


