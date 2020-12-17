#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:27:18 2020

@author: medrclaa

Random Omission Experiment for GCS StationSim model.
"""

import sys
import os

sys.path.append("../../../stationsim")

import numpy as np
from math import floor
import multiprocessing

sys.path.append("..")
sys.path.append("../..")
from modules.ukf_fx import fx
from modules.ukf_plots import ukf_plots
import modules.default_ukf_gcs_configs as configs
from modules.ex1.ukf_ex1 import omission_index, hx1, obs_key_func, ex1_pickle_name, ex1_plots

sys.path.append("../../..")
sys.path.append("../../../..")
from stationsim.ukf2 import *
#from stationsim_gcs_model import Model
from stationsim.stationsim_density_model import Model


def omission_params(n, prop, model_params, ukf_params):
    """update ukf_params with fx/hx and their parameters for experiment 1
    
    - assign population size and proportion observed.
    - randomly select agents to observed for index/index2
    - assign initial covariance p as well as sensor and process noise (q,r)
    - assign transition and measurement functions (fx,hx)
    - assign observation key function and numpy file name for saving later.
    
        
    Parameters
    ------
    n, prop : float
        `n` population and proportion observed 0<=`prop`<=1
        
    model_params, ukf_params : dict
        dictionaries of model `model_params` and ukf `ukf_params` parameters 
        
    Returns
    ------
    model_params, ukf_params : dict
        updated dictionaries of model `model_params` and ukf `ukf_params`
        parameters ready to use in ukf_ss
    """
    
    #finish model params and initiate base_model
    #population
    model_params["pop_total"] = n
    #model type either None or Grand_Central
    model_params["station"] = "Grand_Central"
    #load in station gcs model with given model params (default + pop/station)
    base_model = Model(**model_params)

    #experiment 1 specific parameters
    ukf_params["prop"] = prop
    ukf_params["sample_size"]= floor(n * prop)
    ukf_params["index"], ukf_params["index2"] = omission_index(n, ukf_params["sample_size"])
    
    #noise structures
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n) # process noise
    ukf_params["r"] = np.eye(2 * ukf_params["sample_size"])# sensor noise
    
    # Kalman functions and their experiment specific kwargs
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"] = {"base_model": base_model} 
    ukf_params["fx_kwargs_update"] = None
    
    ukf_params["hx"] = hx1
    ukf_params["hx_kwargs"] = {"index2" : ukf_params["index2"], "n" : n,
                               "index" : ukf_params["index"],}
    
    # function to say how each agent is observed for plotting
    ukf_params["obs_key_func"] = obs_key_func
    
    # what to save experiment as. This is a default and is often overwritten.
    ukf_params["record"] = True
    ukf_params["light"] = True
    ukf_params["file_name"] =  ex1_pickle_name(n, prop)
        
    return model_params, ukf_params, base_model
        
def ex1_main(n, prop, pool, recall, do_pickle, source, destination):
    """main function to run experiment 1
    
    - build model and ukf dictionary parameters based on n and prop
    - initiate Model and ukf_ss based on new dictionaries
    - run ABM with filtering on top
    - make plots using finished run data
    
    Parameters
    ------
    n, prop : float
        `n` population and proportion observed 0<=`prop`<=1
    
    recall, do_pickle : bool
        `recall` a previous run or  `do_pickle` pickle a new one?
        
    pickle_source : str
        `pickle_source` where to load/save any pickles.
    """

    if not recall:
        # load parameters
        model_params = configs.model_params
        #model_params["step_limit"] = 100
        ukf_params = configs.ukf_params
        #update parameters for ex1 and init model
        model_params, ukf_params, base_model = omission_params(n, prop,
                                                               model_params, ukf_params)
        print(f"Population: {n}")
        print(f"Proportion Observed: {prop}")
        
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        pickle_main(ukf_params["file_name"],pickle_source, do_pickle, u)
    
    else:
        # if recalling, load a pickle.
        f_name = ex1_pickle_name(n, prop)
        
        # try loading class_dicts first. If no dict then class instance.
        try:
            u  = pickle_main("dict_" + f_name, source, do_pickle)
        except:
            u  = pickle_main(f_name, source, do_pickle)
 
        model_params, ukf_params = u.model_params, u.ukf_params
    
    ex1_plots(u, destination, "ukf_gcs_", True, False)

    return u
    
if __name__ == "__main__":
    recall = False #recall previous run
    do_pickle = True #pickle new run
    pickle_source = "../../pickles/" #where to load/save pickles from
    destination = "../../plots/"
    n = 5 #population size
    prop = 1.0 #proportion observed
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

    u = ex1_main(n, prop, pool, recall, do_pickle, pickle_source, destination)