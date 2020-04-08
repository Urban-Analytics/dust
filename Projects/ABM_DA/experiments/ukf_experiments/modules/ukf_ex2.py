#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:19:48 2019

@author: rob

File for running experiment 2. We observe an aggregate grid square of counts
for how many agents are within each square as our observations. 

"""

import sys
import os
"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")

from modules.ukf_fx import fx
from modules.poly_functions import poly_count, grid_poly
from modules.ukf_plots import ukf_plots
import modules.default_ukf_configs as configs

sys.path.append("../../stationsim")
sys.path.append("modules")
from ukf2 import ukf_ss, pickle_main, batch_save, batch_load
from stationsim_model import Model

import numpy as np

def obs_key_func(state,**obs_key_kwargs):
    """categorises agent observation type for a given time step
    0 - unobserved
    1 - aggregate
    2 - gps style observed
    
    
    """
    
    key = np.ones(obs_key_kwargs["pop_total"])
    
    return key

def aggregate_params(n, bin_size, model_params, ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 2
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    
    model_params["pop_total"] = n
    base_model = Model(**model_params)
    
    ukf_params["bin_size"] = bin_size
    ukf_params["poly_list"] = grid_poly(model_params["width"],
              model_params["height"],ukf_params["bin_size"])
        
    ukf_params["p"] = np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = 1 * np.eye(2*n)
    ukf_params["r"] = 0.001 * np.eye(len(ukf_params["poly_list"]))#sensor noise 
    
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"]  = {"base_model" : base_model}
    ukf_params["hx"] = hx2
    ukf_params["hx_kwargs"] = {"poly_list":ukf_params["poly_list"]}
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["obs_key_kwargs"]  = {"pop_total" : n}

    ukf_params["file_name"] = ex2_pickle_name(n, bin_size)    
    
    
    return model_params, ukf_params, base_model
    
def hx2(state,**hx_kwargs):
    
    
    """Convert each sigma point from noisy gps positions into actual measurements
    
    - take some desired state vector of all agent positions
    - count how many agents are in each of a list of closed polygons using poly_count
    - return these counts as the measured state equivalent
    
    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted
    
    **hx_args
        generic hx kwargs
    Returns
    ------
    counts : array_like
        forecasted `counts` of how many agents in each square to 
        compare with actual counts
    """
    poly_list = hx_kwargs["poly_list"]
    
    counts = poly_count(poly_list, state)
    
    if np.sum(counts)>0:
        counts /= np.sum(counts)
    return counts

def ex2_pickle_name(n, bin_size):
    
    
    """build name for pickle file
    
    Parameters
    ------
    n, bin_size : float
        `n` population and `bin_size` aggregate square size
    
    Returns
    ------
    
    f_name : str
        return `f_name` file name to save pickle as
    """
    
    f_name = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"  
    return f_name

    
def ex2_plots(instance, destination, prefix, save, animate):
    
    
    """plots for experiments 2
    
    Parameters
    ------
    
    instance : class
    
        ukf_ss `instance` containing a finished stationsim run to publish
        
    destination, prefix : str
        
        `destination to save 
    save, animate : bool
    """
    
    marker_attributes = {
    "markers" : {-1 : "o", 0 : "X",  1 : "^"},
    "colours" : {-1 : "black", 0 : "orangered", 1 : "yellow"},
    "labels" :  {-1 : "Pseudo-Truths" ,0 : "Unobserved", 1 : "Aggregated"}
    }
    
    plts = ukf_plots(instance, destination, prefix, save, animate, marker_attributes)
        
    "pull data and put finished agents to nan"
    obs, preds, truths, nan_array= instance.data_parser()
    obs_key = instance.obs_key_parser()
    forecasts = np.vstack(instance.forecasts)

    truths *= nan_array
    preds *= nan_array
    forecasts *= nan_array
    
    plts.pair_frame(truths, preds, obs_key, 50, "plots/")
    plts.heatmap_frame(truths,50, "plots/")
    plts.error_hist(truths, preds,"Aggregate")
    
    "remove nan rows to stop plot clipping"
    plts.path_plots(preds[::instance.sample_rate], "Predicted", 
                    polygons = instance.ukf_params["poly_list"])
    plts.path_plots(truths, "True")    
    
    
    if animate:
                
        #plts.trajectories(truths, "plots/")
        #plts.heatmap(truths,truths.shape[0], "plots/")
        plts.pair_frames(truths, forecasts, np.vstack(obs_key), 
                         truths.shape[0], "plots/")

def ex2_main(n, bin_size, recall, do_pickle, source, destination):
    
    
    """main function to run experiment 2
    
    - build model and ukf dictionary parameters based on n and bin_size
    - initiate Model and ukf_ss based on new dictionaries
    - run ABM with filtering on top
    - make plots using finished run data
    
    Parameters
    ------
    n, bin_size : float
        `n` population and aggregate grid square size `bin_size`
    
    recall, do_pickle : bool
        `recall` a previous run or  `do_pickle` pickle a new one?
        
    source, destination : str
        `source` where to load/save any pickles and the `destination` of 
        any plots
    """
    
    if not recall:
        
                
        model_params = configs.model_params
        model_params["random_seed"] = 8
        ukf_params = configs.ukf_params
        
        model_params, ukf_params, base_model = aggregate_params(n,
                                        bin_size, model_params, ukf_params)
        
        batch = False
        ukf_params["batch"] = batch
        if batch:
            print("WARNING: Batch set to true and will not generate a random model each time.")
            try:
                seed = 50
                file_name =   f"batch_test_{n}_{seed}.pkl"
                batch_truths, batch_start_model = batch_load(file_name)
                print("batch data found.")
            except:
                print("no model found. generating one with given seed")
                file_name =   f"batch_test_{n}_{seed}.pkl"
                batch_save(model_params, n, seed)
                batch_truths, batch_start_model = batch_load(file_name)
                print("new model generated.")
                new_seed = int.from_bytes(os.urandom(4), byteorder='little') if seed == None else seed
                np.random.seed(new_seed)
            
            
            base_model = batch_start_model    

        print(f"Population: {n}")
        print(f"Square grid size: {bin_size}")
        
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        if do_pickle:
            pickle_main(ukf_params["file_name"], pickle_source, do_pickle, u)
       
    else:
        f_name = ex2_pickle_name(n, bin_size)
        try:
            u  = pickle_main("dict_" + f_name, source, do_pickle)
        except:
            print(f_name)
            print("dictionary not found. trying to load class")
            u  = pickle_main(f_name, source, do_pickle)
            
    ex2_plots(u, destination, "agg_ukf_", True, False)
    return u
       
if __name__ == "__main__":
    n = 20
    bin_size = 5
    recall = False #  recall previous run
    do_pickle = True #  pickle new run
    pickle_source = "pickles/"
    destination  = "plots/"
    u = ex2_main(n, bin_size, recall, do_pickle, pickle_source, destination)
    