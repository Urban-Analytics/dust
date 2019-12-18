#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:19:48 2019

@author: rob
"""
import sys

from ukf_fx import fx
from poly_functions import poly_count, grid_poly
from ukf_plots import ukf_plots
from default_ukf_configs import model_params,ukf_params

sys.path.append("../../../stationsim")
from ukf2 import ukf_ss, pickle_main
from stationsim_model import Model

import numpy as np

def obs_key_func(state,model_params,ukf_params):
    """categorises agent observation type for a given time step
    0 - unobserved
    1 - aggregate
    2 - gps style observed
    
    
    """
    
    key = np.ones(model_params["pop_total"])
    
    return key

def aggregate_params(n, bin_size,model_params=model_params,ukf_params=ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 2
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    model_params["pop_total"] = n

    ukf_params["bin_size"] = bin_size
    ukf_params["poly_list"] = grid_poly(model_params["width"],
              model_params["height"],ukf_params["bin_size"])
        
    ukf_params["p"] = np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2*n)
    ukf_params["r"] = np.eye(len(ukf_params["poly_list"]))#sensor noise 
    
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx2
    
    
    
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["pickle_file_name"] = ex2_pickle_name(n, bin_size)    
    
    
    return model_params, ukf_params
    
def hx2(state,model_params,ukf_params):
    
    
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
    
    counts = poly_count(ukf_params["poly_list"],state)
    
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
    plts = ukf_plots(instance, destination, prefix, save, animate)
        
    "pull data and put finished agents to nan"
    obs, preds, truths, nan_array= instance.data_parser()
    obs_key = instance.obs_key_parser()
    ukf_params = instance.ukf_params
    
    truths *= nan_array
    preds *= nan_array
    
    plts.pair_frame(truths, preds, obs_key, 50)
    plts.heatmap_frame(truths,50)
    plts.error_hist(truths, preds,"Aggregate")

    "remove nan rows to stop plot clipping"
    plts.path_plots(preds[::instance.sample_rate], "Predicted")
    plts.path_plots(truths, "True")    
    
    
    if animate:
                
        #plts.trajectories(truth)
        plts.heatmap(truths,ukf_params,truths.shape[0])
        plts.pair_frames_animation(truths,preds)

def ex2_main(n, bin_size, recall, do_pickle, pickle_source):
    
    
    """main function to run experiment 1
    
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
        
    pickle_source : str
        `pickle_source` where to load/save any pickles.
    """
    
    if not recall:
        model_params, ukf_params = aggregate_params(n, bin_size)
        
        print(f"Population: {n}")
        print(f"Square grid size: {bin_size}")
        
        base_model = Model(**model_params)
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        pickle_main(ukf_params["pickle_file_name"], pickle_source, do_pickle, u)
       
    else:
        f_name = ex2_pickle_name(n, bin_size)
        u = pickle_main(f_name, pickle_source, do_pickle)

    ex2_plots(u, "../plots/", "agg_ukf_", True, False)
    
       
if __name__ == "__main__":
    n = 30
    bin_size = 25
    recall = True #  recall previous run
    do_pickle = True #  pickle new run
    pickle_source = "../test_pickles/"

    ex2_main(n, bin_size, recall, do_pickle, pickle_source)
    