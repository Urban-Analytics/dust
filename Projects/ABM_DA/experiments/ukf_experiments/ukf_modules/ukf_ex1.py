#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:16:33 2019

@author: rob

"""
import sys
from ukf_fx import fx
from ukf_plots import ukf_plots
from default_ukf_configs import model_params,ukf_params

sys.path.append("../../../stationsim")
from ukf2 import ukf_ss, pickler, depickler
from stationsim_model import Model

import numpy as np
from math import floor

def omission_index(n,sample_size):
    
    
    """randomly pick agents without replacement to omit 
    used in experiment 1 hx function
    
    Parameters 
    ------
    n,p : int
         population `n` and proportion `p` observed. need p in [0,1]
         
    Returns
    ------
    index,index2: array_like:
        `index` of which agents are observed and `index2` their correpsoding
         xy coordinate columns from the whole state space.
         E.g for 5 agents we have 5x2 = 10 xy coordinates.
         if we choose 0th and 2nd agents we also choose 0,1,4,5 xy columns.
    """
    index = np.sort(np.random.choice(n,sample_size,replace=False))
    "double up index to choose x and y positions columns. both are used."
    index2 = np.repeat(2*index,2) 
    "nudge every second item to take the ith+1 column (y coordinate corresponding to chosen x)"
    index2[1::2] += 1
    return index, index2


def hx1(state, model_params, ukf_params):
    
    
    """Convert each sigma point from noisy gps positions into actual measurements
    
    -   omits pre-definied unobserved agents given by index/index2
    
    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted

    Returns
    ------
    state : array_like
        `state` actual observed state
    """
    
    return  state[ukf_params["index2"]] 

def obs_key_func(state, model_params, ukf_params):
    
    
    """which agents are observed
    if agent in index2 fully observed and assign 2.
    else not unobserved and assign 0.
    
    Parameters
    ------
    state : array_like
        model `state` of agent positions
    model_params, ukf_params : dict
        dictionaries of parameters `model_params` for stationsim 
        and `ukf_params` for the ukf.
    
    Returns
    ------
    
    key : array_like
        `key` array same shape as the true positions indicates
        each agents observation type for each time point 
        (unobserved,aggregate,gps)
    
    """
    
    key = np.zeros(model_params["pop_total"])
    key[ukf_params["index"]] +=2
    return key

def omission_params(n, prop, model_params = model_params, ukf_params=ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 1
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    model_params["pop_total"] = n
    ukf_params["prop"] = prop
    ukf_params["sample_size"]= floor(n * ukf_params["prop"])

    
    ukf_params["index"], ukf_params["index2"] = omission_index(n, ukf_params["sample_size"])
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * ukf_params["sample_size"])#sensor noise
    
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx1
    
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["pickle_file_name"] = f"ukf_agents_{n}_prop_{prop}.pkl"    
        
    return model_params, ukf_params


def ex1_plots(instance,plot_dir,save, animate, prefix):
    
    
    """do plots for experiment 1
    
    - pull data from ukf_ss instance
    - filter out measurements from non-active agents
    - plot one pairs plot frame linking ukf predictions to true values by tethers
    - plot population of agent median L2 errors in a  histogram
    - plot agent paths for observations, ukf preditions, and true values
    - if animated
    - plot true paths for agents in trajectories
    - plot a pair plot frame for every model step and animate them together.
    
    Parameters
    ------
    
    instance : class
        uks_ss class `instance` after completed run.
    plot_dir, prefix : str
        `plot_dir` where to save the plot to e.g. "" for this directory or 
        "ukf_results/ for ukf_results"
        `prefix` puts some prefix in front of various picture file names.
        e.g. "ukf_" or "agg_ukf_" so you can differentiate them and not overwrite
        them
    save , animate : bool
        `save` plots or `animate` whole model run?
    
    """
    plts = ukf_plots(instance,plot_dir,prefix)
    "single frame plots"
    obs,preds,truths,nan_array= instance.data_parser()
    obs_key = instance.obs_key_parser()
    ukf_params = instance.ukf_params
    
    obs *= nan_array[::instance.sample_rate,instance.ukf_params["index2"]]
    truths *= nan_array
    preds *= nan_array

    index2 = ukf_params["index2"]
    not_index2 = np.array([i for i in np.arange(truths.shape[1]) if i not in index2])
    plts.pair_frame(truths, preds, obs_key, 50)
    plts.error_hist(truths[:,index2], preds[:,index2],"Observed Errors", save)
    plts.error_hist(truths[:,not_index2], preds[:,not_index2],"Unobserved Errors", save)
    plts.path_plots(obs, "Observed", save)
    "remove nan rows to stop plot clipping"
    plts.path_plots(preds[::instance.sample_rate], "Predicted", save)
    plts.path_plots(truths, "True", save)

    if animate:
        plts.trajectories(truths)
        plts.pair_frames_animation(truths,preds,range(truths.shape[0]))
        

if __name__ == "__main__":
    recall = True #recall previous run
    do_pickle = True #pickle new run
    pickle_source = "../test_pickles/"
    n= 30
    prop = 0.5
    
    if not recall:
        model_params, ukf_params = omission_params(n, prop)
        print(model_params)
        print(ukf_params)
        
        base_model = Model(**model_params)
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        
        if do_pickle:
            pickler(u, pickle_source, ukf_params["pickle_file_name"])
            
    else:
        f_name = f"ukf_agents_{n}_prop_{prop}.pkl"
        u = depickler(pickle_source, f_name)
        ukf_params = u.ukf_params
        model_params = u.model_params

    ex1_plots(u,"../plots/",False, False,"ukf_")

    
    
    
    