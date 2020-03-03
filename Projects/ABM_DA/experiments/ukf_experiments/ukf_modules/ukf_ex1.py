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
from ukf2 import ukf_ss, pickle_main
from stationsim_model import Model

import numpy as np
from math import floor

def omission_index(n,sample_size):
    
    
    """randomly pick agents without replacement to observe 
    
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
    
    
    """Measurement function for ex1 taking observed subset of agent positions
    
    - take full desired state X of all agent observations.
    - take subset of observed agent as our measured state Y.
    
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
    
    - if agent in index2 fully observed and assign 2.
    - else unobserved and assign 0.
    
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
        `key` array same shape as the measured state indicating
        each agents observation type for each time point 
        (unobserved,aggregate,gps) for 0, 1, or 2.
    
    """
    
    key = np.zeros(model_params["pop_total"])
    key[ukf_params["index"]] +=2
    return key

def omission_params(n, prop, model_params = model_params, ukf_params=ukf_params):
    
    
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
    ukf_params["pickle_file_name"] =  ex1_pickle_name(n, prop)
        
    return model_params, ukf_params

def ex1_pickle_name(n, prop):
    
    
    """build name for pickle file
    
    Parameters
    ------
    n, prop : float
        `n` population and `proportion` observed
        
    Returns
    ------
    
    f_name : str
        return `f_name` file name to save pickle as
    """
    
    f_name = f"ukf_agents_{n}_prop_{prop}.pkl"
    return f_name


def ex1_plots(instance, destination, prefix, save, animate):
    
    
    """do plots for experiment 1
    
    - extract truths, obs, ukf predictions (preds), and forecasts
    - remove inactive agent measurements to prevent bias
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
    plts = ukf_plots(instance, destination, prefix, save, animate)

    obs,preds,truths,nan_array= instance.data_parser()
    obs_key = instance.obs_key_parser()
    ukf_params = instance.ukf_params
    index2 = ukf_params["index2"]

    obs *= nan_array[::instance.sample_rate,index2]
    truths *= nan_array
    preds *= nan_array
    
    "indices for unobserved agents"
    not_index2 = np.array([i for i in np.arange(truths.shape[1]) if i not in index2])
    plts.pair_frame(truths, preds, obs_key, 50)
    plts.error_hist(truths[:,index2], preds[:,index2],"Observed Errors")
    plts.error_hist(truths[:,not_index2], preds[:,not_index2],"Unobserved Errors")
    plts.path_plots(obs, "Observed")
    "remove nan rows to stop plot clipping"
    plts.path_plots(preds[::instance.sample_rate], "Predicted")
    plts.path_plots(truths, "True")

    if animate:
        plts.trajectories(truths)
        plts.pair_frames_animation(truths,preds,range(truths.shape[0]))
        
def ex1_main(n, prop, recall, do_pickle, pickle_source):
    
    
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
        model_params, ukf_params = omission_params(n, prop)
        
        print(f"Population: {n}")
        print(f"Proportion Observed: {prop}")
        
        base_model = Model(**model_params)
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        pickle_main(ukf_params["pickle_file_name"],pickle_source, do_pickle,
                    u, model_params, ukf_params)
    
    else:
        "if recalling, load a pickle."
        f_name = ex1_pickle_name(n, prop)
        u  = pickle_main(f_name, pickle_source, do_pickle)
        model_params, ukf_params = u.model_params, u.ukf_params
        
    ex1_plots(u, "../plots/", "ukf_", True, False)

    
if __name__ == "__main__":
    recall = True #recall previous run
    do_pickle = True #pickle new run
    pickle_source = "../test_pickles/" #where to load/save pickles from
    n = 30 #population size
    prop = 0.5 #proportion observed
    
    ex1_main(n, prop, recall, do_pickle, pickle_source)