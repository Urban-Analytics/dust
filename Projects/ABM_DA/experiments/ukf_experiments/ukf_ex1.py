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

sys.path.append("../../stationsim")
from ukf2 import ukf_ss, pickler, depickler
from stationsim_model import Model

import numpy as np
from math import floor

def omission_index(n,sample_size):
    
    
    """randomly pick agents to omit 
    used in experiment 1 hx function
    
    Parameters 
    ------
    n,p : int
         population `n` and proportion `p` observed. need p in [0,1]
         
    Returns
    ------
    index,index2: array_like:
        `index` of which agents are observed and `index2` their correpsoding
        index for xy coordinates from the desired state vector.
    """
    index = np.sort(np.random.choice(n,sample_size,replace=False))
    index2 = np.repeat(2*index,2)
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
    obs_state : array_like
        `obs_state` actual observed state
    """
    obs_state = state[ukf_params["index2"]]
    
    return obs_state   

def obs_key_func(state, model_params, ukf_params):
    """which agents are observed"""
    
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
    plts = ukf_plots(instance,plot_dir,prefix)
    "single frame plots"
    obs,preds,full_preds,truth,obs_key,nan_array= instance.data_parser()
    ukf_params = instance.ukf_params
    truth *= nan_array
    preds *= nan_array
    full_preds *= nan_array


    index2 = ukf_params["index2"]
    plts.pair_frame(truth, preds, obs_key, 50)
    plts.error_hist(truth[:,index2], preds[:,index2],"Observed Errors", save)
    plts.error_hist(truth[:,~index2], preds[:,~index2],"Unobserved Errors", save)
    plts.path_plots(truth, preds, save)
    
    if animate:
        plts.trajectories(truth)
        if ukf_params["sample_rate"]>1:
            plts.pair_frames_animation(truth,full_preds,range(truth.shape[0]))
        else:
            plts.pair_frames_animation(truth,preds)
    

if __name__ == "__main__":
    recall = True #recall previous run
    do_pickle = True #pickle new run
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
            pickler("", ukf_params["pickle_file_name"], u)
            
    else:
        f_name = f"ukf_agents_{n}_prop_{prop}.pkl"
        source = ""
        u = depickler(source, f_name)
        ukf_params = u.ukf_params
        model_params = u.model_params

    ex1_plots(u,"",True, False,"ukf_")

    
    
    
    