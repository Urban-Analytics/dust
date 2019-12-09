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

sys.path.append("../../stationsim")
from ukf2 import ukf_ss, pickler, depickler
from stationsim_model import Model

import numpy as np

def obs_key_func(state,model_params,ukf_params):
        """which agents are observed"""
        
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
    ukf_params["pickle_file_name"] = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"    
    
    
    return model_params, ukf_params
    
def hx2(state,model_params,ukf_params):
        """Convert each sigma point from noisy gps positions into actual measurements
        
        -   uses function poly_count to count how many agents in each closed 
            polygon of poly_list
        -   converts perfect data from ABM into forecasted 
            observation data to be compared and assimilated 
            using actual observation data
        
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
    
def ex2_plots(instance,plot_dir,animate,prefix):
    plts = ukf_plots(instance,plot_dir,prefix)
        
    "pull data and put finished agents to nan"
    obs,preds,full_preds,truth,obs_key,nan_array= instance.data_parser()
    ukf_params = instance.ukf_params
    truth[~nan_array]=np.nan
    preds[~nan_array]=np.nan
    full_preds[~nan_array]=np.nan

    plts.pair_frame(truth, preds, obs_key, 50)
    plts.heatmap_frame(truth,ukf_params,50)
    plts.error_hist(truth, preds,"Aggregate", False)
    plts.path_plots(truth,preds, False)
    
    if animate:
                
        #plts.trajectories(truth)
        plts.heatmap(truth,ukf_params,truth.shape[0])
        if ukf_params["sample_rate"]>1:
            plts.pair_frames_animation(truth,full_preds,range(truth.shape[0]))
        else:
            plts.pair_frames_animation(truth,preds)
        
if __name__ == "__main__":
    """__main__ experimental main function for ukf2. mostly for testing in spyder.
    
    Refere to notebook ukf_experiments2 to see this in action
    
    """
    recall = True #recall previous run
    do_pickle = True #pickle new run
    n = 30
    bin_size = 25
    if not recall:
        model_params, ukf_params = aggregate_params(n, bin_size)
        
        print(model_params)
        print(ukf_params)
        
        base_model = Model(**model_params)
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        
        if do_pickle:
            pickler("", ukf_params["pickle_file_name"], u)
       
    else:
        
        f_name = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"  
    
        source = ""

        u = depickler(source, f_name)
    
    "unhash the necessary one"

    ex2_plots(u,"",False,"agg_ukf_")
    