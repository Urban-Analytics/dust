#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:57:18 2020

@author: medrclaa
"""
import sys
import numpy as np
import multiprocessing

from rjmcmc_ukf import rjmcmc_ukf

sys.path.append("..")
import default_ukf_configs as configs
from ukf_fx import fx
from ukf_plots import ukf_plots

sys.path.append("../../../../stationsim")
from stationsim_model import Model


def hx3(state, **hx_kwargs):
    return state

def ex3_pickle_name(n):
    
    
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
    
    f_name = f"rjmcmc_ukf_agents_{n}.pkl"
    return f_name

def obs_key_func(state, **hx_kwargs):
    """categorises agent observation type for a given time step
    0 - unobserved
    1 - aggregate
    2 - gps style observed
    
    Parameters
    --------
    state : array_like
        Desired state (e.g. full state of agent based model) on which we wish
        to test the obs_key on. Generally this is only used if the obs key
        is calculated dynamically 
        
    hx_kwargs : dict
        key word arguments from the observation function used to 
    """
    n = hx_kwargs["pop_total"]
    key = 2 * np.ones(n)
    
    return key

def rjmcmc_params(n, model_params, ukf_params):
    
    model_params["pop_total"] = n
    base_model = Model(**model_params)
    
    prop = 1
    ukf_params["prop"] = prop
        
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = 0.01 * np.eye(2 * n)
    ukf_params["r"] = 0.01 * np.eye(2 * n)#sensor noise
    
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"] = {"base_model":base_model} 
    ukf_params["hx"] = hx3
    ukf_params["hx_kwargs"] = {"pop_total" : n}
    
    ukf_params["obs_key_func"] = obs_key_func
    
    ukf_params["file_name"] =  ex3_pickle_name(n)
    return model_params, ukf_params, base_model


def ex3_main(n):
    
    model_params = configs.model_params
    model_params["gates_out"] = 3
    ukf_params = configs.ukf_params
    model_params, ukf_params, base_model = rjmcmc_params(n, model_params,
                                                         ukf_params)                                                        
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    rjmcmc_UKF= rjmcmc_ukf(model_params, ukf_params, base_model)
    rjmcmc_UKF.main(pool)
    pool.close()
    pool.join()

    instance = rjmcmc_UKF.ukf_1
    destination = "../../plots"
    prefix = "rjmcmc_ukf_"
    save = False
    animate = False
    plts = ukf_plots(instance, destination, prefix, save, animate)


    truths = instance.truth_parser(instance)
    nan_array= instance.nan_array_parser(truths, instance.base_model)
    preds = instance.preds_parser(instance, True, truths)
    
    ukf_params = instance.ukf_params
    #forecasts = np.vstack(instance.forecasts)
    
    "remove agents not in model to avoid wierd plots"
    truths *= nan_array
    preds *= nan_array
    #forecasts*= nan_array
    
    "indices for unobserved agents"

    #plts.path_plots(obs, "Observed")
    plts.path_plots(preds[::instance.sample_rate], "Predicted")
    plts.path_plots(truths, "True")
    return rjmcmc_UKF

if __name__ == "__main__":
    
    n = 10
    rjmcmc_UKF = ex3_main(n)
    
    
    