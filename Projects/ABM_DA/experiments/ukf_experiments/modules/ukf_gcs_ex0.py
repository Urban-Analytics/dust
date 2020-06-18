#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:49:27 2019

Experiment 0 module.

The idea here is to find a suitable range of parameters under which the UKF performs well.
We vary two parameters, namely the observation noise and sampling_rate. 

Noise provides the standard deviation to an additive Gaussian noise for our observations. 
A larger noise value gives noisy position data that is less accurate and worsens the 
quality of the observations. 0 noise gives the ground truth value. 

The sampling rate (aka assimilation rate) determines how often we assimilate forecasted
values from the ukf with our observations. A larger sampling rate allows forecasts
to run longer before being assimilated and potentially drift further from the truth.
Increasing this value significantly worsens the quality of stationsim prediction.
We require a natural number for this rate 1, 2, 3, ...

@author: rob
"""

import sys
import os

sys.path.append("../modules")
sys.path.append("../../../stationsim")
    
"import required modules"
from ukf_fx import fx
from ukf_plots import L2s
import default_ukf_gcs_configs as configs

"can misbehave when importing with ex1/ex2 modules as well"

from ukf2 import ukf_ss
from stationsim_gcs_model import Model

import numpy as np

def benchmark_params(n, noise, sample_rate, model_params, ukf_params):
    """update ukf_params with fx/hx and their parameters for experiment 0
    
    - assign population size, observation noise, and sampling/assimilation rate
    - assign initial covariance p as well as sensor and process noise (q,r)
    - assign transition and measurement functions (fx,hx)
    - assign observation key function and numpy file name for saving later.
    
    
    Parameters
    ------
    n, noise, sample_rate : float
        `n` population, additive `noise`, and `sample_rate` sampling rate
        
    model_params, ukf_params : dict
        dictionaries of model `model_params` and ukf `ukf_params` parameters 
        
    Returns
    ------
    model_params, ukf_params : dict
        updated dictionaries of model `model_params` and ukf `ukf_params`
        parameters ready to use in ukf_ss
    """
    
    model_params["pop_total"] = n
    ukf_params["noise"] = noise
    ukf_params["sample_rate"] = sample_rate
    model_params["station"] = "Grand_Central"
    base_model = Model(**model_params)

    #inital guess at state covariance
    ukf_params["p"] = np.eye(2 * n) 
    #process noise
    ukf_params["q"] = 0.05 * np.eye(2 * n)
    #sensor noise
    ukf_params["r"] = 0.01 * np.eye(2 * n)
    
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"] = {"base_model": base_model}
    ukf_params["hx"] = hx0    
    ukf_params["hx_kwargs"] = {}
    ukf_params["obs_key_func"] = obs_key_0
    
    ukf_params["file_name"] = f"config_agents_{n}_rate_{sample_rate}_noise_{noise}"
    
    return model_params, ukf_params, base_model

def hx0(state, **hx_kwargs):
    
    
    """ null transition function does nothing for ex0 hx. two states are the same.
    
    Parameters
    ------
    state : array_like
        `state` vector of agent positions
    
    model_params, ukf_params : dict
        dictionaries of model `model_params` and ukf `ukf_params` parameters
    """
    
    return state

def obs_key_0(state, **hx_params):
    """obs_key is constantly 2s (observed) for ex0
    
    Parameters
    ----------
    state : array_like
        `state` vector of agent positions
    **hx_params : kwargs
        `hx_params` some keyword arguements for transition function hx
        to calculate observation key.

    Returns
    -------
    obs_key : array_like
        vector indicating observation status of each agent (0, 1, or 2).

    """
    obs_key = 2 * np.ones(int(state.shape[0]/2))
    return obs_key

def ex0_save(instance,source,f_name):
    
    
    """save grand median L2s between truths and obs,preds, and ukf
    
    - extract truths, obs, ukf predictions (preds), and forecasts
    - remove inactive agent measurements to prevent bias
    - calculate L2 distances between truth vs obs,preds and ukfs. 
    - take grand median of L2s for each estimator
    - store scalar grand medians in 3x1 numpy array (obs,forecasts,ukf)
    - save for depickle later
    
    Parameters
    -------
    
    instance : class
        ukf_ss `class` instance of finished ABM run for saving
        
    
    
    """

    truths = instance.truth_parser(instance)
    nan_array= instance.nan_array_parser(truths, instance.base_model)
    obs_key = instance.obs_key_parser()
    obs = instance.obs_parser(instance, True, truths, obs_key)
    preds = instance.preds_parser(instance, True, truths)
    forecasts = np.vstack(instance.forecasts)

    obs *= nan_array
    truths *= nan_array
    preds *= nan_array
    forecasts *= nan_array
    
    truths = truths[::instance.sample_rate,:]
    preds = preds[::instance.sample_rate,:]
    forecasts = forecasts[::instance.sample_rate,:]
    obs = obs[::instance.sample_rate,:]
    
    
    obs_error = np.nanmedian(np.nanmedian(L2s(truths,obs),axis=0))
    forecast_error = np.nanmedian(np.nanmedian(L2s(truths,forecasts),axis=0))
    ukf_error =  np.nanmedian(np.nanmedian(L2s(truths,preds),axis=0))
    
    mean_array = np.array([obs_error, forecast_error, ukf_error])
    print ("obs", "forecast", "ukf")
    print(mean_array)
    f_name = source + f_name
    print(f"saving to {f_name}")
    np.save(f_name, mean_array)
    
def ex0_main(n, noise, sampling_rate):
    
    
    """ main ex0 function 
    
    - assign population size, observation noise, and sampling rate
    - build ukf and model parameter dictionariues
    - build base model and init ukf_ss class
    - run stationsim with ukf filtering
    """
    
    model_params = configs.model_params
    ukf_params = configs.ukf_params
    
    model_params, ukf_params, base_model = benchmark_params(n, 
                                                            noise, 
                                                            sampling_rate, 
                                                            model_params,
                                                            ukf_params)
    print(model_params)
    print(ukf_params)
    
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    ex0_save(u, "../results/", ukf_params["file_name"])
  
#%%  
    
if __name__ == "__main__":
    n= 30
    noise = 2
    sampling_rate = 10    
    ex0_main(n,  noise, sampling_rate)
