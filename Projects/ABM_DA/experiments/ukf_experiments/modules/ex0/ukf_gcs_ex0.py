#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment 0 Grand Central module.

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

# vanilla modules
import sys
import os
import numpy as np

# modules for experiment
sys.path.append("../..")
from modules.ukf_fx import fx2
from modules.ukf_plots import L2s
import modules.default_ukf_gcs_configs as configs
from modules.ex0.ukf_ex0 import hx0, obs_key_0, ex0_save

# modules for model/base ukf
sys.path.append("../../..")
sys.path.append("../../../..")
from stationsim.ukf2 import *
from stationsim.stationsim_gcs_model import Model


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
    ukf_params["q"] = np.eye(2 * n)
    #sensor noise
    ukf_params["r"] = np.eye(2 * n)
    
    ukf_params["fx"] = fx2
    ukf_params["fx_kwargs"] = {}
    ukf_params["fx_kwargs_update"] = None
    
    ukf_params["hx"] = hx0    
    ukf_params["hx_kwargs"] = {}
    ukf_params["obs_key_func"] = obs_key_0
    
    ukf_params["record"] = True
    ukf_params["file_name"] = f"config_agents_{n}_rate_{sample_rate}_noise_{noise}"
    
    return model_params, ukf_params, base_model
    
def ex0_main(n, noise, sampling_rate):
    """ main ex0 function 
    
    - assign population size, observation noise, and sampling rate
    - build ukf and model parameter dictionariues
    - build base model and init ukf_ss class
    - run stationsim with ukf filtering
    
    Parameters
    --------
    n, noise, sampling_rate: float
        population `n` observation `noise` and `sampling_rate`
        
    Returns
    ------
    u : cls
        `u` finished ukf instance.
    """
    # model and filter parameters
    model_params = configs.model_params
    #model_params["step_limit"] = 300
    ukf_params = configs.ukf_params
    # update param dictionaries
    model_params, ukf_params, base_model = benchmark_params(n, 
                                                            noise, 
                                                            sampling_rate, 
                                                            model_params,
                                                            ukf_params)
    print(model_params)
    print(ukf_params)
    # run ukf filter
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    ex0_save(u, "../../results/", ukf_params["file_name"])
    return u
  
#%%  
    
if __name__ == "__main__":
    n= 5
    noise = 1
    sampling_rate = 1
    u =  ex0_main(n,  noise, sampling_rate)
