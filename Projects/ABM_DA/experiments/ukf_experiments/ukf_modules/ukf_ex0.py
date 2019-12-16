#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:49:27 2019

Experiment 0 module. Similar ideas to experiment 1 but keep prop fixed at 1
and vary noise/sample_rate

@author: rob
"""

import sys

"import required modules"
from ukf_fx import fx
from ukf_plots import L2s
from default_ukf_configs import model_params,ukf_params

sys.path.append("../../../stationsim")
from ukf2 import ukf_ss
from stationsim_model import Model

import numpy as np

def ex0_params(n, noise, sample_rate, model_params = model_params, ukf_params=ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 1
    
    Parameters
    ------
    n, noise, sample_rate : float
        `n` population, additive `noise`, and `sample_rate` sampling rate
        
        ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    model_params["pop_total"] = n
    ukf_params["noise"] = noise
    ukf_params["sample_rate"] = sample_rate
    
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * n)#sensor noise
    
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx0    
    ukf_params["obs_key_func"] = None
    
    ukf_params["file_name"] = f"config_agents_{n}_rate_{sample_rate}_noise_{noise}"
    
    return model_params, ukf_params

def hx0(state, model_params, ukf_params):
    
    
    "do nothing for hx. two states are the same."
    return state


def ex0_save(instance,source,f_name):
    
    
    """save grand median L2s between truths and obs,preds, and ukf"""
    
    obs, preds, truths,nan_array= instance.data_parser()
    forecasts = np.vstack(instance.forecasts)

    obs *= nan_array[::instance.sample_rate]
    truths *= nan_array
    preds *= nan_array
    forecasts *= nan_array
    
    truths = truths[::instance.sample_rate,:]
    preds = preds[::instance.sample_rate,:]
    forecasts = forecasts[::instance.sample_rate,:]

    obs_error = np.nanmedian(np.nanmedian(L2s(truths,obs),axis=0))
    forecast_error = np.nanmedian(np.nanmedian(L2s(truths,forecasts),axis=0))
    ukf_error =  np.nanmedian(np.nanmedian(L2s(truths,preds),axis=0))
    
    mean_array = np.array([obs_error, forecast_error, ukf_error])
    print ("obs", "forecast", "ukf")
    print(mean_array)
    np.save(source + f_name, mean_array)
    
def ex0_main():
    
    
    """ main ex0 function"""
    n= 10
    noise = 0.5
    sampling_rate = 5
    model_params, ukf_params = ex0_params(n, noise, sampling_rate)
    print(model_params)
    print(ukf_params)
    
    base_model = Model(**model_params)
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    ex0_save(u, "../ukf_results/", ukf_params["file_name"])
  
#%%  
    
if __name__ == "__main__":
    ex0_main()
    
    
    
    