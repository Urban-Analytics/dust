#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:25:37 2019

@author: rob

THIS IS JUST A PROOF OF CONCEPT FOR NOW. IGNORE IT

"""

import sys
sys.path.append("../..")
from stationsim.ukf2 import fx
from ukf_ex1 import omission_index

import numpy as np
from math import floor

def hx3(state, model_params, ukf_params):


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
    ukf_params["index"], ukf_params["index2"] = omission_index(model_params["pop_total"], ukf_params["sample_size"])

    obs_state = state[ukf_params["index2"]]
    
    return obs_state   
    
def lateral_params(model_params, ukf_params):
    
    
    """update ukf_params with fx/hx for lateral partial omission.
    I.E every agent has some observations over time
    (potential experiment 3)
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    n = model_params["pop_total"]
    ukf_params["prop"] = 0.5
    ukf_params["sample_size"]= floor(n * ukf_params["prop"])

    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * ukf_params["sample_size"])#sensor noise
    
   
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx3
    
    def obs_key_func(state,ukf_params):
        """which agents are observed"""
        
        key = np.zeros(model_params["pop_total"])
        key[ukf_params["index"]] +=2
        return key
    
    ukf_params["obs_key_func"] = obs_key_func
        
    return ukf_params