#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 15:12:16 2020

@author: medrclaa

Experiment modules for UKF experiment where observations come from within the view
of cone like cameras only
"""

import sys
import os

"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")

from modules.ukf_fx import fx
from modules.sensors import camera_Sensor, boundary_Polygon
from modules.ukf_plots import ukf_plots
import modules.default_ukf_configs as configs
import numpy as np
sys.path.append("../../stationsim")
from ukf2 import ukf_ss, pickle_main
from stationsim_model import Model


def obs_key_func(state,**obs_key_kwargs):
    
    """categorises agent observation type for a given time step
    0 - unobserved
    1 - aggregate
    2 - gps style observed
    
    For ex4 this assumes observed if within a camera and 0 otherwise
    
    Parameters
    ------
    state : array_like
        true ABM `state` as a ravelled vector where every 2 entries represent
        an xy coordinate.
    
    obs_key_kwargs : kwargs
        generic `obs_key_kwargs` for the obs_key_func. Varies by experiment.
        
    Returns
    ------
    
    """
    pass

def cone_params(n, cameras, model_params, ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 2
    
    Parameters
    ------
    n : int
        `n` population total
        
    cameras: list
        list of camera_Sensor sensor objects.
    
    model_params, ukf_params : dict
        
    Returns
    ------
    model_params, ukf_params : dict
    """
    
    model_params["pop_total"] = n
    base_model = Model(**model_params)
    
    ukf_params["cameras"] = cameras
        
    ukf_params["p"] = np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2*n) #process noise
    "sensor noise here dynamically updated depending on how many agents in the cameras."
    ukf_params["r"] = np.eye(2*n)#sensor noise 
    
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"]  = {"base_model" : base_model}
    ukf_params["hx"] = hx4
    ukf_params["hx_kwargs"] = {"cameras": cameras}
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["obs_key_kwargs"]  = {"pop_total" : n}

    ukf_params["file_name"] = """ex4_pickle_name(n, cameras)    """
    
    
    return model_params, ukf_params, base_model


def hx4(state, **hx_kwargs):
    
    cameras = hx_kwargs["cameras"]
    
    index = []
    for camera in cameras:
        index += camera.observe(state)
        
    index = np.array(index)
    index2 = np.repeat(2*index,2) 
    index2[1::2] = 2*index + 1
    
    return state[index2]

def ex4_pickle_name(n, cameras):
    pass
    

if __name__ == "__main__":
    
    model_params = configs.model_params
    ukf_params = configs.ukf_params
    n = 5
    
    camera_poles = [np.array([10,0]), np.array([150,0])]
    camera_centres = [np.array([0,100]), np.array([150,200])]
    camera_arcs = [0.125]*2
    boundary = boundary_Polygon(model_params["width"], model_params["height"])
    
    cameras = []
    for i in range(len(camera_poles)):
        pole = camera_poles[i]
        centre = camera_centres[i]
        arc = camera_arcs[i]
        cameras.append(camera_Sensor(pole, centre, arc, boundary))
    
    model_params, ukf_params , base_model = cone_params(n, cameras, model_params, ukf_params)
    
    for _ in range(5):
        base_model.step()    
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    
    
