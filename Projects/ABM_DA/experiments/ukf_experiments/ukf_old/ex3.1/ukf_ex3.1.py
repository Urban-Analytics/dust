#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:16:33 2019

@author: rob


"""
import sys
import os
from shapely.geometry import LineString, Point
import numpy as np
from math import floor
import multiprocessing
import logging
from copy import deepcopy
import datetime

sys.path.append("..")
sys.path.append("../../../../stationsim")

from ukf_fx import fx3
from ukf_plots import ukf_plots
import default_ukf_configs as configs
from sensors import generate_Camera_Rect

from ukf2 import ukf_ss, pickle_main
from stationsim_model import Model

sys.path.append("../ex3")
from ex3.ukf_ex3 import get_gates, set_gates, gates_dict
    

def nearest_gates(intersections, gates_locations, get_gates_dict):
    """ where is the nearest gate to the intersection.
    
    Parameters
    ----------
    intersections : TYPE
        DESCRIPTION.
    gates_locations : TYPE
        DESCRIPTION.
    get_gates_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    gates = []
    for i in range(int(intersections.shape[1])):
        point = intersections[:, i]
        res = point - gates_locations
        dist = np.linalg.norm(point-gates_locations, axis = 1)
        gate = np.argmin(dist)
        gate_location = gates_locations[gate, :]
        gate_id = get_gates_dict[str(gate_location)]
        try:
            gates.append(gate)
        except:
            print(gate)
            print(intersections)
    # find nearest point
    # convert to gate id intigers
    return gates
    
    
def fx3_kwargs_updater(base_model, fx_kwargs):
    """update kwargs

    Returns
    -------
    None.

    """
    fx_kwargs["old_state"] = fx_kwargs["state"]
    fx_kwargs["state"] = base_model.get_state("location")
    
    intersections = linear_projection(fx_kwargs["state"],
                                      fx_kwargs["old_state"],
                                      fx_kwargs["boundary"])
    fx_kwargs["gates"] = nearest_gates(intersections,
                                       fx_kwargs["exit_gates"],
                                       fx_kwargs["get_gates_dict"])
    
    return fx_kwargs


def linear_projection(state, old_state, boundary):
    """predict where an agent with a linear path will leave stationsim

    Parameters
    ----------
    state , old_state : array_like
        previous `old_state` and current `state` locations of agents to make a linear projection
        for
    boundary : TYPE
        DESCRIPTION.

    Returns
    -------
    intersection : TYPE
        DESCRIPTION.

    """
    if len(state) != len(old_state):
        new_state = np.zeros(len(state))
        new_state[0::2] = state[0::4]
        new_state[1::2] = state[1::4]
        state = new_state
    res = state - old_state
    #split into x and y coordinates for arctan2
    x = res[0::2]
    y = res[1::2]
    #arctan2 is a special function for converting xy to polar. 
    #NOT THE SAME AS REGULAR ARCTAN
    angles = np.arctan2(y, x)
    dist = 1000
    intersections = []
    for i in range(int(len(state)/2)):
        p = state[(2*i):((2*i)+2)]
        line = LineString([Point(p), Point(p[0] + (dist * np.cos(angles[i])),
                                           p[1] + dist * np.sin(angles[i]))])
        intersect = boundary.exterior.intersection(line)
        if type(intersect) == Point:
            intersect = intersect.coords.xy
        else: 
            # !! change this to keep gate same later.
            intersect = intersect[-1].coords.xy  
            
        intersections.append(intersect)                        
    return np.hstack(intersections)

    
def hx3_1(*args, **hx_kwargs):
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
    return args[0]

def obs_key_func(state, **hx_kwargs):
    
    
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
    key = np.ones(state.shape)
    
    return key

def ex3_params(n, model_params, ukf_params):
    
    
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
    model_params["station"] = None
    base_model = Model(**model_params)
    model_params["exit_gates"] =  base_model.gates_locations[-model_params["gates_out"]:]
    model_params["get_gates_dict"], model_params["set_gates_dict"] = gates_dict(base_model)

    width = model_params["width"]
    height = model_params["height"]
    ukf_params["boundary"] = generate_Camera_Rect(np.array([0, 0]), 
                                                  np.array([0, height]),
                                                  np.array([width, height]), 
                                                  np.array([width, 0]))    
    ukf_params["p"] = np.eye(1 * 2 * n) #inital guess at state covariance
    ukf_params["q"] = 0.05 * np.eye(1 * 2 * n)
    ukf_params["r"] = 0.01 * np.eye(1 * 2* n)#sensor noise

    ukf_params["x0"] = base_model.get_state("location")
    
    ukf_params["fx"] = fx3
    ukf_params["fx_kwargs"] = {"state" : ukf_params["x0"],
                               "boundary" : ukf_params["boundary"],
                               "get_gates_dict" : model_params["get_gates_dict"],
                               "set_gates_dict" : model_params["set_gates_dict"],
                               "set_gates" : set_gates,
                               "exit_gates" : model_params["exit_gates"]} 
    
    ukf_params["fx_kwargs_update"] = fx3_kwargs_updater

    ukf_params["hx"] = hx3_1
    ukf_params["hx_kwargs"] = {}
    
    ukf_params["obs_key_func"] = obs_key_func    
    ukf_params["file_name"] =  ex3_pickle_name(n)
        
    return model_params, ukf_params, base_model

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
    
    f_name = f"ueg_ukf_agents_{n}.pkl"
    return f_name


def ex3_plots(instance, destination, prefix, save, animate):
    
    
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
    ukf_params = instance.ukf_params
    
    truths = instance.truth_parser(instance)
    #obs, obs_key = instance.obs_parser(instance, True)
    preds = instance.preds_parser(instance, True)
    forecasts =  instance.forecasts_parser(instance, True)
    nan_array= instance.nan_array_parser(instance, truths, instance.base_model)
    
    "remove agents not in model to avoid wierd plots"
    #obs *= nan_array
    truths *= nan_array
    preds *= nan_array
    forecasts*= nan_array
    
    "indices for unobserved agents"
    #plts.pair_frame(truths, preds, obs_key, 10, destination)
    plts.error_hist(truths[::instance.sample_rate, :], 
                    preds[::instance.sample_rate, :],"Observed Errors")

    #plts.path_plots(obs, "Observed")
    plts.path_plots(preds[::instance.sample_rate, :], "Predicted")
    plts.path_plots(truths, "True")

    if animate:
        #plts.trajectories(truths, "plots/")
        plts.pair_frames(truths, preds, obs_key,
                         truths.shape[0], "../../plots/")

        


def ex3_main(n, recall, do_pickle, source, destination):
    
    
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
        model_params = configs.model_params
        ukf_params = configs.ukf_params
        model_params, ukf_params, base_model = ex3_params(n,
                                                               model_params, 
                                                               ukf_params)
        base_models = []
        for i in range(int((4 * n) + 1)):
            base_models.append(deepcopy(base_model))
        
        print(f"Population: {n}")        
        u = ukf_ss(model_params, ukf_params, base_model, base_models)
        u.main()
        pickle_main(ukf_params["file_name"],pickle_source, do_pickle, u)
    
    else:
        "if recalling, load a pickle."
        f_name = ex3_pickle_name(n)
        
        "try loading class_dicts first. If no dict then class instance."
        try:
            u  = pickle_main("dict_" + f_name, source, do_pickle)
        except:
            u  = pickle_main(f_name, source, do_pickle)
 
        model_params, ukf_params = u.model_params, u.ukf_params
    
    ex3_plots(u, destination, "ukf_", True, False)

    return u
    
if __name__ == "__main__":
    recall = False #recall previous run
    do_pickle = True #pickle new run
    pickle_source = "../../pickles/" #where to load/save pickles from
    destination = "../../plots/"
    n = 5 #population size

    u = ex3_main(n, recall, do_pickle, pickle_source, destination)