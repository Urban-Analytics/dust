#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:57:18 2020

@author: medrclaa
"""
import sys
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from rjmcmc_ukf import rjmcmc_ukf

sys.path.append("..")
sys.path.append("../..")
import modules.default_ukf_configs as configs
from modules.ukf_fx import fx2
from modules.ukf_plots import ukf_plots

sys.path.append("../../../..")
sys.path.append("../../..")
from stationsim.stationsim_model import Model
from stationsim.ukf2 import pickle_main


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

def get_gates(base_model, get_gates_dict):
    """get model exit gate combination from stationsim model
    
    Parameters
    ------
    stationsim : cls
        some `stationsim` class we wish to extract the gates of
    Returns
    -------
    gates : int
        Intinger incidating which of the exit `gates` an agent is heading 
        to.
    """
    #gate centroids from model
    gates = [agent.loc_desire for agent in base_model.agents]
    #convert centroid into intigers using gates_dict
    for i, desire in enumerate(gates):
        gates[i] = get_gates_dict[str(desire)]
    return gates

def set_gates(base_model, new_gates, set_gates_dict):
    """ assign stationsim a certain combination of gates
    
    Parameters
    ------
    
    gates : list
        slist of agents exit `gates`
        
    Returns 
    ------
    
    stationsim : cls
        some `stationsim` class we wish to extract the gates of
    """

    for i, gate in enumerate(new_gates):
        new_gate = set_gates_dict[gate]
        base_model.agents[i].loc_desire = new_gate
    return base_model
    
def rj_params(n, jump_rate, n_jumps, model_params, ukf_params):
    
    model_params["pop_total"] = n
    model_params["gates_out"] = 3

    base_model = Model(**model_params)
    model_params["exit_gates"] =  base_model.gates_locations[-model_params["gates_out"]:]
    
    ukf_params["vision_angle"] = np.pi/8
    ukf_params["jump_rate"] = jump_rate
    ukf_params["n_jumps"] = n_jumps
    
    ukf_params["p"] = 0.1 * np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = 0.01 * np.eye(2 * n)
    ukf_params["r"] = 0.01 * np.eye(2 * n)#sensor noise
    
    ukf_params["fx"] = fx2
    ukf_params["fx_kwargs"] = {"base_model":base_model} 
    ukf_params["hx"] = hx3
    ukf_params["hx_kwargs"] = {"pop_total" : n}
    
    ukf_params["obs_key_func"] = obs_key_func
    
    ukf_params["file_name"] =  ex3_pickle_name(n)
    return model_params, ukf_params, base_model

def error_plot(true_gates, estimated_gates):
    distances = []
    true_gates = np.array(true_gates)
    estimated_gates = np.array(estimated_gates)
    for i in range(1, estimated_gates.shape[0]):
        distance = np.sum((true_gates - estimated_gates[i,:]) != 0)
        distances.append(distance)
    f = plt.figure()
    plt.plot(distances)
    plt.xlabel("time")
    plt.ylabel("Error Between Estimated and True Gates. ")

def ex3_main(n, jump_rate, n_jumps, recall):
    ukf_params = configs.ukf_params
    model_params = configs.model_params
    destination = "../../plots/"
    pickle_source = "../../pickles/"
    prefix = "rjmcmc_ukf_"
    save = True
    animate = True
    do_pickle = True
                       
    model_params, ukf_params, base_model = rj_params(n, jump_rate, n_jumps, model_params,
                                                         ukf_params)       
       
               
    if not recall:
        rjmcmc_UKF= rjmcmc_ukf(model_params, ukf_params, base_model,
                                                             get_gates,
                                                             set_gates)
        rjmcmc_UKF.main()
    
        instance = rjmcmc_UKF.ukf_1
        pickle_main(ukf_params["file_name"],pickle_source, do_pickle, rjmcmc_UKF)
    if recall:
        f_name = ex3_pickle_name(n)
        
        "try loading class_dicts first. If no dict then class instance."
        try:
            rjmcmc_UKF  = pickle_main("dict_" + f_name, pickle_source, do_pickle)
        except:
            rjmcmc_UKF  = pickle_main(f_name, pickle_source, do_pickle)
            
        model_params, ukf_params = rjmcmc_UKF.model_params, rjmcmc_UKF.ukf_params
            
    instance = rjmcmc_UKF.ukf_1
    plts = ukf_plots(instance, destination, prefix, save, animate)

    truths = instance.truth_parser(instance)
    nan_array= instance.nan_array_parser(instance, truths, instance.base_model)
    obs, obs_key = instance.obs_parser(instance, True)
    preds = instance.preds_parser(instance, True)
    forecasts =  instance.forecasts_parser(instance, True)
    
    ukf_params = instance.ukf_params
    #forecasts = np.vstack(instance.forecasts)
    
    "remove agents not in model to avoid wierd plots"
    truths *= nan_array
    preds *= nan_array
    #forecasts*= nan_array
    
    "indices for unobserved agents"

    #plts.path_plots(obs, "Observed")
    plts.pair_frame(truths, preds, obs_key, 10, destination)

    error_plot(rjmcmc_UKF.true_gate, rjmcmc_UKF.estimated_gates)
    
    plts.error_hist(truths[::instance.sample_rate,:], 
                    preds[::instance.sample_rate,:],"Observed Errors")
    
    plts.path_plots(preds[::instance.sample_rate], "Predicted")
    plts.path_plots(truths, "True")
    plts.path_plots(preds[::instance.sample_rate], "Forecasts")

    if animate:
        #plts.trajectories(truths, "plots/")
        plts.pair_frames(truths[::instance.sample_rate], 
                         preds[::instance.sample_rate],
                         obs_key[::instance.sample_rate],
                         truths[::instance.sample_rate].shape[0], "../../plots/")
    
    return rjmcmc_UKF

if __name__ == "__main__":
    
    n = 50
    jump_rate = 10
    n_jumps = 5
    recall = False
    rjmcmc_UKF = ex3_main(n, jump_rate, n_jumps, recall)
    
    
    