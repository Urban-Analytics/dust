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


sys.path.append("../..")
from modules.ukf_fx import fx
from modules.ukf_plots import ukf_plots
import modules.default_ukf_gcs_configs as configs
from modules.ex3.rjmcmc_ukf import rjmcmc_ukf


sys.path.append("../../..")
sys.path.append("../../../..")
#from stationsim.stationsim_gcs_model import Model
from stationsim.stationsim_gcs_model import Model
from stationsim.ukf2 import *


def hx3(state, **hx_kwargs):
    return state


def ex3_pickle_name(n, jump_rate):
    """build name for pickle file
    
    Parameters
    ------
    n: float
        `n` population 
        
    Returns
    ------
    
    f_name : str
        return `f_name` file name to save pickle as
    """

    f_name = f"rjmcmc_ukf_gcs_agents_{n}_jump_rate{jump_rate}.pkl"
    return f_name


def obs_key_func(state, **hx_kwargs):
    """categorises agent observation type for a given time step
    
    0 - unobserved
    1 - aggregate
    2 - gps style observed
    
    For experiment 3 all agents are fully observed (2).
    
    Parameters
    --------
    state : array_like
        Desired state (e.g. full state of agent based model) on which we wish
        to test the obs_key on. Generally this is only used if the obs key
        is calculated dynamically 
        
    hx_kwargs : dict
        key word arguments from the observation function used to 
    
    Returns
    ------
    key : array_like
        (n_agents x 1) array indicating the type of obsesrvation for each agent.
        
    """
    n = hx_kwargs["pop_total"]
    key = 2 * np.ones(n)
    return key


def get_gates(base_model, get_gates_dict):
    """get model exit gate combination from stationsim model
    
    NOTE: this function varies from ABM to ABM. gcs and regular stationsim
    both have different functions to do this due to different attribute names.
    
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
    # gate centroids from model
    # convert centroid into intigers using gates_dict
    gates = []
    for i, agent in enumerate(base_model.agents):
        desire = base_model.gates_locations[agent.gate_out]
        gates.append(get_gates_dict[str(desire)])

    return gates


def set_gates(base_model, new_gates, set_gates_dict):
    """ Assign stationsim a certain combination of gates
    
    NOTE: this function varies from ABM to ABM. gcs and regular stationsim
    both have different functions to do this due to different attribute names.
    
    Parameters
    ------
    
    gates : list
        slist of agents exit `gates`
        
    Returns 
    ------
    
    stationsim : cls
        some `stationsim` class we wish to extract the gates of
    """
    # go through each agent and assign it a new exit gate
    for i, gate in enumerate(new_gates):
        gate_loc = set_gates_dict[gate]
        base_model.agents[i].gate_out = gate
        base_model.agents[i].loc_desire = gate_loc
    return base_model

def gates_dict(base_model):
        """assign each exit gate location an integer for a given model
        
        Parameters
        ----------
        base_model : cls
            some model `base_model` e.g. stationsim
        Returns
        -------
        gates_dict : dict
            `gates_dict` dictionary with intiger keys for each gate and a 
            2x1 numpy array value for each key giving the location.
        """
        
        gates_locations = base_model.gates_locations[-base_model.gates_out:]
        get_gates_dict = {}
        set_gates_dict = {}
        for i in range(gates_locations.shape[0]):
            gate_location = gates_locations[i,:]
            key = i
            get_gates_dict[str(gate_location)] = key
            set_gates_dict[key] = gate_location
            
        return get_gates_dict, set_gates_dict 
    
def rj_params(n, jump_rate, n_jumps, model_params, ukf_params):
    """build parameter dictionaries for ABM and rjukf

    Parameters
    ----------
    n : int
        `n` population total
    model_params, ukf_params : dict
        `model_params` dictionary of ABM parameters and `ukf_params`
        rk_ukf parameters

    Returns
    -------
    model_params, ukf_params : dict
        updated `model_params` dictionary of ABM parameters and `ukf_params`
        rk_ukf parameters
    base_model : cls
        `base_model` initiated agent based model of user choice.

    """
    # number of agents and intiate agent based model with model_params
    model_params["pop_total"] = n
    model_params["station"] = "Grand_Central"

    base_model = Model(**model_params)
    model_params["width"] = base_model.width
    model_params["height"] = base_model.height
    
    # force agents to sit at starting locations rather than 0.
    # makes assimilations not have terrible initial conditions.
    
    for agent in base_model.agents:    
        agent.location = agent.set_agent_location(agent.gate_in)

    model_params["exit_gates"] = base_model.gates_locations
    model_params["get_gates_dict"], model_params["set_gates_dict"] = gates_dict(base_model)

    ukf_params["vision_angle"] = np.pi/6
    ukf_params["jump_rate"] = jump_rate
    ukf_params["n_jumps"] = n_jumps
    
 
    #noise structures
    ukf_params["p"] = 1.0 * np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = 0.2 * np.eye(2 * n)
    ukf_params["r"] = 0.2 * np.eye(2 * n)#sensor noise
    
    # transition function kwargs and updater.
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"] = {"base_model" : base_model} 
    ukf_params["fx_kwargs_update"] = None

    #mesurement function kwargs
    ukf_params["hx"] = hx3
    ukf_params["hx_kwargs"] = {"pop_total" : n}
    
    # Observation Key Function
    ukf_params["obs_key_func"] = obs_key_func
    
    # Don't store results in ukf class. only store them in rjmcmc_UKF top layer
    # for space saving.
    ukf_params["record"] = False
    ukf_params["file_name"] =  ex3_pickle_name(n, jump_rate)
    
    return model_params, ukf_params, base_model


def error_plot(true_gates, estimated_gates, sample_rate):
    """plot difference between true and current estimate of exit gates 
    
    This uses a binary metric such that for each agent the distance is:
        distance = 0 if the estimate and true gates for an agent match
        distance = 1 if they do not match
    
    The sum of these ones and zeros is then taken as the error metric.
    A smaller metric implies more gates are correct.
    This metric is bound between 0 (all agents correct) and the population 
    total (no agents correct).
    
    Parameters
    ----------
    true_gates, estimated_gates : array_like
        arrays of the actual exit gates `true_gates` and predicted gates
        `estimated_gates` of the rjukf.
    Returns
    -------
    None.

    """
    distances = []
    true_gates = np.array(true_gates)
    estimated_gates = np.array(estimated_gates)
    for i in range(1, estimated_gates.shape[0]):
        distance = np.sum((true_gates - estimated_gates[i, :]) != 0)
        distances.append(distance)
    f = plt.figure()
    plt.plot(np.arange(len(distances))*sample_rate, distances)
    plt.xlabel("time")
    plt.ylabel("Error Between Estimated and True Gates. ")

def ex3_plots(instance, destination, prefix, save, animate):
    plts = ukf_plots(instance, destination, prefix, save, animate)

    truths = truth_parser(instance)
    nan_array= nan_array_parser(instance, truths, instance.base_model)
    #obs, obs_key = obs_parser(instance, True)
    obs_key = obs_key_parser(instance, True)
    preds = preds_parser(instance, True)
    #forecasts =  forecasts_parser(instance, True)
    gates = np.array(instance.estimated_gates).astype('float64')
    
    ukf_params = instance.ukf_params
    #forecasts = np.vstack(instance.forecasts)
    
    "remove agents not in model to avoid wierd plots"
    truths *= nan_array
    preds *= nan_array
    #forecasts*= nan_array
    #gates *= nan_array[::instance.sample_rate, 0::2]
    "indices for unobserved agents"

    subset = np.arange(truths.shape[1]//2)
    subset2 = np.repeat((2*subset), 2)
    subset2[1::2] +=1
    
    #turn on for subset plots 
    #if instance.model_params["random_seed"] == 8:
    #   subset = np.array([1, 5, 3])
    #   subset2 = np.repeat((2*subset), 2)
    #   subset2[1::2] +=1
        
    #plts.path_plots(obs, "Observed")
    plts.pair_frame(truths, preds, obs_key, 100, destination)

    error_plot(instance.true_gate, instance.estimated_gates, instance.sample_rate)
    
    plts.error_hist(truths[::instance.sample_rate,:], 
                    preds[::instance.sample_rate,:],"Observed Errors")
    
    plts.path_plots(preds[::instance.sample_rate, subset2], "Predicted")
    plts.path_plots(truths[:, subset2], "True")
    #plts.path_plots(forecasts[::instance.sample_rate, subset2], "Forecasts")
    plts.dual_path_plots(truths[::instance.sample_rate, subset2], preds[::instance.sample_rate, subset2],
                         "True and Predicted")

    plts.gate_choices(gates[:,], instance.sample_rate)    
  
    #split_gate_choices(gates[:, subset], instance.sample_rate)
    if animate:
        #plts.trajectories(truths, "plots/")
        plts.pair_frames(truths[::instance.sample_rate], 
                         preds[::instance.sample_rate],
                         obs_key[::instance.sample_rate],
                         truths[::instance.sample_rate].shape[0], "../../plots/")


def ex3_main(n, jump_rate, n_jumps, recall):
    """main function for applying rjukf to stationsim
    
    - load parameters
    - new run or load pickle
    - do plots

    Parameters
    ----------
    n : int
        number of agents `n`
    recall : bool
        `recall` a previously finished pickle of a run or 
        start from scratch.

    Returns
    -------
    rjmcmc_UKF : cls
        `rjmcmc_UKF` a class instance of the rjukf fitted to stationsim.

    """
    # load in some specific parameters
    model_params = configs.model_params
    #model_params["step_limit"] = 200
    #model_params["random_seed"] = 8
    ukf_params = configs.ukf_params

    # build rest of parameter dictionaries
    model_params, ukf_params, base_model = rj_params(n, jump_rate, n_jumps,
                                                     model_params,
                                                     ukf_params)
    # files for where to save plots and load pickles
    destination = "../../plots/"
    pickle_source = "../../pickles/"
    prefix = "rjmcmc_ukf_gcs_"
    # bools for saving plots, making animations, and saving class pickles
    save = True
    animate = False
    do_pickle = True

    # if not recalling, run the rjukf from scratch
    if not recall:
        rjmcmc_UKF = rjmcmc_ukf(model_params, ukf_params, base_model,
                                get_gates,
                                set_gates)

        rjmcmc_UKF.main()
        instance = rjmcmc_UKF.ukf_1
        pickle_main(ukf_params["file_name"], pickle_source, do_pickle, rjmcmc_UKF)
    # if recalling load a pickled run
    if recall:
        f_name = ex3_pickle_name(n, jump_rate)

        "try loading class_dicts first. If no dict then class instance."
        try:
            rjmcmc_UKF = pickle_main("dict_" + f_name, pickle_source, do_pickle)
        except:
            rjmcmc_UKF = pickle_main(f_name, pickle_source, do_pickle)

        model_params, ukf_params = rjmcmc_UKF.model_params, rjmcmc_UKF.ukf_params

    # plots
    instance = rjmcmc_UKF
    ex3_plots(instance, destination, prefix, save, animate)
    
    return rjmcmc_UKF


if __name__ == "__main__":
    n = 30
    jump_rate = 5
    n_jumps = 5
    recall = False
    rjmcmc_UKF = ex3_main(n, jump_rate, n_jumps, recall)
