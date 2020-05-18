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
import numpy as np

from ukf_fx import fx
from sensors import camera_Sensor, generate_Camera_Rect, generate_Camera_Cone
from ukf_plots import ukf_plots
import default_ukf_configs as configs

sys.path.append("../../../stationsim")
from ukf2 import ukf_ss, pickle_main
from stationsim_model import Model


def obs_key_func(state, **hx_kwargs):
    """categorises agent observation types for a given time step
    
    For this experiment an agent is observed (2)
    if within a camera and (0) otherwise.
    
    Parameters
    ------
    state : array_like
        true ABM `state` as a ravelled vector where every 2 entries represent
        an xy coordinate. This is a default parameter for all obs_key_functions
        and isnt used in this case. This is due to the obs_key depending
        on the state of the true ABM rather than the state of the sigma point.
    
    hx_kwargs : kwargs
        generalised `hx_kwargs` for the obs_key_func. Varies by experiment.
        
    Returns
    ------
    key `array_like`
        row of 0s and 2s indicating each agents observation type
    """
    
    n = hx_kwargs["n"]
    index = hx_kwargs["index"]
    
    key = np.zeros(n)
    if index.shape[0]>0:
        key[index] +=2
    return key

def cone_params(n, cameras, model_params, ukf_params):
    """update ukf_params with fx/hx and their parameters for experiment 4
    
    -add ground truth stationsim model base_model
    Parameters
    ------
    n : int
        `n` population total
        
    cameras: list
        list of camera_Sensor objects. Each camera has a polygon it observes.
    
    model_params, ukf_params : `dict`
        default stationsim `model_params` and ukf `ukf_params` parameter 
        dictionaries to be updated for the experiment to run.
    Returns
    ------
    model_params, ukf_params : `dict`
        updated default stationsim `model_params` and ukf `ukf_params` 
        dictionaries
    base_model : `class`
        initiated stationsim model `base_model` used as the ground truth.
    """
    #stationsim truth model
    model_params["pop_total"] = n
    base_model = Model(**model_params)
    #cameras
    ukf_params["cameras"] = cameras
    #noise structures
    ukf_params["p"] = 0.1 * np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = 0.01* np.eye(2*n) #process noise
    "sensor noise here dynamically updated depending on how many agents in the cameras."
    ukf_params["r"] = 0.01* np.eye(2*n)#sensor noise 
    #kalman functions
    ukf_params["fx"] = fx
    ukf_params["fx_kwargs"]  = {"base_model" : base_model}
    ukf_params["hx"] = hx4
    ukf_params["hx_kwargs"] = {"cameras": cameras, "n": n,}
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["hx_kwargs_update_function"] = hx4_kwargs_updater
    #pickle file name
    ukf_params["file_name"] = ex4_pickle_name(n)
    return model_params, ukf_params, base_model
    
def hx4_kwargs_updater(state, *hx_update_args, **hx_kwargs):
    """update the hx_kwargs dictionary with a new observed index for the new obsevation state.

    list whether or not each agent is inside a cameras polygon.

    Parameters
    ----------
    state: 'array_like'
    some true noisy observed `state` of agent positions.
    hx_kwargs: `dict`
    dictionary of hx keyword arguements to update
    hx_update_args: `list`
    additional arguements for updating hx_kwargs

    Returns
    -------
    hx_kwargs: `dict`
    dictionary of updated hx keyword arguements
    """

    cameras = hx_kwargs["cameras"]

    index = []
    for camera in cameras:
        index += camera.observe(state)

    index = np.array(index)
    index2 = np.repeat(2 * index, 2)
    index2[1::2] = 2 * index + 1

    #new_hx_kwargs = {
    #    "index":  index,
    #"index2": index2,
    # }

    hx_kwargs["index"] = index
    hx_kwargs["index2"] = index2

    return hx_kwargs

def hx4(state, **hx_kwargs):
    index2 = hx_kwargs["index2"]
    if index2.shape[0] == 0:
        new_state = []
    else:
        new_state = state[index2]
    return new_state

def ex4_pickle_name(n):
    f_name = f"rect_agents_{n}.pkl"
    return f_name
    

def ex4_plots(instance, destination, prefix, save, animate):
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
    destination, prefix : str
        `destination` where to save the plot to e.g. "" for this directory or 
        "ukf_results/ for ukf_results"
        `prefix` puts some prefix in front of various picture file names.
        e.g. "ukf_" or "agg_ukf_" so you can differentiate them and not overwrite
        them
    save , animate : bool
        `save` plots or `animate` whole model run? Animations can take a long
        time and a lot of ram for larger models.
    
    """
    
    plts = ukf_plots(instance, destination, prefix, save, animate)

    truths = instance.truth_parser(instance)
    preds = instance.preds_parser(instance, True, truths)
    forecasts = np.vstack(instance.forecasts)
    obs_key = instance.obs_key_parser()
    sample_rate = ukf_params["sample_rate"]
    
    #obs = instance.obs_parser(instance, True, truths, obs_key)
    #raw_obs = instance.obs_key_parser(instance, False)

    
    "indices for unobserved agents"
    plts.pair_frame(truths, preds, obs_key, 10, destination)
    plts.error_hist(truths, preds, "camera error hist")
    #plts.path_plots(obs, "Observed")
    "remove nan rows to stop plot clipping"
    plts.path_plots(preds[::instance.sample_rate], "Predicted")
    plts.path_plots(truths, "True")

    if animate:
        #plts.trajectories(truths, "plots/")
        plts.pair_frames(truths, forecasts, obs_key,
                         truths.shape[0], "../plots/")


def cone_main(width, height):
    """
    

    Parameters
    ----------
    width, height : float
        `width` and `height` of corridor
        
    Returns
    -------
    cameras : list
        list of camera_Sensor objects with cone observations
    """
    # where cameras are
    camera_poles = [np.array([20,0]), np.array([30,0])]
    # where cameras are facing
    camera_centres = [np.array([40,100]), np.array([10,100])]
    #how wide are the cameras. 
    camera_arcs = [1/10]*2
    boundary = generate_Camera_Rect(np.array([0, 0]), 
                                np.array([0, height]),
                                np.array([width, height]), 
                                np.array([width, 0]))
    
    cameras = []
    for i in range(len(camera_poles)):
        polygon = generate_Camera_Cone(camera_poles[i], camera_centres[i], 
                                    camera_arcs[i], boundary)
        cameras.append(camera_Sensor(polygon))
    return cameras

def generate_Rect_Coords(n_poly, poly_width, width, height):
    """generate coordinates of evenly spaced rectangles over a corridor
    
    Parameters
    ----------
    n_poly : `int`
        DESCRIPTION.
    poly_width : `float`
        how wide is each rectangle
    width, height : `float
        width and height of corridor in which rectangles are generated
    Returns
    -------
    bls, tls, trs, brs : `array_like`
        bottom left `bls` top left `tls` top rights `trs` 
        and bottom rights `brs`. Each are nx2 numpy arrays of coordinates
        indicating the corresponding corner of each rectangle.
    """
    if n_poly*poly_width > width:
        print("warning polygons overlap. may effect results.")
        
    #generate left and right edge x coordinates of all rectangles
    left_xs = np.linspace(0, width-poly_width,n_poly)
    right_xs = left_xs+poly_width
    #generate top and bottom edge y coordinates of all rectangles
    down_ys = np.zeros(n_poly)
    up_ys = down_ys + height
    #stack these 4 columns into 4x2 coordinates.
    #each row the four corners. e.g. bls is the bottom left corner of a rectangle.
    bls = np.vstack([left_xs, down_ys]).T
    tls = np.vstack([left_xs, up_ys]).T
    trs = np.vstack([right_xs, up_ys]).T
    brs = np.vstack([right_xs, down_ys]).T
    return bls, tls, trs, brs
    
def square_main(n_poly, poly_width, width, height):
    """Main function for assembling rectangle cameras evenly over a corridor
    
    Parameters
    ----------
    n_poly, poly_width : int
        How many rectangles `n_poly` and how wide `poly_width`
    width, height: float
        `width` and `height` of rectangular corridor

    Returns
    -------
    cameras : list
        list of camera_Sensor objects with rectangular observed polygons 
        according to n_poly and poly_width.
    """

    #generate rectangle coordinates based on n_poly and poly_width
    bls, tls, trs, brs = generate_Rect_Coords(n_poly, poly_width,
                                              width, height)
    boundary = generate_Camera_Rect(np.array([0, 0]), 
                                np.array([0, height]),
                                np.array([width, height]), 
                                np.array([width, 0]))
    
    cameras = []
    #loop over number of polygons.
    #extract the ith corner from each list, make a polygon
    #and init a camera with it
    for i in range(len(bls)):
        polygon = generate_Camera_Rect(bls[i], tls[i], 
                                    trs[i], brs[i], boundary)
        cameras.append(camera_Sensor(polygon))
    return cameras

if __name__ == "__main__":
    
    #load default params
    model_params = configs.model_params
    ukf_params = configs.ukf_params
    width = model_params["width"]
    height = model_params["height"]
    #change these
    #population number of rectangles, and rectangle width
    n = 10
    n_poly = 5
    poly_width = 12
    
    #generate cameras
    cameras = cone_main(width, height)
    #cameras = square_main(n_poly, poly_width, width, height)
    #generate ukf parameters based on cameras and population total
    model_params, ukf_params , base_model = cone_params(n, cameras, 
                                                        model_params, ukf_params)
    #initiatie and run ukf class
    u = ukf_ss(model_params,ukf_params,base_model)
    u.main()
    
    ex4_plots(u, "../plots", "partial", True, True)
    
