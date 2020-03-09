# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")

from modules.ukf_fx import fx
from modules.poly_functions import poly_count, grid_poly
from modules.ukf_plots import ukf_plots
import modules.default_ukf_configs as configs

sys.path.append("../../stationsim")
sys.path.append("modules")
from stationsim_model import Model

import numpy as np


def obs_key_func(state, **obs_key_kwargs):
    """categorises agent observation type for a given time step
    0 - unobserved
    1 - aggregate
    2 - gps style observed


    """

    key = np.ones(obs_key_kwargs["pop_total"])

    return key


def aggregate_params(n, bin_size, model_params, rsukf_params):
    """update ukf_params with fx/hx and their parameters for experiment 2

    Parameters
    ------
    rsukf_params : dict

    Returns
    ------
    ukf_params : dict
    """

    model_params["pop_total"] = n
    base_model = Model(**model_params)

    rsukf_params["bin_size"] = bin_size
    rsukf_params["poly_list"] = grid_poly(model_params["width"],
                                        model_params["height"], ukf_params["bin_size"])

    rsukf_params["p"] = np.eye(2 * n)  # inital guess at state covariance
    rsukf_params["q"] = np.eye(2 * n)
    rsukf_params["r"] = np.eye(len(ukf_params["poly_list"]))  # sensor noise

    rsukf_params["fx"] = fx
    rsukf_params["fx_kwargs"] = {"base_model": base_model}
    rsukf_params["hx"] = hx2
    rsukf_params["hx_kwargs"] = {"poly_list": ukf_params["poly_list"]}
    rsukf_params["obs_key_func"] = obs_key_func
    rsukf_params["obs_key_kwargs"] = {"pop_total": n}

    rsukf_params["file_name"] = ex2_pickle_name(n, bin_size)

    return model_params, rsukf_params, base_model


def hx2(state, **hx_kwargs):
    """Convert each sigma point from noisy gps positions into actual measurements

    - take some desired state vector of all agent positions
    - count how many agents are in each of a list of closed polygons using poly_count
    - return these counts as the measured state equivalent

    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted

    **hx_args
        generic hx kwargs
    Returns
    ------
    counts : array_like
        forecasted `counts` of how many agents in each square to
        compare with actual counts
    """
    poly_list = hx_kwargs["poly_list"]

    counts = poly_count(poly_list, state)

    if np.sum(counts) > 0:
        counts /= np.sum(counts)
    return counts


def ex2_pickle_name(n, bin_size):
    """build name for pickle file

    Parameters
    ------
    n, bin_size : float
        `n` population and `bin_size` aggregate square size

    Returns
    ------

    f_name : str
        return `f_name` file name to save pickle as
    """

    f_name = f"ex2_rsukf_agents_{n}_bin_{bin_size}.pkl"

    return f_name

if __main__ == "__main__":
    model_params = configs.model_params

    rsukf_params = {
    "n_max" : 50
    }

    model_params, rsukf_params, base_model = aggregate_params(n, bin_size, model_params, rsukf_params)

    rsu = rsukf_ss()
