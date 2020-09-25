#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 10:48:59 2020

@author: rob
"""
import sys
import numpy as np
from depickle import grand_plots, main
import pandas as pd

#sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")
# old ukf import here to keep depickle happy
# !!to do. convert pickles to lighters versions without scipy and old ukf issues.
sys.path.append("../ukf_old")

from stationsim.ukf2 import pickle_main, truth_parser, preds_parser

from modules.ukf_plots import L2s as L2_parser
from modules.ukf_fx import HiddenPrints


def ex1_restrict(distances, instance, *args):
    """split L2s for separate observed unobserved plots.

    parameters
    ---------
    
    distances : array_like
        `distances` array of L2 distances between truths and predictions for 
        splitting into two.
    instance : cls ukf instance for some parameters.
    
    *args: args
        any arguments needed for splitting up distances. Typically this
        just a list with the string either observed or unobserved.
        
    Returns
    ------
    distances : array_like
        `distances` subset of original distances for aggregating
    """
    try:
        observed = args[0]["observed"]
    except:
        observed = args["observed"]
    index = instance.index

    if observed:
        distances = distances[:, index]
    elif not observed:
        distances = np.delete(distances, index, axis=1)

    return distances

def ex1_grand(source, destination, recall):
    """ main function for parsing experiment 1 results into summary plots
    

    Parameters
    ----------
    source, destination : str
        `source` of experiment pickles and `destination` of the resulting plots
    recall : bool
        `recall` previously aggregated pickles from a np/pandas frame for speed?

    Returns
    -------
    None.

    """
    prefix = "ukf*"
    
    file_params = {
        "agents":  [10, 20, 30],
        "prop": [0.25, 0.5, 0.75, int(1)],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source + prefix,
        "destination": destination,
    }
    
    "plot observed/unobserved plots"
    obs_bools = [True, False]
    obs_titles = ["Observed", "Unobserved"]
    for i in range(len(obs_bools)):
        "initialise plot for observed/unobserved agents"
        g_plts = grand_plots(file_params, True, restrict=ex1_restrict,
                             observed=obs_bools[i])
        "make dictionary"
        if not recall:
            L2 = g_plts.data_extractor()
            "make pandas dataframe for seaborn"
            error_frame = g_plts.data_framer(L2)
            "make choropleth numpy array"
            error_array = g_plts.choropleth_array(error_frame)
            np.save(source + obs_titles[i] + "_distance_error_array", error_array)
            error_frame.to_pickle(source + obs_titles[i] + "_distance_error_array.pkl")
        else:
            error_frame = pd.read_pickle(source + obs_titles[i] + "_distance_error_array.pkl")
            error_array = np.load(source + obs_titles[i] + "_distance_error_array.npy")
            
        "make choropleth"
        g_plts.choropleth_plot(error_array, "Numbers of Agents", "Proportion Observed",
                               obs_titles[i])
        "make boxplot"
        g_plts.boxplot(error_frame, "Proportion Observed", "Grand Median L2s",
                       obs_titles[i])


def ex1_grand_no_split(source, destination, recall):
    """ Same as ex1_grand but doesn't split into observed/unobserved agents
    

    Parameters
    ----------
    source, destination : str
        `source` of experiment pickles and `destination` of the resulting plots
    recall : bool
        `recall` previously aggregated pickles from a np/pandas frame for speed?

    Returns
    -------
    None.

    """
    prefix = "ukf*"
    
    file_params = {
        "agents":  [10, 20, 30],
        "prop": [0.25, 0.5, 0.75, int(1)],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination
    }

    "initialise plot for observed/unobserved agents"
    g_plts = grand_plots(file_params, True, restrict=None, observed=True)
    "make dictionary"
    with HiddenPrints():
        L2 = g_plts.data_extractor()
        
    if not recall:
        L2 = g_plts.data_extractor()
        "make pandas dataframe for seaborn"
        error_frame = g_plts.data_framer(L2)
        "make choropleth numpy array"
        error_array = g_plts.choropleth_array(error_frame)
        error_frame.to_pickle(source + "Mixed_distance_error_array.pkl")
        np.save(source + "Mixed_distance_error_array", error_array)
    else:
        error_array = np.load(source + "Mixed_distance_error_array.npy")
        error_frame = pd.read_pickle(source + "Mixed_distance_error_array.pkl")
        
    "make pandas dataframe for seaborn"
    error_frame = g_plts.data_framer(L2)
    "make choropleth numpy array"
    error_array = g_plts.choropleth_array(error_frame)
    "make choropleth"
    g_plts.choropleth_plot(error_array, "Numbers of Agents",
                           "Proportion Observed", "Mixed")
    "make boxplot"
    g_plts.boxplot(error_frame, "Proportion Observed",
                   "Grand Median L2s", "Mixed")

# %%
if __name__ == "__main__":
    
    recall = True
    #main(ex1_grand, "/Users/medrclaa/scalability_results/ukf*", "../plots/", recall)
    #main(ex1_grand, "/media/rob/ROB1/ukf_results/", "../plots/", recall)
    
    #main(ex1_grand_no_split, "/Users/medrclaa/ukf_results/ukf*", "../plots/")
    main(ex1_grand_no_split, "/media/rob/ROB1/ukf_results/", "../plots/", recall)
