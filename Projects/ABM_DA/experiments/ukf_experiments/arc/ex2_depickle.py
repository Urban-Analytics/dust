#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 12:13:59 2020

@author: rob
"""

import sys
import numpy as np
import pandas as pd

from depickle import grand_plots, main

sys.path.append("../../..")
sys.path.append("../..")
sys.path.append("..")
#sys.path.append("../ukf_old")

from stationsim.ukf2 import pickle_main, truth_parser, preds_parser

from modules.ukf_plots import L2s as L2_parser
from modules.ukf_fx import HiddenPrints


def ex2_grand(source, destination, recall):
    """ main function for parsing experiment 2 results into summary plots
    

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
    
    prefix = "agg*"
    file_params = {
        "agents":  [10, 20],
        "bin": [10, 25],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination,
    }

    "init plot class"
    g_plts = grand_plots(file_params, True)
    "make dictionary"
    
    if not recall:
        L2 = g_plts.data_extractor()
        "make pandas dataframe for seaborn"
        error_frame = g_plts.data_framer(L2)
        "make choropleth numpy array"
        error_array = g_plts.choropleth_array(error_frame)
        error_frame.to_pickle(source + "Aggregate_distance_error_array.pkl")
        np.save(source + "Aggregate_distance_error_array", error_array)
    else:
        error_array = np.load(source + "Aggregate_distance_error_array.npy")
        error_frame = pd.read_pickle(source + "Aggregate_distance_error_array.pkl")
        
    with HiddenPrints():
        L2 = g_plts.data_extractor()
        
    "make choropleth"
    g_plts.choropleth_plot(error_array, "Numbers of Agents",
                           "Proportion Observed", "Aggregate")
    "make boxplot"
    g_plts.boxplot(error_frame, "Grid Square Size",
                   "Grand Median L2s", "Aggregate")


if __name__ == "__main__":
    
    recall = True
    # main(ex2_grand, "/Users/medrclaa/new_aggregate_results/agg_ukf*", "../plots")
    main(ex2_grand, "/media/rob/ROB1/ukf_results/", "../plots", recall)

