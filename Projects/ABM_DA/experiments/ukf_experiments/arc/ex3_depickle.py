#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:30:55 2020

@author: rob
+"""
import depickle
import pandas as pd
import os
from arc_ukf_ex3 import ex3_saver

def ex3_grand_position(source, destination, recall):

    file_params = {
        "agents":  [10, 20, 30],
        "jump_rate": [5, 10, 20],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination
    }

        
    "initialise plot for observed/unobserved agents"
    g_plts = depickle.grand_plots(file_params, True, restrict=None, observed=True)
    
    if not recall:
            "make dictionary"
            L2 = g_plts.data_extractor()
            "make pandas dataframe for seaborn"
            error_frame = g_plts.data_framer(L2)
            error_frame.to_pickle(os.path.split(source)[:-1][0] + "/rjukf_locations_frame.pkl")
    else:
        error_frame = pd.read_pickle(os.path.split(source)[:-1][0] + "/rjukf_locations_frame.pkl")

    "make choropleth numpy array"
    error_array = g_plts.choropleth_array(error_frame)
    "make choropleth"
    g_plts.choropleth_plot(error_array, "Numbers of Agents",
                           "Reversible Jump Window", "")
    "make boxplot"
    g_plts.boxplot(error_frame, "Reversible Jump Window",
                   "Grand Median L2s (log(metres))", "")

def ex3_grand_gates(source, destination, recall):
    """ main function for parsing experiment 3 results into summary plots
    

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
    file_params = {
        "agents":  [10, 20, 30],
        "jump_rate": [5, 10, 20],
        "source": source,
        "destination": destination
    }

    "initialise plot for observed/unobserved agents"
    g_plts = depickle.grand_plots(file_params, True)
    "make dictionary"
    if not recall:
        gate_distances = g_plts.gates_extractor()
        data_frame = g_plts.gates_data_frame(gate_distances)
        data_frame.to_pickle(os.path.split(source)[:-1][0] + "/rjukf_gates_frame.pkl")
        
    else:
        data_frame = pd.read_pickle(os.path.split(source)[:-1][0] + "/rjukf_gates_frame.pkl")
    "make pandas dataframe for seaborn"
    "make choropleth numpy array"
    g_plts.gates_data_lineplot(data_frame,"Time", "Error (Number of Gates)", "rjmcmc_UKF_")
    
if __name__ == "__main__":
    # add recall function for times sake
    source =  f"/home/rob/ukf_ex3_tests/rj*"
    destination = "../plots/"
    recall = True
    ex3_grand_position(source, destination, recall)
    ex3_grand_gates(source, destination, recall)
