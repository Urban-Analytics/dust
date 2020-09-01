#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:30:55 2020

@author: rob
"""
import depickle


def ex3_grand_position(source, destination):

    file_params = {
        "agents":  [10],
        "jump_rate": [5],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination
    }


    "initialise plot for observed/unobserved agents"
    g_plts = depickle.grand_plots(file_params, True, restrict=None, observed=True)
    "make dictionary"
    L2 = g_plts.data_extractor()
    "make pandas dataframe for seaborn"
    error_frame = g_plts.data_framer(L2)
    "make choropleth numpy array"
    error_array = g_plts.choropleth_array(error_frame)
    "make choropleth"
    g_plts.choropleth_plot(error_array, "Numbers of Agents",
                           "n_jumps", "")
    "make boxplot"
    g_plts.boxplot(error_frame, "Proportion Observed",
                   "Grand Median L2s", "")

def ex3_grand_gates(source, destination):

    file_params = {
        "agents":  [10],
        "jump_rate": [5],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination
    }

    "initialise plot for observed/unobserved agents"
    g_plts = depickle.grand_plots(file_params, True)
    "make dictionary"
    gate_distances = g_plts.gates_extractor()
    
    "make pandas dataframe for seaborn"
    data_frame = g_plts.gates_data_frame(gate_distances)
    "make choropleth numpy array"
    g_plts.gates_data_lineplot(data_frame, "rjmcmc_UKF_")
    
if __name__ == "__main__":
    source = f"/home/rob/ukf_ex3_tests/rj*"
    destination = "../plots/"
    ex3_grand_position(source, destination)
    ex3_grand_gates(source, destination)
