#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:51:50 2020

@author: rob
"""
import sys
from depickle import grand_plots, main

sys.path.append("..")

def ex0_grand(source, destination, recall):
    """ main function for parsing experiment 0 results into summary plots
    

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
    n = 30  # population size
    file_params = {
        "rate":  [1.0, 2.0, 3.0, 5.0, 10.0],
        "noise": [0., 0.5, 1.0, 5.0, 10.0],
        # "source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source": source,
        "destination": destination,
    }

    g_plts = grand_plots(file_params, True)
    L2 = g_plts.numpy_parser()
    L2_frame, best_array = g_plts.numpy_extractor(L2)
    g_plts.comparison_choropleth(n, L2_frame, best_array,
                                 "Observation Noise Standard Deviation",
                                 "Data Assimilation Window", "bench")
    g_plts.comparisons_3d(n, L2_frame, best_array)



if __name__ == "__main__":
    recall = False
    main(ex0_grand,  f"/home/rob/gcs_ex0_results/gcs*", "../plots/", recall)