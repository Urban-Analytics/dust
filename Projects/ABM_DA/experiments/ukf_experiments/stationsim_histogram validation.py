#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:49:24 2020
@author: medrclaa
stationsim spatial validation tests.
testing the proximity of one seeded stationsim run to another
"""

from stationsim_model import Model
import numpy as np
from scipy.stats import binned_statistic, normaltest, norm
from scipy.stats import chisquare
from math import ceil
import matplotlib.pyplot as plt
from seaborn import kdeplot
from sklearn.mixture import GaussianMixture
import sys
import os

"import pickle from ukf for easy saving of results."
from ukf2 import pickler, depickler

import pandas as pd
from linearmodels import PanelOLS
 
class HiddenPrints:
    
    
    """stop repeating printing from stationsim 
    We get a low of `iterations : X` prints as it jumps back 
    and forth over every 100th step. This stops that.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_model_sample(n_runs, model_params, seed = [None]):

    
    """ function for generating stationsim model runs to test
    
    Parameters
    ------
    n_runs : int
        `n_runs`number of stationsim runs to generate. 
        Must be positive intiger.
        
    model_params : dict
        `model_params` dictionary of model parameters  required for 
        stationsim to run. See stationsim_model.py for more details.
        
    seed : list
        `seed` seeding for stationsim. Can a single seed for every run
        or a list of n_runs seeds. 
    """
    
    models = []
    
    if len(seed) == 1 :
        seed *= n_runs
    elif len(seed) != n_runs:
        print("not enough seeds specificed. Either provide precisely 1 or " 
              + f"{n_runs} seeds")
    
    for _ in range(n_runs):
        model = Model(**model_params)
        while model.status == 1:
            model.step()
        models.append(model)
    
    return models

#%%
"""IGNORE THIS CELL THIS TEST IS TOO COMPLICATED
Attemps at modelling agent time taken histograms as normal mixture.
Fit normal mixture using sklearns GaussianMixture.
Test fit using pearsons chi squared test.
"""

def left_merge(counts, bins, thresh):
    
    
    """ used in pearsons chi test.
    if a bin has <thresh items in it merge it 
    with the bin to the left
    e.g with thresh = 5
    |0-2|3-4|5-6|
    |20 |6  |3  |
    becomes
    |0-2|3-6|
    |20 |9  |   
    
    This function takes two lists and merges them from left to right
    """

    n = counts.shape[0]

    if counts[0]<thresh:
        print("warning 0th bin less than threshold. may provide poor results.")
    
    for i in range(1, n - 1):
        count = counts[n - i]
        if count < thresh:
            counts[n - i - 1] += count
            counts = np.delete(counts, n - i)
            bins = np.delete(bins, n - i)
    
    return counts, bins
   
def data_parser(data, bin_size):
    
    
    """
    take list of  data and return binned counts left merged so no bin has <5 
    entries for pearsons chi test.
    """
    


    max_bin = int(ceil(np.max(data) / bin_size)) * bin_size
    bins = np.arange(0, max_bin + bin_size, bin_size)    
    data = binned_statistic(data, data , bins = bins, statistic = "count")

    merged_counts, merged_bins = left_merge(data[0], data[1], 5)
    
    return merged_counts, merged_bins
    
def pearson_test(merged_counts, merged_bins, dist, **dist_kwargs):
    expected = []
    for i in range(1, len(merged_counts)):
        "calculate poisson probability of being within each bin."
        e = (dist.cdf(merged_bins[i], **dist_kwargs) - 
                            dist.cdf(merged_bins[i-1], **dist_kwargs))
        
        expected.append(e * np.sum(merged_counts))
        
    "final section of pdf between last bin edge and infinity"
    expected.append((1-dist.cdf(merged_bins[-1], **dist_kwargs))* np.sum(merged_counts))
    expected = np.array(expected).ravel()
    chisq, p = chisquare(merged_counts, expected)    
    return chisq, p
    
def anscombe_transform(x):
    return np.sqrt(x + 3/8)

def normal_test(x):
    
    k,m = normaltest(x)
    return m
                  
def mixture_cdf(x, **dist_kwargs):
    
    weights = dist_kwargs["weights"]
    means = dist_kwargs["means"]
    covs = dist_kwargs["covs"]

    out = 0
    
    for i in range(means.shape[0]):
        out += weights[i] * norm.cdf(x,means[i],covs[i])
    
    return out

def hist_main():
    models = generate_model_sample(1)
    
    data = [model.steps_exped for model in models][0]
    data = np.array(data)
    data = anscombe_transform(data)

    bin_size = (np.max(data)-np.min(data))/10
    merged_counts, merged_bins = data_parser(data, bin_size)
    
    data2 = data.reshape(-1,1)
    mix = GaussianMixture(3, "full")
    mix.fit(data2)
    weights = mix.weights_
    means = mix.means_
    covs = mix.covariances_
    mix.cdf = mixture_cdf
    
    dist_kwargs = {"weights" : weights, "means" : means, "covs" : covs}
    
    chisq, p = pearson_test(merged_counts, merged_bins, mix, **dist_kwargs)
    print(chisq, p)
    normal_x = np.linspace(min(data), max(data), 200)
    normal_y = np.zeros(normal_x.shape)
    for i in range(weights.shape[0]):
        normal_y += weights[i] * norm.pdf(normal_x, means[i], np.sqrt(covs[i])).ravel()
   
    
    
    
    
    plt.figure()
    plt.hist(data,label = "step delay densities",density = True)
    plt.plot(normal_x,normal_y, label = "gaussian mixture estimate")
    plt.xlabel("time taken")
    plt.ylabel("density")
    plt.legend()
    
    
