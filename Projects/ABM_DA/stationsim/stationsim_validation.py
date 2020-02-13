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
    
    
#%%
"""
cell for spatial collision tests
based on 
https://docs.astropy.org/en/stable/stats/ripley.html
https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:ripley_s_k_and_pair_correlation_function"
"""
from astropy.stats import RipleysKEstimator
from sklearn.linear_model import LinearRegression


class stationsim_RipleysK():
    
    
    """
    """
    
    def __init__(self, model_params):
        
        
        """
        """
        
        for key in model_params.keys():
            setattr(self, key, model_params[key])
            
        a = self.width
        b = self.height
        c = self.pop_total
        d = self.gates_speed
        self.id =  f"{a}_{b}_{c}_{d}"

    def ripleysKE(self, data):
        
        
        """
        """
        
        width = self.width
        height = self.height
        
        area = width*height
        rke = RipleysKEstimator(area = width * height, 
                                x_max = width, y_max = height,
                                y_min = 0, x_min = 0)
        
        r = np.linspace(0, np.sqrt(area/2), 10)
        rkes = []        
        for i, item in enumerate(data):
        
            #plt.plot(r, rke.poisson(r)) 
            rkes.append(rke(item, radii=r, mode='none')) 
            #plt.plot(r, rke(data, radii=r, mode='translation')) 
            #plt.plot(r, rke(data, radii=r, mode='ohser')) 
            #plt.plot(r, rke(data, radii=r, mode='var-width')) 
            #plt.plot(r, rke(data, radii=r, mode='ripley')) 
            print(i/len(data)*100," percent complete \r")
        return rkes, r
       
    def reg_rkes(self, rkes, r):
        
        
        """
        """
        
        reg_rkes = []
        for item in rkes:
            reg_rkes.append(np.vstack([r, item]).T)
        reg_rkes = np.vstack(reg_rkes)
        
        return reg_rkes
    
    def panel_regression_prep(self, rkes, r, id_number):
        
        
        """
        """
        
        reg_rkes = self.reg_rkes(rkes, r)
        
        data = pd.DataFrame(reg_rkes)
        
        
        data.columns = ["x", "y"]
        "pop x as its dropped by set_index"
        
        ids = []
        for i in range(len(rkes)):
            ids += [str(id_number) + "_" + str(i)]*len(r)
        data["ids"] = ids
                
        split = [id_number] * reg_rkes.shape[0]
        data["split"] = split
        
        return data
    
    def compare_panel_regression_prep(self, data1, data2):
        
        
        """
        """
        
        data = pd.concat([data1, data2])
        
        split = data.pop("split")
        split = pd.Categorical(split)
        data["split"] = split
        
        return data
    
    def spaghetti_plot(self, rkes, r):
        
        
        """
        """
        
        f = plt.figure()
        for item in rkes:
            plt.plot(item)
        plt.xlabel("radius ")
        plt.ylabel("ripleys K score")
        
        
        
    def simple_regression(self, reg_rkes):
        
        
        """
        """
        
        LR = LinearRegression(fit_intercept = False)
        fit = LR.fit(reg_rkes[:,0].reshape(-1,1),reg_rkes[:,1])
        
        g = plt.figure()
        plt.scatter(reg_rkes[:,0], reg_rkes[:,1])
        plt.plot(r, r*fit.coef_[0])
        plt.xlabel("radius of ripleys scan circle")
        plt.ylabel("ripleys K score") 
        title = self.id + f'Ripleys vs y = {fit.coef_}r '
        plt.title(title)
        
        self.coef = fit.coef_

    def collision_kde(self, data):
        
        
        """
        """
        
        f = plt.figure()
        kdeplot(data[:,0],data[:,1])
        plt.xlim(0,model_params["width"])
        plt.ylim(0,model_params["height"])
        plt.close()
        
    def save_results(self, model_params):
        
        
        """
        """
        
        save = {
                "model_params" : model_params, 
                "rkes" : self.rkes,
                "r" : self.r,
                "coef": self.coef,
                }
        pickler(save, "..", self.id + "_ripleys_k.pkl")

    def main(self, n, source = None, pickle_file = None):
        
        with HiddenPrints():
            models = generate_model_sample(n, model_params)
        collisions = [model.history_collision_locs for model in models]
        
        rkes, r = self.ripleysKE(collisions)
        
        data1 = self.panel_regression_prep(rkes,r,0)
        
        try:
            data2 = depickler(source, pickle_file)
        except:
            
            with HiddenPrints():
                models2 = generate_model_sample(n, model_params)
            collisions2 = [model.history_collision_locs for model in models2]
            
            rkes2, r2 = self.ripleysKE(collisions2)
            
            data2 = self.panel_regression_prep(rkes2,r2,1)
                
        data= self.compare_panel_regression_prep(data1, data2)
        data = data.set_index(["ids", "x"], drop = False)
        #mod = PanelOLS(data.y, data[["x", "split"]])
        mod = PanelOLS.from_formula("y ~ x*split -split", data)
        
        return data, mod
#%%
if __name__ == "__main__":
    
    model_params = {

    'width': 400,
    'height': 50,
    'pop_total': 50,
    
    'gates_in': 3,
    'gates_out': 2,
    'gates_space': 1,
    'gates_speed': 0.01,
    
    'speed_min': .2,
    'speed_mean': 1,
    'speed_std': 1,
    'speed_steps': 3,
    
    'separation': 5,
    'max_wiggle': 1,
    
    'step_limit': 3600,
    
    'do_history': True,
    'do_print': True,
    }
    n = 100
    ssRK = stationsim_RipleysK(model_params)
    data, mod = ssRK.main(n, "", "/Users/medrclaa/dust/Projects/ABM_DA/stationsim/400_50_50_0.01ripleys_k.pkl")
    
    
    
    