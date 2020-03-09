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
import matplotlib.pyplot as plt
import sys
import os
from astropy.stats import RipleysKEstimator
import pandas as pd

from linearmodels import PanelOLS, RandomEffects
from linearmodels.panel import compare
import statsmodels.api as sm
import statsmodels.formula.api as smf

class HiddenPrints:
    
    
    """stop repeating printing from stationsim 
    We get repeats of `iterations : X` prints as it jumps back 
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
"""
cell for spatial collision tests
based on 
https://docs.astropy.org/en/stable/stats/ripley.html
https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:ripley_s_k_and_pair_correlation_function"
"""


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
            print("\r" + str((i+1)/len(data)*100) + "% complete                    ", end = "")
        print("")
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
    


    def two_group_spaghetti(self, r, rkes1, rkes2, mod):
        
        
        """
        plot two groups with two different slopes
        """
    
        f = plt.figure()
        
        for i in range(len(rkes1)):
            plt.plot(r, rkes1[i], color = "black")
            plt.plot(r, rkes2[i], color = "orangered")
                    
        slopes = np.array(mod.fit().params)

        plt.plot(r, r**2*slopes[0] + r*slopes[2], color = "cyan", linewidth = 6, 
                 label = "group 1", alpha = 0.4)

        plt.plot(r, r**2*(slopes[0]+slopes[1]) + r*(slopes[2]+slopes[3]) ,
                 color = "magenta", linewidth = 6, label = "group 2", alpha = 0.4)
        plt.legend()
        plt.xlabel("r")
        plt.ylabel("RK Score")
        
    def main(self, n, source = None, pickle_file = None, save = False):
        
        with HiddenPrints():
            models = generate_model_sample(n, model_params)
        collisions = [model.history_collision_locs for model in models]
        
        rkes, r = self.ripleysKE(collisions)
        
        data1 = self.panel_regression_prep(rkes,r,0)
        
        try:
            pickle_data2 = depickler(source, pickle_file)
            rkes2, r2 = pickle_data2["rkes"], pickle_data2["r"]
    
        except:
        
            with HiddenPrints():
                models2 = generate_model_sample(n, model_params)
            collisions2 = [model.history_collision_locs for model in models2]
            
            rkes2, r2 = self.ripleysKE(collisions2)
        
        
        data2 = self.panel_regression_prep(rkes2,r2,1)
        
        
        data= self.compare_panel_regression_prep(data1, data2)
        mod1 = "refer to R script"
        """
        data = data.set_index(["ids", "x"], drop = False)
        
        
        #mod = PanelOLS(data.y, data[["x", "split"]])
        #mod3 = PanelOLS.from_formula("y ~ I(x)*split -split", data)
        #mod2 = PanelOLS.from_formula("y ~ I(x**2)*split -split", data)
        #mod1 = PanelOLS.from_formula("y ~ I(x**2)*split + I(x)*split -split", data)
    
        mod1 = RandomEffects.from_formula("y ~ I(x**2)*split + I(x)*split - split", data)
        
        #print(compare({"x+x**2": mod1.fit(), "x**2": mod2.fit(), "x": mod3.fit()}))
        
               
        
        mod1 = smf.mixedlm("y ~  I(x**2)*z(split) + x*z(split) - 1", data, groups=data["ids"], re_formula="~I(x**2) + x - 1")
        
        
        self.two_group_spaghetti(r, rkes, rkes2, mod1)
        """
        if save:
            data.to_csv("rk" + self.id + ".csv", index_col = False)
            
        return models, data, mod1
    
#%%
if __name__ == "__main__":
    
    model_params = {

    'width': 200,
    'height': 50,
    'pop_total': 10,
    
    'gates_in': 3,
    'gates_out': 2,
    'gates_space': 1,
    'gates_speed': 1,
    
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
    #data, mod = ssRK.main(n, "", "/Users/medrclaa/dust/Projects/ABM_DA/stationsim/400_50_50_0.01ripleys_k.pkl")
    models, data, mod = ssRK.main(n, save = True)
    

