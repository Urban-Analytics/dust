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
import sys
import os
from astropy.stats import RipleysKEstimator

import pandas as pd




class HiddenPrints:
    
    """stop unnecessary printing from stationsim 
    
    We get a lot of `iterations : X` prints for a large number of 
    stationsim runs. This stops the printing for tidiness.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
class stationsim_RipleysK():
    
    
    """
    """


    def generate_model_sample(self, n_runs, model_params):
    
        
        """ function for generating stationsim model runs to test
        
        Parameters
        ------
        n_runs : int
            `n_runs`number of stationsim runs to generate. 
            Must be positive intiger.
            
        model_params : dict
            `model_params` dictionary of model parameters  required for 
            stationsim to run. See stationsim_model.py for more details.
            
        Returns
        ------
        models : list
            a list of completed stationsim `models` given the required
            width, height, pop_total, and gate_speed.
        """
        
        "placeholder list"
        models = []
        
        "supress excessive printing"
        with HiddenPrints():
            for _ in range(n_runs):
                "generate model and run til status goes back to 0 (finished)"
                model = Model(**model_params)
                while model.status == 1:
                    model.step()
                models.append(model)
            
        return models

    def ripleysKE(self, models, model_params):
        
        
        """ Generate Ripley's K (RK) curve for collisions in StationSim region.
        
        For more info on RK see:
        https://docs.astropy.org/en/stable/stats/ripley.html
        https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:ripley_s_k_and_pair_correlation_function"
        
        
        Parameters
        ------
        
        Returns 
        ------
        
        """
        
        "define area of stationsim."
        width = model_params["width"]
        height = model_params["height"]
        area = width*height
        
        "init astropy RKE class with stationsim boundaries/area"
        rke = RipleysKEstimator(area = width * height, 
                                x_max = width, y_max = height,
                                y_min = 0, x_min = 0)
        
        """generate list of radii to assess. We generate 10 between 0 
        and the root of half the total area. More radii would give a higher 
        resolution but increases the computation time.
        
        see https://wiki.landscapetoolbox.org/doku.php/spatial_analysis_methods:
        ripley_s_k_and_pair_correlation_function
        for more details on this"
        """
        r = np.linspace(0, np.sqrt(area/2), 10)
        
        "generate the full list of radii for data frame later."
        "just repeats r above for how many models we have"
        rs = [r]*len(models)
        
        "placeholder list for RK estimates"
        rkes = []        
        for i, model in enumerate(models):
            
            """estimate RK curve given model collisions and list of radii
            Note mode arguement here for how to deal with common edge effect problem.
            Choice doesnt seem to have much effect in this case.
            Either none or translation recommended.
            """
            
            collisions = np.vstack(model.history_collision_locs)
            #rkes.append(rke(collisions, radii=r, mode='none')) 
            rkes.append(rke(collisions, radii=r, mode='translation'))
            #rkes.append(rke(collisions, radii=r, mode='ohser')) 
            #rkes.append(rke(collisions, radii=r, mode='var-width')) 
            #rkes.append(ke(collisions, radii=r, mode='ripley')) 
            
            "this can take a long time so here's a progess bar"
            
            print("\r" + str((i+1)/len(models)*100) 
                  + "% complete                  ", end = "")

        
        
        return rkes, rs
       
    
    def panel_regression_prep(self, rkes, rs, id_number):
        
        
        """ Build the list of model RKEs into a dataframe
        
        Output dataframe has 4 columns
        
        x: list of radii on which RK is estimates. renamed to x for use 
        in R regression later
        
        y: RKE estimate corresponding to a given model's collisions and radii. 
        Renamed to y for R regression later
        
        ids: identifies which specific model the row belongs to.
        This has format "A_B" it is the Bth model of batch A. 
        For example if the id reads "0_8" it is from the eighth model of the 
        0th batch
        
        split: identifies which batch the row belongs to. Either 0 (control group)
        or 1 (test group)
        
        Parameters
        ------
        
        rkes, rs : list
            list of lists of radii `rs` and corresponding RK estimates `rkes`. 
            Each pair of sublists corresponds to an individual model.
            
        id_number : int
            Assign each row of the data frame a number 0 if it is a control group
            (e.g. python station results) or 1 if it is a test group (e.g. cpp results.)
            
        Returns
        ------
        
        data: array_like
            assembled RK `data` from our list of models ready for fitting a
            regression in R. Should have 4 columns as defined above. 
        """
        
        num_rs = len(rs[0])
        rkes = np.ravel(rkes)
        rs = np.ravel(rs)
        
        data = pd.DataFrame([rs, rkes]).T
        
        "rename rs and rkes columns for regression later."
        data.columns = ["x", "y"]
        
        """generate individual model ID numbers. Each model has 10 entries 
        (per radii) so every 10 entrires belong to the ith model.
        """
        
        ids = (np.arange(len(rkes))//num_rs).tolist()
        
        
        for i, item in enumerate(ids):
            ids[i] = str(id_number) + "_" + str(item)
        data["ids"] = ids
        
        split = [id_number] * len(rkes)
        data["split"] = split
        
        return data
        
    
    
    def generate_Control_Frame(self, model_params, n_runs):
        
        
        """
        """
        
        models = ssRK.generate_model_sample(n_runs, model_params)
        rkes, rs = ssRK.ripleysKE(models, model_params)
    
        data = ssRK.panel_regression_prep(rkes, rs, 0)
        
        width = model_params["width"]
        height = model_params["height"]
        pop_total = model_params["pop_total"]
        gates_speed = model_params["gates_speed"]
               
        f_name = "RK_csvs/control_" + f"{width}_{height}_{pop_total}_{gates_speed}"
        f_name += ".csv"
        
        self.save_Frame(data, f_name)
        
        return data
        
    def generate_Test_Frame(self, model_params, n_runs):
        
        
        """
        """
        
        models = ssRK.generate_model_sample(n_runs, model_params)
        rkes, rs = ssRK.ripleysKE(models, model_params)
    
        data = ssRK.panel_regression_prep(rkes, rs, 1)
        
        return data
        
    def save_Frame(self, data, f_name):
        data.to_csv(f_name, index = False)

    def load_Frame(self, f_name):
        return pd.read_csv(f_name)
    
    
    def main(self, n_test_runs, model_params):
     
        width = model_params["width"]
        height = model_params["height"]
        pop_total = model_params["pop_total"]
        gates_speed = model_params["gates_speed"]
               
        f_name = "RK_csvs/control_" + f"{width}_{height}_{pop_total}_{gates_speed}" + ".csv"
        
        try:
            data_control = self.load_Frame(f_name)
            print("Control data found at: " + f_name)
        except:
            print("No control frame found for given parameters.")
            print("Generating control frame using large number of models (100).")
            print("This may take a while if you have a large population of agents")
            data_control = self.generate_Control_Frame(model_params, 100)
            
        data_test = self.generate_Test_Frame(model_params, n_test_runs)
        
        data = pd.concat([data_control, data_test])
        
        f_name = "RK_csvs/joint_" + f"{width}_{height}_{pop_total}_{gates_speed}"
        f_name += ".csv"
        self.save_Frame(data, f_name)

        
    
#%%
if __name__ == "__main__":
    
    model_params = {

    'width': 200,
    'height': 50,
    'pop_total': 30,
    'gates_speed': 1,

    'gates_in': 3,
    'gates_out': 2,
    'gates_space': 1,
    
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
    n = 20
    ssRK = stationsim_RipleysK()
    ssRK.main(n, model_params)
    
    
    
    
    
    #data, mod = ssRK.main(n, "", "/Users/medrclaa/dust/Projects/ABM_DA/stationsim/400_50_50_0.01ripleys_k.pkl")
    

