#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 10:32:41 2020

@author: medrclaa

Testing stationsim to see whether there is substantial crowding or
if 
"""

import sys
import numpy as np
from math import floor

"local imports"
from ukf_fx import fx
from ukf_plots import ukf_plots
import default_ukf_configs 

"""
try/except here used if running __main__ here. 
if calling from a parent directory (e.g. ukf_experiments)
call them there instead with the corresponding file append.
e.g.     
sys.path.append("../stationsim")
"""

try:
    sys.path.append("../../../stationsim")
    from ukf2 import ukf_ss, pickle_main
    from stationsim_model import Model

except:
    "couldn't import from stationsim."
    pass


    

if __name__  == "__main__":
    model_params = {

    "pop_total" : 50,
    'width': 50,
    'height': 50,
    
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
    

    
    model = Model(**model_params)
    
    while model.status !=0:
        model.step()
    
    
    marker_attributes = {
    "markers" : {-1 : "o"},
    "colours" : {-1 : "black"},
    "labels" :  {-1 :"Pseudo-Truths" }
    }
    
    "dummy ukf params and class for plots"
    ukf_params = {}
    u = ukf_ss(model_params, ukf_params, model)
    
    
    plts = ukf_plots(u, "../plots/", "crowd_test_", False, True, marker_attributes)
        
    truths = np.array(model.history_state).flatten().reshape((model.step_id,
                                                              2*model.pop_total))
    
    nan_array = np.ones(shape = truths.shape)*np.nan
    for i, agent in enumerate(model.agents):
        "find which rows are  NOT (None, None). Store in index. "
        array = np.array(agent.history_locations)
        index = ~np.equal(array,None)[:,0]
        "set anything in index to 1. I.E which agents are still in model."
        nan_array[index,2*i:(2*i)+2] = 1
        
    truths*=nan_array
    
    plts.trajectories(truths)
    model.get_collision_map()

    
    