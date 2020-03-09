#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:54:49 2020

@author: medrclaa
"""

import sys
import numpy as np

sys.path.append("../stationsim")
from stationsim_model import Model
import unittest

def model_Run(model):
    
    
    """runs a stationsim model until everyone leaves
    """
    
    
    for _ in range(3600):
        model.step() # step model
        status = [agent.status for agent in model.agents] #check agent statuses
        if all(status) == 2:
            break
        
    return model
    
 
    
class Test_StationSim_Seeding(unittest.TestCase):
    
    def test_StationSim_seeding(self):
        
        
        """ Test stationsim seeding by running two models with same seed 
        and comparing positions.
        
        Parameters
        ------
        model_params : dict
            `model_params` dictionary of stationsim model_parameters.
            
        seed : int
            `seed` for fixing numpy random outputs. must be 0<int<2**32-1.
        
        see stationsim_model for model parameter defintions"""
        
        model_params = {
            
        'pop_total':5,
        'width': 200,
        'height': 100,
        
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
        
        "random_seed" : 8**8
        }
        
        model1 = Model(**model_params)
        model1 = model_Run(model1)
        array1 = np.hstack(model1.history_state)
        
        model2 = Model(**model_params)
        model2 = model_Run(model2)
        array2 = np.hstack(model2.history_state)
        
        self.assertAlmostEqual(np.nansum(array1-array2), 0)


if __name__ == "__main__":
    
    test_seeding = Test_StationSim_Seeding()
    unittest.main()
