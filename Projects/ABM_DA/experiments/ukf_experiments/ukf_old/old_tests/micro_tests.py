#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 15:33:05 2020

@author: rob

Testing for the micro level defined as tests for agents' basic elements and 
interactions from their perspective. Aim to htest that each individual testable
element is functioning as desired. I.E, testfunctional and structural defects
in a testable element. For example:
    
    - test agent building blocks like behaviour knowledge etc.
    - general deterministic/white box functions
    - building blocks of the environment e.g. non agents
    - test agents lifetime output
    - test an agent achieves something in a considerable time frame unver
        varying IC.
    - test interactions between elements
    - test quality properties of agents (e.g. number of behaviours scheduled at a 
    specific time.)
"""
import numpy as np
import unittest
from copy import deepcopy
import sys

sys.path.append("../ukf_modules")
from default_ukf_configs import model_params, ukf_params
from ukf_ex1 import omission_params

sys.path.append("../../../stationsim")
from stationsim_model import Model
from ukf2 import ukf_ss

default_model_params = deepcopy(model_params)
default_ukf_params = deepcopy(ukf_params)

def generate_test_ABM(start_model):
    
    
    """generate a seeded StationSim ABM run for testing
    
    """
    test_model = deepcopy(start_model)
    for _ in range(model_params["step_limit"]):
        
        test_model.step()
        activity = [agent.status for agent in test_model.agents]
        
        if sum(activity) == 2 * model_params["pop_total"]:
            break
        else:
            pass
        
    locs = np.reshape(np.ravel(test_model.history_state), (-1,model_params["pop_total"]))

    return test_model, locs

def generate_test_ukf(n, prop, start_model):
    
    """ generate a seeded ukf_model for testing
    
    """
    test_model = deepcopy(start_model)
    n = model_params["pop_total"]
    model_params, ukf_params = omission_params(n, prop)
    u = ukf_ss(model_params,ukf_params,test_model)
    u.main()
    
    obs,preds,truths,nan_array= u.data_parser()

    return u, obs, preds, truths





"""
The following two classes are for ABM specific testing. 
Will be one for the future.
"""

class micro_Tests():
    
    """
    testing individual mechanisms of agents/ environment factors and the UKF
    The idea is to test "white box" mechanisms in the standard unit testing 
    expected input/output fashion.
    """
    
    def __init__(self, model):
        self.model = model
        
class Test_macros():
    
    """
    testing the ABM/UKF as an entire entity (environment).
    
    The idea is to test "black box" elements made so by the stochasticity of agents.
    The standard idea of a certain input gives a specific output applies but is often not 
    straightforwards due to said randomness. As such, we implement a number of monte carlo
    techniques asserting satistically over multiple ABM runs.
    We do NOT test deterministic elements such as individual agent mechanisms
    """
    
    def __init__(self, seed, model, model_params, ukf_params):
        self.model = model
        self.model_params = deepcopy(model_params)
        self.ukf_params = deepcopy(ukf_params)
        self.model_params["pop_total"] = 5    
        self.seed = seed
        
    def test_abm(self):
        
        """seed the ABM, run it twice and check the results are the same"""
        
        "needs a big seed. make it an int <2**32 -1"

        np.random.seed(self.seed) 
        test_model ,locs = generate_test_ABM(start_model)
        np.random.seed(self.seed) 
        test_model2 ,locs2 = generate_test_ABM(start_model)
        
        assert locs == locs2

    def test_ukf(self):
        np.random.seed(self.seed) 
        u ,obs, preds, truths = generate_test_ukf(model_params["pop_total"], 1, start_model)
        np.random.seed(self.seed) 
        u2, obs2, preds2, truths2 = generate_test_ukf(model_params["pop_total"], 1, start_model)
        
        assert obs == obs2
        assert preds == preds2
        assert truths == truths2
        
class Test_UKF():
        
    
    """ unit tests for the UKF given some desired experiment module input
    
    """
    
    def __init__(self):
        
        pass
    
    def test_Sigmas(self):
        
        pass
    
    def test_Unscented_Mean(self):
        
        pass
    
    def test_Covariance(self):
   
        pass
    
    
class Test_ex1(object): 
    
    
    """tests for 1st ukf experiment module
    
    """
    
    def __init__(self, model):
        pass
    
if __name__ == "__main__":
    start_model = Model(**model_params)
    macros = Test_macros(start_model, 8**8, default_model_params, default_ukf_params)

    