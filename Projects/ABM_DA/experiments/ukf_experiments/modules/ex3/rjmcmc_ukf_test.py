#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 15:21:47 2020

@author: medrclaa
rjmcmc_ukf tests
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as AAAE
from rjmcmc_ukf import rjmcmc_ukf
import sys
sys.path.append("..")
from sensors import generate_Camera_Rect
import default_ukf_configs as configs
sys.path.append("../../../../stationsim")
from stationsim_model import Model
import multiprocessing

class Test_rjmcmc_ukf(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """unittest.TestCase's special __init__

        """
        # init params for a SEEDED stationsim
        cls.model_params = {

        'width': 200,
        'height': 50,
        'pop_total': 2,
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
        
        'do_history' : True,
        'do_print' : True,
        
        'random_seed' : 8,
        }
        cls.ukf_params = configs.ukf_params
        cls.base_model = Model(**cls.model_params)
        pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
        cls.rjmcmc_ukf = rjmcmc_ukf(cls.model_params, cls.ukf_params,
                                    cls.base_model, pool)
    
    def test_gates_dict(self):
        get_gates_dict, set_gates_dict  = self.rjmcmc_ukf.gates_dict(self.base_model)
        get_expected = {'[200.          16.66666667]': 0,
                    '[200.          33.33333333]': 1}
        set_expected = {0 : np.array([200.        ,  16.66666667]),
                        1 : np.array([200.        ,  33.33333333])}
        
        self.assertEqual(get_gates_dict, get_expected)
        #because cant almost assert dictionaries with numpy values
        AAAE([*set_gates_dict.values()], [*set_expected.values()])
        self.assertEqual([*set_gates_dict.keys()], [*set_expected.keys()])

        
    def test_get_gates(self):
        """Assert that rjmcmc_ukf.test_gates returns the gates from some stationsim
        """
        #nudge stationsim so all agents in model
        for _ in range(10):
            self.base_model.step()
            
        rjmcmc_ukf = self.rjmcmc_ukf
        get_gates_dict, set_gates_dict = rjmcmc_ukf.gates_dict(self.base_model)
        gates = rjmcmc_ukf.get_gates(self, self.base_model, get_gates_dict)
        expected = [0, 1]
        self.assertEqual(gates, expected)
        
    def test_set_gates(self):
        rjmcmc_ukf = self.rjmcmc_ukf
        get_gates_dict, set_gates_dict = self.rjmcmc_ukf.gates_dict(self.base_model)
        new_gates = [0, 1]
        base_model = rjmcmc_ukf.set_gates(rjmcmc_ukf, self.base_model, 
                                          new_gates, set_gates_dict)
        
        gates = rjmcmc_ukf.get_gates(self, base_model, get_gates_dict)
        expected = [0, 1]
        self.assertEqual(gates, expected)

    
    def test_draw_new_gates(self):
        rjmcmc_ukf = self.rjmcmc_ukf
        get_gates_dict, set_gates_dict = self.rjmcmc_ukf.gates_dict(self.base_model)

        current_gates = self.rjmcmc_ukf.get_gates(self, self.base_model, get_gates_dict)
        gate_probabilities = np.array([[1, 0],
                                       [0, 1 ]])
        n_gates = 5
        new_gates = rjmcmc_ukf.draw_new_gates(gate_probabilities, 
                                              current_gates, n_gates)
        base_model = rjmcmc_ukf.set_gates(rjmcmc_ukf, self.base_model, 
                                          new_gates, set_gates_dict)
        expected = rjmcmc_ukf.get_gates(rjmcmc_ukf, base_model, get_gates_dict)
        
        self.assertEqual(new_gates, expected)
    
    def test_agent_probabilites(self):
        rjmcmc_ukf = self.rjmcmc_ukf
        theta = ((np.pi/10))
        gate_probabilites = rjmcmc_ukf.agent_probabilities(self.base_model, theta, True)
        expected = np.array([[0.5, 0.5],
                             [0.5, 0.5]])
        AAAE(gate_probabilites, expected)
        
        
if __name__ == "__main__":
    unittest.main()