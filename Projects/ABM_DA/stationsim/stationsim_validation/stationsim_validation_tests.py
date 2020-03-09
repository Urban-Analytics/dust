#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 14:47:41 2020

@author: medrclaa
"""


import unittest
from stationsim_validation import stationsim_RipleysK
import numpy as np
from numpy.testing import assert_array_almost_equal as AAAE
import pandas as pd
import multiprocessing

class Test_ssRK(unittest.TestCase):
    
    
    @classmethod
    def setUpClass(cls):
        
        "initiate test class with some model parameters and init ssRK"
        
        cls.model_params = {

        'width': 200,
        'height': 50,
        'pop_total': 10,
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
        cls.ssRK = stationsim_RipleysK()
    
    
    def test_generate_Model_Sample(self):
        
        """ test generate_Model_Sample generates a list of stationsim models
        
        build a list of stationsim models of length 1 
        model has random_seed arguement 8
        will produce an expected number of collisions (484)
        assert number of collisions is the same
        
        (perhaps extend this to test further attributes.)
        """
        
        n = 1
        models = self.ssRK.generate_Model_Sample(n,
                                                 self.model_params)
        
        actual = len(models[0].history_collision_locs) 
        expected = 484
        
        "assert list of length 1"
        self.assertEqual(type(models), list)
        self.assertEqual(len(models), 1)
        
        "assert model has seed 8 by comparing number of collisions"
        self.assertEqual(actual, expected)


    def test_RipleysKE(self):
        
        """Test if Ripley's K calculation works using seeded model.
        
        Generates 2 list of length 1 numpy arrays for radii and RK scores.
        Passes if arrays are equal to expected seeded results.
        
        """
        
        n = 1
        models = self.ssRK.generate_Model_Sample(n,
                                                 self.model_params)
        actual_rkes, actual_rs = self.ssRK.ripleysKE(models, 
                                                     self.model_params)
        
        expected_rkes = np.array([   0.      ,  747.576653, 1735.5369  ,
                                  2419.18413 , 3162.740367, 3830.83855 , 
                                  4518.04765 , 5203.250371, 5935.255249, 
                                  6653.870481])
        
        expected_rs = np.array([ 0.      ,  7.856742, 15.713484,
                                   23.570226, 31.426968, 39.28371 ,
                                   47.140452, 54.997194, 62.853936,
                                   70.710678])
        AAAE(actual_rkes[0].ravel(), expected_rkes)
        AAAE(actual_rs[0].ravel(), expected_rs)
        
        
    def test_panel_Regression_Prep(self):
        
        """test the pandas dataframe assembly produces the 
        4 column results as desired.
            
        Generate the same model 10 times but have different 
        ids and splits we want to test
        
        Assemble into a frame
        
        Compare the frame with a test frame from RK_csvs
        """
        
        n = 10
        models = self.ssRK.generate_Model_Sample(n,
                                                 self.model_params)
        rkes, rs = self.ssRK.ripleysKE(models, 
                                                     self.model_params)
        actual_data = self.ssRK.panel_Regression_Prep(rkes, rs, 0)
        
        expected_data = self.ssRK.load_Frame("RK_csvs/test_panel_regression_prep.csv")

        pd.util.testing.assert_frame_equal(actual_data, expected_data)
        
        
    def test_ssRK_Main(self):
        
        """Integration test for ssRK main. hard to assert with highly 
        random case so just assert True if it runs for now.
        
        perhaps fix this later by integrating the R testing as well
        to get a nice True/False answer.
        """
        
        #change to very low pop total for speed
        self.model_params["pop_total"] = 5 
        n_test_runs = 10
        #no seeding this time
        self.model_params["random_seed"] = None
        test_models = self.ssRK.generate_Model_Sample(n_test_runs,
                                                 self.model_params)
        
        self.ssRK.main(test_models, self.model_params)
        

if __name__ == "__main__":
    unittest.main()