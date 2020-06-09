#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 20:12:18 2020

@author: medrclaa
"""


import unittest
import numpy as np


from arc_ukf_gcs_ex0 import arc_ex0_main
from arc_ukf_gcs_ex1 import arc_ex1_main
from arc_ukf_gcs_ex2 import arc_ex2_main

class Test_arc_ukf(unittest.TestCase):
    
    def test_ex0(self):
        """test experiment 0 runs on arc using some quick parameters

        """
        test = True
        if test:
            print("Test set to true. If you're running an experiment, it wont go well.")
            
        # Lists of parameters to vary over
        n = 10 # 10 to 30 agent population by 10
        sample_rate = [1, 2, 5, 10]  # assimilation rates 
        noise = [0, 0.25, 0.5, 1, 2, 5] #gaussian observation noise standard deviation
        run_id = np.arange(0, 30, 1)  # 30 repeats for each combination of the above parameters
    
        # Assemble lists into grand list of all combinations. 
        # Each experiment will use one item of this list.
        parameter_lists = [(n, x, y, z)
                      for x in sample_rate for y in noise for z in run_id]
        arc_ex0_main(n, parameter_lists, test)
        
    def test_ex1(self):
        
        #if testing set to True. if running batch experiments set to False
        test = True
        if test:
            print("Test set to true. If you're running an experiment, it wont go well.")
    
        # agent populations from 10 to 30
        num_age = [10, 20, 30]  
        # 25 to 100 % proportion observed in 25% increments. must be 0<=x<=1
        props = [0.25, 0.5, 0.75, 1]
        # how many experiments per population and proportion pair. 30 by default.
        run_id = np.arange(0, 30, 1)
        #cartesian product list giving all combinations of experiment parameters.
        param_list = [(x, y, z) for x in num_age for y in props for z in run_id]
        
        arc_ex1_main(param_list, test)
        
    def test_ex2(self):
        test = True
        if test:
            print("Test set to true. If you're running an experiment, it wont go well.")
        num_age = [10, 20, 30, 50]  # 10 to 30 agent population by 10
        # unitless grid square size (must be a factor of 100 and 200)
        bin_size = [5, 10, 25, 50]
        run_id = np.arange(0, 30, 1)  # 30 runs
    
        parameter_lists = [(x, y, z) for x in num_age for y in bin_size for z in run_id]
    
        arc_ex2_main(parameter_lists, test)
        
if __name__ == "__main__":
    unittest.main()