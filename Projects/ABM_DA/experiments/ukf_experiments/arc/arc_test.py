#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:07:22 2020

@author: medrclaa

Stand alone script for testing arc in ARC. simply run

pytest arc_test.py

To ensure the working environment is suitable for running experiments.

If you only wish to run a single experiment then you  an easily hash the 
other 2 for quicker testing time. 
"""

import arc
import unittest
import sys

sys.path.append("../ukf_modules")
from ukf_fx import HiddenPrints

class Test_arc(unittest.TestCase):
    
    
    """test the ukf runs for all 3 experiments in arc 
    this is a fairly long test but tests vitually everything runs bar the 
    plotting.
    
    
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_ex0(self):
        
        """run the arc test for the experiment 0 module
        pass the test if the whole arc test completes.
        
        Note that arc_test.py does similar but is actually runnable in 
        arc to check the environment is suitable there. 
        """
        passed = True
        try:
            with HiddenPrints():
                arc.main(arc.ex0_input, arc.ex0_save, test=True)
        except: 
            passed = False
            
        self.assertTrue(passed, "ex0 did not pass")
        
    def test_ex1(self):
        
        
        """another arc module for experiment 1
        We choose n =5 and proportion observed prop = 0.5
        """
        passed = True
        try:
            with HiddenPrints():
                arc.main(arc.ex1_input, test=True)
        except: 
            passed = False
            
        self.assertTrue(passed, "ex1 did not pass")
        
    def test_ex2(self):
        
        
        """another arc module test for experiment 2
        We choose n = 5 and aggregate square size bin_size = 50
        """
        
        passed = True
        try:
            with HiddenPrints():
                arc.main(arc.ex2_input, test=True)
        except:
            passed = False
            
        self.assertTrue(passed, "ex2 did not pass")

    
if __name__ == '__main__':

    "test the three experiments arc functions are working"
    " each test uses 5 agents and some arbitrary parameters for the sake of speed"
    arc_tests  =Test_arc.setUpClass()
    unittest.main()