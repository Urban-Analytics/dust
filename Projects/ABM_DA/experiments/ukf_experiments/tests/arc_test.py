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

import unittest
import os

"""
run file in ukf_experiments. putting test at top level allows the 
large number of 
"""
"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")
    
import arc.arc as arc

from modules.ukf_fx import HiddenPrints

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
        with HiddenPrints():
            arc.main(arc.ex0_input, arc.ex0_save, test=True)
       
        
    def test_ex1(self):
        
        
        """another arc module for experiment 1
        We choose n =5 and proportion observed prop = 0.5
        """
        
        with HiddenPrints():
            arc.main(arc.ex1_input, test=True)

    def test_ex2(self):
        
        
        """another arc module test for experiment 2
        We choose n = 5 and aggregate square size bin_size = 50
        """
        with HiddenPrints():
            arc.main(arc.ex2_input, test=True)

    
if __name__ == '__main__':

    "test the three experiments arc functions are working"
    " each test uses 5 agents and some arbitrary parameters for the sake of speed"
    arc_tests  =Test_arc.setUpClass()
    unittest.main()