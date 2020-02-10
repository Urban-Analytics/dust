#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:31:38 2020

@author: medrclaa


Make sure you have an environment with scipy<=1.2.1. They updated
scipy.spatial.ckdTree and so pickle cant load the experiment data 
with newer versions.

if running in anaconda, I recommend a separate environment for this as a
lot of packages rely on scipy. 


"""

import unittest
from depickle import ex0_grand, ex1_grand, ex2_grand, main



class Test_depickles(unittest.TestCase):
    
    
    """
    Tests to make sure the pickles/numpy loads of finished arc runs are 
    working and producing plots
    """
    
    @classmethod
    def setUpClass(cls):
        pass
    
    def test_ex0_depickle(self):
        passed = True
        try:
            main(ex0_grand, f"/Users/medrclaa/ukf_depickle_test/config*010*", "../plots")
        except:
            passed = False
        self.assertTrue(passed, "ex0 depickle failed")
        
    def test_ex1_depickle(self):
        
        passed = True
        try:
            main(ex1_grand, f"/Users/medrclaa/ukf_depickle_test/ukf*", "../plots")
        except:
            passed = False
        
        self.assertTrue(passed,"ex1 depickle failed")
        
if __name__ == "__main__":
    depickle_tests = Test_depickles.setUpClass()
    
    unittest.main()