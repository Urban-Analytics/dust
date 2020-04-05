#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:00:50 2020

@author: medrclaa
"""
import os
"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")
    
import numpy as np
from numpy.testing import assert_array_equal as aae
import unittest
import sys
import math


from modules.ukf_ex1 import omission_index, hx1, ex1_main, obs_key_func
from modules import ukf_fx



class Test_ex1(unittest.TestCase): 
    
    """tests for 1st ukf experiment module
    
    - test fx/hx and their component functions
    - test obs key function
    - test parameter dictionary builder  and pickle name
    - test plots?
    - test ex1_main?
    - test arc function?
    """
    
    @classmethod
    def setUpClass(cls):
        
        
        """ dummy initialisation
        """
        
        pass
    
    def test_omission_index_full_prop(self):

        """test omission index function for experiment 1 works given prop = 1
        
        This funciton simply assigns a subset of agents to be observed or
        not in a random fashion.
    
        With prop = 1 eery agent is observed so full indices arrays of length
        5 and 10 should be output.
        """
        
        n  = 5    
        prop1 = 1
        index ,index2 = omission_index(n, int(math.floor(n*prop1)))
        ex_index = np.array([0, 1, 2, 3, 4])
        ex_index2 = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        
        aae(index,ex_index)
        aae(index2,ex_index2)
        
        
    def test_omission_index_half_prop(self):
        
        
        """test omissoin index takes a subset when prop<1\\
        
        should return a subset of the above full arrays.
        In this case the 0th and 2nd agents are always chosen
        due to seeding.
        """
        
        n=5
        prop2 = 0.5
        np.random.seed(8**8)
        index, index2 = omission_index(n,int(math.floor(n*prop2)))
        
        ex_index = np.array([0,  2])
        ex_index2 = np.array([0, 1, 4, 5])
        
        aae(index,ex_index)
        aae(index2,ex_index2)
    
        self.index_half = index
        
    def test_omission_index_no_prop(self):
        
        
        """test omission index takes a subset when prop=0.
        
        should return two empty numpy arrays
        """
        
        n=5
        prop2 = 0.0
        index, index2 = omission_index(n,int(math.floor(n*prop2)))
        
        ex_index = np.array([])
        ex_index2 = np.array([])
        
        aae(index,ex_index)
        aae(index2,ex_index2)
        
    def test_hx1_full_prop(self):
        
        
        """ test experiment 1 measurement function return same state for full prop
        
        should return the fully observed state
        """
        
        n=5
        prop2 = 1.0
        index, index2 = omission_index(n,int(math.floor(n*prop2)))
        
        x = np.arange(10)
        actual = hx1(x, index2 = index2)
        
        expected = x
        aae(actual, expected)
        
    def test_hx1_half_prop(self):
        
        
        """test hx1 when some proportion 0<p<1 is observed
        
        
        should return a subset of the full state which is observed.
        """
        
        n=5
        prop2 = 0.5
        np.random.seed(8**8)
        index, index2 = omission_index(n,int(math.floor(n*prop2)))
        x = np.arange(10)
        
        expected = np.array([0,1,4,5])
        actual = hx1(state = x, index2 = index2)
        aae(actual, expected)
            
    def test_hx1_no_prop(self):
        
        
        """ test if hx1 works when prop = 0
        
        should return an empty array as nothing is observed
        """
        
        n=5
        prop2 = 0.0
        index, index2 = omission_index(n,int(math.floor(n*prop2)))
        
        x = np.arange(10)
        actual = hx1(x, index2 = index2)
        
        expected = np.array([])
        aae(actual, expected)
        
class Test_Recall1(unittest.TestCase):
    
    
    """Test whether the recall functions for each experiment run properly 
    and produce their plots (NOT ANIMATIONS as it would take forever.).
    
    """
    
    @classmethod
    def setUpClass(cls):
        
        
        cls.pickle_source = "pickles/" #where to load/save pickles from
        cls.destination = "plots/"
        cls.n = 5 #population size
        cls.prop = 0.5 #proportion observed
        cls.recall = True #recall previous run
        cls.do_pickle = False #pickle new run

        
    def test_ex1_recall(self):
        
        
        """
        test whether an experiment 1 pickle can be succesfully recalled
        and its plots generated
        """

        ex1_main(self.n, self.prop, self.recall, self.do_pickle, self.pickle_source,
                 self.destination)
  

if __name__ == "__main__":
    test_recall1 = Test_Recall1.setUpClass()
    test_ex1 = Test_ex1()
    unittest.main()