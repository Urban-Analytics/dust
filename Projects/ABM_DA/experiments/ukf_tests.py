#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 15:33:05 2020

General unit testing for micro level functions in the UKF. The aim is to have:
    - tests for various deterministic functions
    - interaction tests for meso level ABM/UKF mechanisms
    - similar tests for macro levels
    
Also want:
    - image based tests for plotting etc.
    - test for notebook function
    
NOTE: the unittest.testcase has a custom __init__ you cant overwrite so has
setUpClass() class method instead.
"""
import numpy as np
from numpy.testing import assert_array_equal as aae
from numpy.testing import assert_array_almost_equal as aaae
import unittest
import sys

sys.path.append("../stationsim")
from ukf2 import ukf, ukf_ss, pickle_main, MSSP, covariance, unscented_Mean


#%%


def test_fx(state):
    
    
    """simple linear motion transition function for testing
    
    Parameters
    ------
    
    state : array_like
        array of `state` positions

    Returns
    -------
    state : array_like
        array of `state` positions moved up and right by 1
    """
    
    return state + 1

class Test_ukf(unittest.TestCase):
    

    """ Tests for UKF micro level functions:
    
    - MSSP
    - unscented_Mean
    - covariance
    """
    
    @classmethod
    def setUpClass(cls):
        
        
        """
        initiate various dummy variables.
        """
        
        cls.x = np.array([1, 1, 1, 7])
        cls.n = int(cls.x.shape[0]/2)
        cls.p = np.eye(2*cls.n)/4
        cls.g = 1
        cls.wm = np.ones(cls.n*4 + 1)/(cls.n*4+1)
        cls.wc = cls.wm
        
    def test_MSSP(self):
        
        "Test Merwe's Scaled Sigma Points (MSSP) function  with dummy positions."
        
        
        actual = MSSP(self.x, self.p ,self.g)  
        expected_sigmas = np.array([[1. , 1.5, 1. , 1. , 1. , 0.5, 1. , 1. , 1. ],
                                   [1. , 1. , 1.5, 1. , 1. , 1. , 0.5, 1. , 1. ],
                                   [1. , 1. , 1. , 1.5, 1. , 1. , 1. , 0.5, 1. ],
                                   [7. , 7. , 7. , 7. , 7.5, 7. , 7. , 7. , 6.5]])
                                
        aae(actual, expected_sigmas, "Failure by Merwe Scaled Sigma Point Function MSSP")       

    def test_Unscented_Mean(self):
        
        
        """ test the unscented mean with test fx function and sigmasd
        """
        
        sigmas = MSSP(self.x, self.p, self.g)
        actual_nlsigmas , actual_xhat = unscented_Mean(sigmas, self.wm, test_fx)
        
        expected_nlsigmas = np.array([[2. , 2.5, 2. , 2. , 2. , 1.5, 2. , 2. , 2. ],
                                       [2. , 2. , 2.5, 2. , 2. , 2. , 1.5, 2. , 2. ],
                                       [2. , 2. , 2. , 2.5, 2. , 2. , 2. , 1.5, 2. ],
                                       [8. , 8. , 8. , 8. , 8.5, 8. , 8. , 8. , 7.5]])
        expected_xhat = np.array([2., 2., 2., 8.])
        
        aaae(actual_nlsigmas,expected_nlsigmas)
        aaae(actual_xhat, expected_xhat)

    def test_covariance_within(self):
        
        
        """test cross covariance between one array and mean
        """
        
        sigmas = MSSP(self.x, self.p, self.g)
        nlsigmas, xhat = unscented_Mean(sigmas, self.wm, test_fx)
         
        actual = covariance(nlsigmas,xhat,self.wc)
        expected = np.array([[0.05555556, 0.        , 0.        , 0.        ],
                               [0.        , 0.05555556, 0.        , 0.        ],
                               [0.        , 0.        , 0.05555556, 0.        ],
                               [0.        , 0.        , 0.        , 0.05555556]])
        aaae(actual, expected)
        
    def test_covariane_cross(self):
        
        
        """test cross covariance with two different arrays and 
        unscented means
        """
        sigmas = MSSP(self.x, self.p, self.g)
        nlsigmas , xhat = unscented_Mean(sigmas, self.wm, test_fx)
        actual = covariance(nlsigmas,xhat,self.wc,sigmas[:2,:],self.x[:2],np.ones((4,2)))
        
        expected = np.array([[1.05555556, 1.        ],
                           [1.        , 1.05555556],
                           [1.        , 1.        ],
                           [1.        , 1.        ]])

        aaae(actual, expected)
        
if __name__ == "__main__":
    
    
    """
    Initaties the above classes and runs the tests within
    using unitest.main
    """
    
    ukf_tests = Test_ukf.setUpClass()        
    unittest.main()
    

    