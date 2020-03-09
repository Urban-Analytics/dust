#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 15:07:56 2020

@author: medrclaa
"""

import unittest
import numpy as np
from numpy.testing import assert_array_equal as aae

from poly_functions import grid_poly, poly_count
from ukf_ex2 import ex2_pickle_name, ex2_main, hx2, obs_key_func


class Test_poly_functions(unittest.TestCase):
    
    
    """ class for testing shapely functions used in experiment 2
    
    """
    
    @classmethod
    def setUpClass(cls):
        
        
        """
        init unit test class with grid of polygons used for multiple tests
        """
        cls.grid = grid_poly(10,10,5)   

    def test_Grid_Poly(self):
        
        
        """
        test example polygon grid above works
        should be 4 polygons with area 25 and specified coordinates below
        """
        grid = self.grid
        square = grid[0]
        area = sum([shape.area for shape in grid])
        
        assert len(grid) == 4
        assert area == 100.0
        x, y = square.exterior.coords.xy
        coords= np.array([x,y]).T
        expected = np.array([[0., 0.],
                          [5., 0.],
                          [5., 5.],
                          [0., 5.],
                          [0., 0.]])
        aae(coords, expected)
    
    
    def test_Poly_Count(self):
        
        """test output for poly_count function
        
        testing:
            
            - no grid squares empty
            - some grid squares empty
            - all grid squares empty
            - some subset of the grid
        """
        grid = self.grid
        sub_grid = grid[:2]
        
        x = np.array([4,4,4,6,7,7,7,4])
        x2 = np.array([4,4,4,6,4,7])
        x3 = np.array([])
        
        actual1 = poly_count(grid, x)
        actual2 = poly_count(sub_grid, x)
        actual3 = poly_count(grid, x2)
        actual4 = poly_count(grid,x3)
        
        expected1 = np.array([1,1,1,1])
        expected2 = np.array([1,1])
        expected3 = np.array([1, 2, 0, 0])
        expected4 = np.array([0, 0, 0, 0])
        
        aae(actual1, expected1)
        aae(actual2, expected2)
        aae(actual3, expected3)
        aae(actual4, expected4)
    

        
class Test_Recall2(unittest.TestCase):
    
    
    """Test whether the recall functions for each experiment run properly 
    and produce their plots (NOT ANIMATIONS as it would take forever.).
    
    """
    
    @classmethod
    def setUpClass(cls):
        
        cls.pickle_source = "../test_pickles/" #where to load/save pickles from
        cls.bin_size = 25 #population size
        cls.n = 5 #proportion observed
        cls.destination = "../plots/"
        cls.f_name = ex2_pickle_name(cls.n, cls.bin_size)
        
    def test_ex2_recall(self):
        
        
        """
        test whether an experiment 1 pickle can be succesfully recalled
        and its plots generated
        """
        passed = True
        try:
            ex2_main(self.n, self.bin_size, True, True, self.pickle_source, self.destination)
        except:
            passed = False
        
        self.assertTrue(passed, "ex2 recall failed")
        
if __name__ == "__main__":
    test_polys = Test_poly_functions.setUpClass()
    test_recall2 = Test_Recall2.setUpClass()
    unittest.main()
    