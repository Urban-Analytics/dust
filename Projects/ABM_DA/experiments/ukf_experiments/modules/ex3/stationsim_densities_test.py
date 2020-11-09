#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:06:27 2020

@author: medrclaa
stationsim densities test
"""

import unittest
import numpy as np
from numpy.testing import assert_array_almost_equal as AAAE
from stationsim_densities import exit_points_to_polys, start_gate_heading, boundary_intersects
from stationsim_densities import cut_boundaries, gates_cut_boundaries, plot_vision
import sys
sys.path.append("..")
from sensors import generate_Camera_Rect

class Test_stationsim_densities(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        
        """Custom __init__ for unittest. Blame them not me.
        
        """
        #build boundary polygon
        height = 50
        width = 50
        cls.boundary = generate_Camera_Rect(np.array([0, 0]), 
                                    np.array([0, height]),
                                    np.array([width, height]), 
                                    np.array([width, 0]))
        #where test agents come from
        cls.start_position = np.ravel(np.array([[25, 0],
                                  [0, 0],
                                  [0, 25],
                                  [0, 50],
                                  [25, 50],
                                  [50, 50],
                                  [50, 25],
                                  [50, 0]]))
        
        #all agents positioned in the middle to get a cone facing the opposite
        #edge to the start
        cls.positions = np.ravel(np.array([[25, 25],
                             [25, 25],
                             [25, 25],
                             [25, 25],
                             [25, 25],
                             [25, 25],
                             [25, 25],
                             [25, 25]]))
        #where are the test exit gates centroids
        cls.exit_gates =  np.array([[25, 0],
                              [50, 25],
                              [25, 50],
                              [0, 25],
                              [5, 0],
                              [0, 5],
                              [0, 45],
                              [5, 50],
                              [45, 50],
                              [50, 45],
                              [45, 0],
                              [50, 5]])       
        #build circular polygons out of centroids
        cls.exit_polys = exit_points_to_polys(cls.exit_gates, cls.boundary, 1)
        #which direction are agents heading given start position and position
        cls.angles = start_gate_heading(cls.start_position, cls.positions)
        
    
    def test_sub_90(self):
        """ test an agent with a field of vision <90 in all 8 directions

        I.E test the basic visions work on a flat plane and on a corner.
        There are eight tests in a star pattern testing the probabilities are
        correctly calculated on each flat edge and over one corner only.
        
        Exit gates are built such that we have one in middle of each square edge
        and one either side of corner.
        """
        
        ""
        #starts go clockwise from south. I.E agents exiting roughly clockwise
        #from north
        
        polys = boundary_intersects(self.positions, self.angles,
                                    ((np.pi/8))-0.1, self.boundary)
        cut_bounds = cut_boundaries(polys, self.boundary)
        boundary_gate_lengths = gates_cut_boundaries(self.exit_polys, cut_bounds)
        
        expected_gate_length = np.array([[0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                                           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. ],
                                           [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                                           [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5],
                                           [1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                                           [0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. , 0. , 0. , 0. ],
                                           [0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                                           [0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0.5, 0. , 0. , 0. , 0. ]])
        AAAE(boundary_gate_lengths,expected_gate_length)
        #plot_vision(cut_bounds, self.exit_polys, polys, self.boundary)

        
    def test_sub_180(self):
        """ test an agent with a field of vision <90 in all 8 directions

        I.E test the basic visions work on a flat plane and on a corner
        """
        
        polys = boundary_intersects(self.positions, self.angles,
                                    ((np.pi/3))-0.1, self.boundary)
        cut_bounds = cut_boundaries(polys, self.boundary)
        boundary_gate_lengths = gates_cut_boundaries(self.exit_polys, cut_bounds)
        
        expected_gate_length = np.array([[0.  , 0.  , 0.2 , 0.  , 0.  , 0.  , 0.2 , 0.2 , 0.2 , 0.2 , 0.  ,
        0.  ],
       [0.  , 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.25, 0.25, 0.  ,
        0.  ],
       [0.  , 0.2 , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.2 , 0.2 , 0.2 ,
        0.2 ],
       [0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.  , 0.25,
        0.25],
       [0.2 , 0.  , 0.  , 0.  , 0.2 , 0.2 , 0.  , 0.  , 0.  , 0.  , 0.2 ,
        0.2 ],
       [0.25, 0.  , 0.  , 0.25, 0.25, 0.25, 0.  , 0.  , 0.  , 0.  , 0.  ,
        0.  ],
       [0.  , 0.  , 0.  , 0.2 , 0.2 , 0.2 , 0.2 , 0.2 , 0.  , 0.  , 0.  ,
        0.  ],
       [0.  , 0.  , 0.25, 0.25, 0.  , 0.  , 0.25, 0.25, 0.  , 0.  , 0.  ,
        0.  ]])
        
        AAAE(boundary_gate_lengths,expected_gate_length)
        #plot_vision(cut_bounds, self.exit_polys, polys, self.boundary)
    
    def test_sub_270(self):
        """ test an agent with a field of vision <270 in all 8 directions

        I.E test the basic visions work on a flat plane and on a corner
        """    
        
        polys = boundary_intersects(self.positions, self.angles,
                                    ((5*np.pi/8))-0.1, self.boundary)
        cut_bounds = cut_boundaries(polys, self.boundary)
        boundary_gate_lengths = gates_cut_boundaries(self.exit_polys, cut_bounds)
        
        expected_gate_length = np.array([[0.        , 0.14285714, 0.14285714, 0.14285714, 0.        ,
        0.        , 0.14285714, 0.14285714, 0.14285714, 0.14285714,
        0.        , 0.        ],
       [0.        , 0.125     , 0.125     , 0.        , 0.        ,
        0.        , 0.125     , 0.125     , 0.125     , 0.125     ,
        0.125     , 0.125     ],
       [0.14285714, 0.14285714, 0.14285714, 0.        , 0.        ,
        0.        , 0.        , 0.        , 0.14285714, 0.14285714,
        0.14285714, 0.14285714],
       [0.125     , 0.125     , 0.        , 0.        , 0.125     ,
        0.125     , 0.        , 0.        , 0.125     , 0.125     ,
        0.125     , 0.125     ],
       [0.14285714, 0.14285714, 0.        , 0.14285714, 0.14285714,
        0.14285714, 0.        , 0.        , 0.        , 0.        ,
        0.14285714, 0.14285714],
       [0.125     , 0.        , 0.        , 0.125     , 0.125     ,
        0.125     , 0.125     , 0.125     , 0.        , 0.        ,
        0.125     , 0.125     ],
       [0.14285714, 0.        , 0.14285714, 0.14285714, 0.14285714,
        0.14285714, 0.14285714, 0.14285714, 0.        , 0.        ,
        0.        , 0.        ],
       [0.        , 0.        , 0.125     , 0.125     , 0.125     ,
        0.125     , 0.125     , 0.125     , 0.125     , 0.125     ,
        0.        , 0.        ]])
        
        AAAE(boundary_gate_lengths,expected_gate_length)
        plot_vision(cut_bounds, self.exit_polys, polys, self.boundary)
    
    def test_sub_360(self):
        """ test an agent with a field of vision <360 in all 8 directions

        I.E test the basic visions work on a flat plane and on a corner
        """    
        
        polys = boundary_intersects(self.positions, self.angles,
                                    ((np.pi))-0.1, self.boundary)
        cut_bounds = cut_boundaries(polys, self.boundary)
        boundary_gate_lengths = gates_cut_boundaries(self.exit_polys, cut_bounds)
        
        expected_gate_length = np.array([[0.        , 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909],
       [0.08740708, 0.08740708, 0.08740708, 0.08740708, 0.06296458,
        0.06296458, 0.08740708, 0.08740708, 0.08740708, 0.08740708,
        0.08740708, 0.08740708],
       [0.09090909, 0.09090909, 0.09090909, 0.        , 0.09090909,
        0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909],
       [0.08740708, 0.08740708, 0.08740708, 0.08740708, 0.08740708,
        0.08740708, 0.06296458, 0.06296458, 0.08740708, 0.08740708,
        0.08740708, 0.08740708],
       [0.09090909, 0.09090909, 0.        , 0.09090909, 0.09090909,
        0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909],
       [0.08740708, 0.08740708, 0.08740708, 0.08740708, 0.08740708,
        0.08740708, 0.08740708, 0.08740708, 0.06296458, 0.06296458,
        0.08740708, 0.08740708],
       [0.09090909, 0.        , 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909, 0.09090909, 0.09090909, 0.09090909,
        0.09090909, 0.09090909],
       [0.08740708, 0.08740708, 0.08740708, 0.08740708, 0.08740708,
        0.08740708, 0.08740708, 0.08740708, 0.08740708, 0.08740708,
        0.06296458, 0.06296458]])
        
        AAAE(boundary_gate_lengths,expected_gate_length)
        plot_vision(cut_bounds, self.exit_polys, polys, self.boundary)
        
if __name__ == "__main__":
    unittest.main()