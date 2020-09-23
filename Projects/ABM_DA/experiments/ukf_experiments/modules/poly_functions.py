#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:15:49 2019

@author: rob
"""
import numpy as np
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep

def poly_count(poly_list,points):
    """ counts how many agents in each closed polygon of poly_list
    
    - use shapely polygon functions to count agents in each polygon
    
    Parameters
    ------
    poly_list : list
        list of closed polygons over StationSim corridor `poly_list`
    points : array_like
        flatten list of positions. every even index is an x coordinate
        every odd index is a y coordinate
    
    Returns
    ------
    counts : array_like
        `counts` how many agents in each polygon
    """
    
    # list of counts.
    counts = []
    #convert points to 2xn array
    points = np.array([points[::2],points[1::2]]).T
    #make multipoints class of points
    points =MultiPoint(points)
    
    #loop over each poly and count the points in it
    for poly in poly_list:
        poly = prep(poly)
        counts.append(int(len(list(filter(poly.contains,points)))))
    return counts

def grid_poly(width, height, bin_size):
    """generates complete grid of tesselating square polygons covering corridor in station sim.
   
    Parameters
    -----
    width, height : float
        `width` and `height` of StationSim corridor. 
    
    bin_size : float
     size of grid squares. larger implies fewer squares `bin_size`
     
    Returns
    ------
    polys : list
        list of closed square polygons `polys`
    """
    # placeholder list
    polys = []
    # loop over number of squares widthwise
    for i in range(int(width/bin_size)):
        #loop over number of squares heightwise
        for j in range(int(height/bin_size)):
            
            # get 4 corners of current square
            bl = [x*bin_size for x in (i,j)]
            br = [x*bin_size for x in (i+1,j)]
            tl = [x*bin_size for x in (i,j+1)]
            tr = [x*bin_size for x in (i+1,j+1)]
            
            # add square polygon with these 4 polygons to polys
            polys.append(Polygon((bl,br,tr,tl)))
    return polys

