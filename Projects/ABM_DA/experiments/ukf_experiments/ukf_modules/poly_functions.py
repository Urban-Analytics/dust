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
        list of agent GPS positions to count
    
    Returns
    ------
    counts : array_like
        `counts` how many agents in each polygon
    """
    counts = []
    points = np.array([points[::2],points[1::2]]).T
    points =MultiPoint(points)
    for poly in poly_list:
        poly = prep(poly)
        counts.append(int(len(list(filter(poly.contains,points)))))
    return counts

def grid_poly(width,length,bin_size):
    
    
    """generates complete grid of tesselating square polygons covering corridor in station sim.
   
    Parameters
    -----
    width,length : float
        `width` and `length` of StationSim corridor. 
    
    bin_size : float
     size of grid squares. larger implies fewer squares `bin_size`
     
    Returns
    ------
    polys : list
        list of closed square polygons `polys`
    """
    polys = []
    for i in range(int(width/bin_size)):
        for j in range(int(length/bin_size)):
            bl = [x*bin_size for x in (i,j)]
            br = [x*bin_size for x in (i+1,j)]
            tl = [x*bin_size for x in (i,j+1)]
            tr = [x*bin_size for x in (i+1,j+1)]
            
            polys.append(Polygon((bl,br,tr,tl)))
    "hashed lines for plots to verify desired grid"
    #for poly in polys:
    #    plt.plot(*poly.exterior.xy)
    return polys

