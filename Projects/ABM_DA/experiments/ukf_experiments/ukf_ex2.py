#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:19:48 2019

@author: rob
"""
import sys
sys.path.append("..")
from ukf_experiments.ukf_fx import fx

import numpy as np
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep

def poly_count(poly_list,points):
    
    
    """ counts how many agents in each closed polygon of poly_list
    
    -    use shapely polygon functions to count agents in each polygon
    
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


def obs_key_func(state,model_params,ukf_params):
        """which agents are observed"""
        
        key = np.ones(model_params["pop_total"])
        
        return key

def aggregate_params(model_params, ukf_params, bin_size):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 2
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    
    n = model_params["pop_total"]
    
    ukf_params["bin_size"] = bin_size
    ukf_params["poly_list"] = grid_poly(model_params["width"],
              model_params["height"],ukf_params["bin_size"])
        
    ukf_params["p"] = np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2*n)
    ukf_params["r"] = np.eye(len(ukf_params["poly_list"]))#sensor noise 
    
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx2
    
    
    
    ukf_params["obs_key_func"] = obs_key_func
    ukf_params["pickle_file_name"] = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"    
    
    
    return ukf_params
    
def hx2(state,model_params,ukf_params):
        """Convert each sigma point from noisy gps positions into actual measurements
        
        -   uses function poly_count to count how many agents in each closed 
            polygon of poly_list
        -   converts perfect data from ABM into forecasted 
            observation data to be compared and assimilated 
            using actual observation data
        
        Parameters
        ------
        state : array_like
            desired `state` n-dimensional sigmapoint to be converted
        
        **hx_args
            generic hx kwargs
        Returns
        ------
        counts : array_like
            forecasted `counts` of how many agents in each square to 
            compare with actual counts
        """
        counts = poly_count(ukf_params["poly_list"],state)
        
        return counts