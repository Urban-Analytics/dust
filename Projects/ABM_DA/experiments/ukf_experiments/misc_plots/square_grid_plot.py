
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019

@author: RC

first attempt at a square root UKF class
class built into 5 steps
-init
-Prediction SP generation
-Predictions
-Update SP generation
-Update

UKF filter using own function rather than filterpys

based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
"""

#import pip packages

import sys #for print suppression#
sys.path.append("../dust/Projects/ABM_DA")
from stationsim.stationsim_model import Model
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import datetime
import multiprocessing
from copy import deepcopy
import os #for animations folder handling
from math import ceil,log10

import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep
import geopandas as gpd

#for dark plots. purely an aesthetic choice.

"""
suppress repeat printing in F_x from new stationsim
E.g. 
with HiddenPrints():
    everything done here prints nothing

everything here prints again
https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
"""


def grid_poly(width,length,bin_size):
    """
    generates grid of aggregate square polygons for corridor in station sim.
    UKF should work with any list of connected simple polys whose union 
    lies within space of interest.
    This is just an example poly that is nice but in theory works for any.
    !!potentially add randomly generated polygons or camera circles/cones.
    
    in:
        corridor parameters and size of squares
    out: 
        grid of squares for aggregates
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

poly_list = grid_poly(200,100,25) #generic square grid over corridor
f = plt.figure()
for poly in poly_list:
    x,y = poly.exterior.xy
    plt.plot(x,y,linewidth=3)
plt.xlabel("Corridor Width")
plt.ylabel("Corridor Length")
plt.title("Grid of 32 25x25 Aggregated Squares Over a 200x100 Corridor")
plt.savefig("Square_Grid_Plot.pdf")