#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:17:15 2020

@author: medrclaa
"""

from stationsim_model import Model
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

"import pickle from ukf for easy saving of results."
from ukf2 import pickler, depickler

import pandas as pd
from esda import gamma
import pysal

import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.append("../experiments/ukf_experiments/modules")
from poly_functions import poly_count, grid_poly
from ukf_plots import CompressionNorm

class HiddenPrints:
    
    
    """stop repeating printing from stationsim 
    We get repeats of `iterations : X` prints as it jumps back 
    and forth over every 100th step. This stops that.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def generate_model_sample(n_runs, model_params, seed = [None]):

    
    """ function for generating stationsim model runs to test
    
    Parameters
    ------
    n_runs : int
        `n_runs`number of stationsim runs to generate. 
        Must be positive intiger.
        
    model_params : dict
        `model_params` dictionary of model parameters  required for 
        stationsim to run. See stationsim_model.py for more details.
        
    seed : list
        `seed` seeding for stationsim. Can a single seed for every run
        or a list of n_runs seeds. 
    """
    
    models = []
    
    if len(seed) == 1 :
        seed *= n_runs
    elif len(seed) != n_runs:
        print("not enough seeds specificed. Either provide precisely 1 or " 
              + f"{n_runs} seeds")
    
    for _ in range(n_runs):
        model = Model(**model_params)
        while model.status == 1:
            model.step()
        models.append(model)
    
    return models

class stationsim_Gamma_Indexing():
    
    
    """ class for local spatial correlation on aggregated stationsim collisions.
    
    """
    
    def __init__(self, model_params):
        
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        
        self.bin_size = self.height/5
        self.polygons = grid_poly(self.width, self.height, self.bin_size)
        
        
        
    def aggregate_collisions(self, model):
        return poly_count(self.polygons, np.array(model.history_collision_locs).ravel())
    
    def collisions_heatmap(self, collisions):
        """main heatmap plot
        
        -define custom compression colourmap
        -
        
        
        Parameters
        ------ 
        truths : array_like
            `truth` true positions
            
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """   
        
        """Setting up custom colour map. defining bottom value (0) to be black
        and everything else is just cividis
        """
        
        cmap = cm.cividis
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0.0,0.0,0.0,1.0)
        cmap = col.LinearSegmentedColormap("custom_cmap",cmaplist,N=cmap.N)
        cmap = cmap.from_list("custom",cmaplist)
        
        """
        TLDR: compression norm allows for more interesting colouration of heatmap
        
        For a large number of grid squares most of the squares only have 
        a few agents in them ~5. If we have 30 agents this means roughly 2/3rds 
        of the colourmap is not used and results in very boring looking graphs.
        
        In these cases I suggest we `move` the colourbar so more of it covers the bottom
        1/3rd and we get a more colour `efficient` graph.
        
        n_prop function makes colouration linear for low pops and large 
        square sizes but squeezes colouration into the bottom of the data range for
        higher pops/low bins. 
        
        This squeezing is proportional to the sech^2(x) = (1-tanh^2(x)) function
        for x>=0.
        (Used tanh identity as theres no sech function in numpy.)
        http://mathworld.wolfram.com/HyperbolicSecant.html
        
        It starts near 1 so 90% of the colouration initially covers 90% of 
        the data range (I.E. linear colouration). 
        As x gets larger sech^2(x) decays quickly to 0 so 90% of the colouration covers
        a smaller percentage of the data range.

        E.g if x = 1, n = 30, 30*0.9*sech^2(1) ~= 10 so 90% of the colouration would be 
        used for the bottom 10 agents and much more of the colour bar would be used.

        There's probably a nice kernel alternative to sech
        """
        
        n= np.max(collisions)
        n_prop = n*(1-np.tanh(2)**2)
        norm =CompressionNorm(1e-5,0.9*n_prop,0.1,0.1,1e-8,n)

        sm = cm.ScalarMappable(norm = norm,cmap=cmap)
        sm.set_array([])  
        
        counts = collisions
        
        f = plt.figure(figsize=(24,16))
        ax = f.add_subplot(111)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.05)
        "plot density histogram and locations scatter plot assuming at least one agent available"
        #ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
        ax.set_ylim(0,self.height)
        ax.set_xlim(0,self.width)
        
        
        
        #column = frame["counts"].astype(float)
        #im = frame.plot(column=column,
        #                ax=ax,cmap=cmap,norm=norm,vmin=0,vmax = n)
   
        patches = []
        for item in self.polygons:
           patches.append(mpatches.Polygon(np.array(item.exterior),closed=True))
        collection = PatchCollection(patches,cmap=cmap, norm=norm, alpha=1.0, edgecolor="w")
        ax.add_collection(collection)

        "if no agents in model for some reason just give a black frame"
        if np.nansum(counts)!=0:
            collection.set_array(np.array(counts))
        else:
            collection.set_array(np.zeros(np.array(counts).shape))

        for k,count in enumerate(counts):
            plt.plot
            ax.annotate(s=count, xy=self.polygons[k].centroid.coords[0], 
                        ha='center',va="center",color="w")
        
        "set up cbar. colouration proportional to number of agents"
        ax.text(0,self.height+1,s="Total Collisions: " + str(np.sum(counts)),color="k")
        
        
        cbar = plt.colorbar(sm,cax=cax,spacing="proportional")
        cbar.set_alpha(1)
        #cbar.draw_all()
        
        "set legend to bottom centre outside of plot"
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        "labels"
        ax.set_xlabel("Corridor width")
        ax.set_ylabel("Corridor height")
        #ax.set_title("Agent Densities vs True Positions")
        cbar.set_label(f"Collision Counts")

    def gamma_indexing(self, values, weights):
        
        
        """perform gamma indexing on some list of polygons, counts, and spatial weighting
        """
        g = pysal.Gamma(values, weights, operation = "s", standardize= "no")
        return g
    
    def rook_Weights(self):
        
        w = pysal.lib.weights.Rook(self.polygons)
        return w
        
if __name__ == "__main__":
    
    model_params = {

    'width': 400,
    'height': 50,
    'pop_total': 25,
    
    'gates_in': 3,
    'gates_out': 2,
    'gates_space': 1,
    'gates_speed': 0.001,
    
    'speed_min': .2,
    'speed_mean': 1,
    'speed_std': 1,
    'speed_steps': 3,
    
    'separation': 5,
    'max_wiggle': 1,
    
    'step_limit': 3600,
    
    'do_history': True,
    'do_print': True,
    }
    
    sGI = stationsim_Gamma_Indexing(model_params)
    
    model = generate_model_sample(1, model_params)
    
    collision_times = []
    collision_times_array = np.array(model[0].history_collision_times)
    for i in range(model[0].step_id):
        collision_times.append(np.sum(collision_times_array==i))
        
    plt.plot(range(model[0].step_id), collision_times)
    collisions = sGI.aggregate_collisions(model[0])
    
    sGI.collisions_heatmap(collisions)
    


