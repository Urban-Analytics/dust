#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 09:50:10 2019

@author: rob

!!rewrite this 

Produces more generalised diagnostic over multiple runs using multiple
numbers of agents for arc_ukf.py only. This produces a chloropleth style map 
showing the grand median error over both time and agents for various fixed numbers of agents 
and proportions observed.


import data from arc with following in bash terminal
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
change to relevant directories

"""

import pickle
import sys
#sys.path.append("../../stationsim")
sys.path.append("../..")

from ukf_plots import L2s as L2_parser

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.patheffects as pe

import glob
import seaborn as sns
import pandas as pd


#%%
class grand_plots:
    
    
    """class for results of multiple ukf runs
    
    """

    def __init__(self,params,save):
        self.param_keys = [key for key in params.keys()]
        self.p1 = params[self.param_keys[0]]
        self.p2 = params[self.param_keys[1]]
        self.source = params["source"]
        self.save = save
    def depickle_data_parser(self,instance):
        
        
        """simplified version of ukf.data_parser. just pulls truths/preds
        
        Returns
        ------
            
        truth : array_like
            `a` noisy observations of agents positions
        preds : array_like
            `b` ukf predictions of said agent positions
        """
        
        """pull actual data. note a and b dont have gaps every sample_rate
        measurements. Need to fill in based on truths (d).
        """
        truth =  np.vstack(instance.truths) 
        preds2 = np.vstack(instance.ukf_histories)
        
        "full 'd' size placeholders"
        preds= np.zeros((truth.shape[0],instance.pop_total*2))*np.nan
        
        "fill in every sample_rate rows with ukf estimates and observation type key"
        "!!theres probably an easier way to do this"
        for j in range(int(preds.shape[0]//instance.sample_rate)):
            preds[j*instance.sample_rate,:] = preds2[j,:]
            
        return truth,preds
        
    def frame_extractor(self):
        """pull multiple class runs into pandas frame for analysis
        """
        keys = self.param_keys
        "placeholder dictionary for all parameters" 
        L2 = {}
        for i in self.p1:
            "file names for all files with parameter 1 value i"
            files={}
            "loop over second parameter to load in all files for given value of i"
            for j in self.p2:
                f_name = self.source + f"{keys[0]}_{i}_{keys[1]}_{j}-*"
                files[j] = glob.glob(f_name)
                
            "sub dictionary for each second parameter"       
            L2[i] = {} 
            for _ in files.keys():
                "collect all individual UKF run error metrics into a list"
                L2_2=[]
                for file in files[_]:
                    f = open(file,"rb")
                    u = pickle.load(f)
                    f.close()
                    truth, preds = self.depickle_data_parser(u)
                    distances = L2_parser(truth[::u.sample_rate,:], preds[::u.sample_rate,:])
                    
                    L2_2.append(np.nanmedian(np.nanmean(distances,axis=0)))
    
                L2[i][_] = np.hstack(L2_2)
                
        sub_frames = []
    
        for i in self.p1:
            for j in self.p2:
                L2s = L2[i][j]
                sub_frames.append(pd.DataFrame([[i]*len(L2s),[j]*len(L2s),L2s]).T)
    
        "stack into grand frames and label columns"
        error_frame = pd.concat(sub_frames)
        error_frame.columns = [keys[0], keys[1], "L2 agent errors"]
    
        self.error_frame = error_frame

    def choropleth_array(self):
        
        
        """converts pandas frame into generalised numpy array for choropleth
        
        """
        error_frame2 = self.error_frame.groupby(by =[str(self.param_keys[0]),str(self.param_keys[1])]).mean()
        error_array = np.ones((len(self.p1),len(self.p2)))*np.nan
        
        for  i, x  in enumerate(self.p1):
            for  j, y in enumerate(self.p2):
                error_array[i,j] = error_frame2.loc[(x,y),][0]
    
        self.error_array = error_array

    def choropleth_plot(self):
       
        
        """choropleth style plot for grand medians
        
        Parameters
        ------
        data : array_like
            L2 `data` matrix from grand_L2_matrix
        n,bin_size : float
            population `n` and square size `bin_size`
        save : bool
            `save` plot?
    
        """    
        "rotate frame 90 degrees so population on x axis"
        data = np.rot90(self.error_array,k=1) 
        keys = self.param_keys
        "initiate plot"
        f,ax=plt.subplots(figsize=(8,8))
        "colourmap"
        cmap = cm.viridis
        "set nans for 0 agents unobserved to white (not black because black text)"
        cmap.set_bad("white") 
        
        " mask needed to get bad white squares in imshow"
        data2 = np.ma.masked_where(np.isnan(data),data)
        "rotate again so imshow right way up (origin bottom left i.e. lower)"
        data2=np.flip(data2,axis=0) 
        im=ax.imshow(data2,interpolation="nearest",cmap=cmap,origin="lower")
        
        "labelling"
        ax.set_xticks(np.arange(len(self.p1)))
        ax.set_yticks(np.arange(len(self.p2)))
        ax.set_xticklabels(self.p1)
        ax.set_yticklabels(self.p2)
        ax.set_xticks(np.arange(-.5,len(self.p1),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(self.p2),1),minor=True)
        ax.grid(which="minor",color="k",linestyle="-",linewidth=2)
        ax.set_xlabel("Number of Agents")
        ax.set_ylabel("Aggregate Grid Squre Size")
        #plt.title("Grand L2s Over Varying Agents and Percentage Observed")
    
    
        "text on top of squares for clarity"
        data = np.flip(data,axis=0)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                plt.text(j,i,str(data[i,j].round(2)),ha="center",va="center",color="w",
                         path_effects=[pe.Stroke(linewidth = 0.7,foreground='k')])
                
        "colourbar alignment and labelling"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.05)
        cbar=plt.colorbar(im,cax,cax)
        cbar.set_label("Grand Mean L2 Error")
        
        "further labelling and saving"
        cbar.set_label("Aggregate Median L2s")
        ax.set_ylabel("Aggregate Grid Squre Width")
        
        if self.save:
            plt.savefig("Aggregate_Grand_L2s.pdf")
        
    def boxplot(self):
        """produces grand median boxplot for all 30 ABM runs for choropleth plot
        
           
        Parameters
        ------
        frame : array_like
            L2 `data` matrix from grand_L2_matrix
        n,bin_size : float
            population `n` and square size `bin_size`
        save,seperate : bool
            `save` plot?
            `seperate` box plots by population or have one big catplot?
        
        """       
        keys = self.param_keys
        data = self.error_frame
        f_name = f"Aggregate_grand_median_boxplot.pdf"
        y_name = "L2 agent errors"
        f = plt.figure()
        sns.catplot(x=str(keys[1]),y=y_name,col=str(keys[0]),kind="box", data=data)
        plt.tight_layout()
        if self.save:
            plt.savefig(f_name)
       
        
        
        
"""parameter dictionary
parameter1: usually number of "agents"

parameter2: proportion observed "prop" or size of aggregate squares "bin" 
    int 1 here because I named the results files poorly.!! fix later using grep?"

source : "where files are loaded from plus some file prefix such as "ukf" or "agg_ukf" e.g. dust repo or a USB


"""
depickle_params = {
        "agents" :  [10,20,30],
        #"prop" : [0.25, 0.5, 0.75, int(1)],
        "bin" : [5,10,25,50],
        #"source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
        "source" : "/media/rob/ROB1/ukf_results_100_2/agg_ukf_",
        }

"init plot class"
g_plts = grand_plots(depickle_params,True)
"make frame"
g_plts.frame_extractor()
"make choropleth numpy array"
g_plts.choropleth_array()
"make choropleth"
g_plts.choropleth_plot()
"make boxplot"
g_plts.boxplot()