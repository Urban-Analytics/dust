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

    def __init__(self,params, save, restrict = None, **kwargs):
        """initialise plot class
        
        Parameters
        ------
        params: dict
            `params` dictionary defines 2 parameters to extract data over 
            and a file source.
            
            e.g.
            depickle_params = {
                    "agents" :  [10,20,30],
                    "bin" : [5,10,25,50],
                    "source" : "/media/rob/ROB1/ukf_results/ukf_",
            }
            Searches for files with 10,20, and 30 "agents" (population size) 
            and 5, 10, 25, and 50 experiment 2 square grid size "bin".
            It extracts theres files from the source.
            Please be careful with this as it 
        
        
        """
        self.param_keys = [key for key in params.keys()]
        self.p1 = params[self.param_keys[0]]
        self.p2 = params[self.param_keys[1]]
        self.source = params["source"]
        self.save = save
        self.restrict = restrict
        self.kwargs = kwargs
        
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
        
        nan_array = np.ones(shape = truth.shape,)*np.nan
        for i, agent in enumerate(instance.base_model.agents):
            array = np.array(agent.history_locations)
            index = np.where(array !=None)[0]
            nan_array[index,2*i:(2*i)+2] = 1
        
        return truth*nan_array, preds*nan_array
        
    def data_extractor(self):
        """pull multiple class runs into arrays for analysis
        
        This is function looks awful... because it is. 
        Heres what it does:
            
        - build grand dictionary L2
        - loop over first parameter e.g. population size
            - create sub dictionary for given i L2[i]
            - loop over second parameter e.g. proportion observed (prop)
                - create placeholder list sub_L2 to store data for given i and j.
                - load each ukf pickle with the given i and j.
                - for each pickle extract the data, calculate L2s, and put the 
                    grand median L2 into sub_L2.
                - put list sub_L2 as a bnpy array into 
                dictionary L2[i] with key j.
        
        This will output a dictionary where for every pair of keys i and j , we accquire
        an array of grand medians.
        """
        "names of first and second parameters. e.g. agents and prop"
        keys = self.param_keys
        "placeholder dictionary for all parameters" 
        L2 = {}
        "loop over first parameter. usually agents."
        for i in self.p1:
            print(i)
            "sub dictionary for parameter i"
            L2[i] = {} 
            for j in self.p2:
                "file names for glob to find. note wildcard * is needed"
                f_name = self.source + f"{keys[0]}_{i}_{keys[1]}_{j}-*"
                "find all files with given i and j"
                files = glob.glob(f_name)
                "placeholder list for grand medians of UKF runs with parameters i and j"
                sub_L2=[]
                for file in files:
                    "open pickle"
                    f = open(file,"rb")
                    u = pickle.load(f)
                    f.close()
                    "pull raw data"
                    truth, preds = self.depickle_data_parser(u)
                    "find L2 distances"
                    distances = L2_parser(truth[::u.sample_rate,:], 
                                          preds[::u.sample_rate,:])
                    if self.restrict is not None:
                        distances = self.restrict(distances, u, self.kwargs)
                    
                    "add grand median to sub_L2"
                    sub_L2.append(np.nanmean(np.nanmedian(distances,axis=0)))
                    "stack list of grand medians as an nx1 vector array"
                    "put array into grand dictionary with keys i and j"
                L2[i][j] = np.hstack(sub_L2)
           
        return L2
    
    def data_framer(self,L2):
        sub_frames = []
        keys = self.param_keys
        for i in self.p1:
            for j in self.p2:
                "extract L2s from L2 dictionary with corresponding i and j."
                L2s = L2[i][j]
                sub_frames.append(pd.DataFrame([[i]*len(L2s),[j]*len(L2s),L2s]).T)
    
        "stack into grand frames and label columns"
        error_frame = pd.concat(sub_frames)
        error_frame.columns = [keys[0], keys[1], "Grand Median L2s"]
    
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

    def choropleth_plot(self, xlabel, ylabel, title):
       
        
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
        "rotate again so imshow right way up for labels (origin bottom left i.e. lower)"
        data2=np.flip(data2,axis=0) 
        im=ax.imshow(data2,interpolation="nearest",cmap=cmap,origin="lower")
        
        
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
        
        "labelling"
        ax.set_xticks(np.arange(len(self.p1)))
        ax.set_yticks(np.arange(len(self.p2)))
        ax.set_xticklabels(self.p1)
        ax.set_yticklabels(self.p2)
        ax.set_xticks(np.arange(-.5,len(self.p1),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(self.p2),1),minor=True)
        ax.grid(which="minor",color="k",linestyle="-",linewidth=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title = title + " Choropleth"
        cbar.set_label(title + " Grand Median L2s")
        
        "save"
        if self.save:
            plt.savefig(title + "_Choropleth.pdf")
        
    def boxplot(self, xlabel, ylabel, title):
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
        f_name = title  + "_boxplot.pdf"
        y_name = "Grand Median L2s"
        
        f = plt.figure()
        cat = sns.catplot(x=str(keys[1]),y=y_name,col=str(keys[0]),kind="box", data=data)
        plt.tight_layout()
        
        for i, ax in enumerate(cat.axes.flatten()):
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(str(keys[0]).capitalize() + " = " + str(self.p1[i]))
        plt.title = title
        if self.save:
            plt.savefig(f_name)
       
        
      
        
        
"""parameter dictionary
parameter1: usually number of "agents"

parameter2: proportion observed "prop" or size of aggregate squares "bin" 
    int 1 here because I named the results files poorly.!! fix later using grep?"

source : "where files are loaded from plus some file prefix such as "ukf" or "agg_ukf" e.g. dust repo or a USB


"""

def ex1_restrict(distances,instance, *kwargs):
    "split L2s for observed unobserved"
    try:
        observed = kwargs[0]["observed"]
    except:
        observed = kwargs["observed"] 
    index = instance.index
    
    if observed:
        distances = distances[:,index]
    elif not observed:
        "~ doesnt seem to work here for whatever reason. using delete instead"
        distances = np.delete(distances,index,axis=1)
        
    return distances
    
def ex1_depickle():

    depickle_params = {
            "agents" :  [10, 20,30],
            "prop" : [0.25, 0.5, 0.75, int(1)],
            #"source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
            "source" : "/media/rob/ROB1/ukf_results/ukf_",
            }
    "plot observed/unobserved plots"
    obs_bools = [True, False]
    obs_titles = ["Observed", "Unobserved"]
    for i in range(len(obs_bools)):
        
        "initialise plot for observed/unobserved agents"
        g_plts = grand_plots(depickle_params, True, restrict = ex1_restrict, observed = obs_bools[i])
        "make dictionary"
        L2 = g_plts.data_extractor()
        "make pandas dataframe for seaborn"
        g_plts.data_framer(L2)
        "make choropleth numpy array"
        g_plts.choropleth_array()
        "make choropleth"
        g_plts.choropleth_plot("Numbers of Agents", "Proportion Observed",obs_titles[i])
        "make boxplot"
        g_plts.boxplot("Proportion Observed", "Grand Median L2s",obs_titles[i])
        
        
def ex2_depickle():

    depickle_params = {
            "agents" :  [10, 20, 30],
            "bin" : [5,10,25,50],
            #"source" : "/home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg_ukf_",
            "source" : "/media/rob/ROB1/ukf_results_100_1/agg_ukf_",
            }

    "init plot class"
    g_plts = grand_plots(depickle_params,True)
    "make dictionary"
    L2 = g_plts.data_extractor()
    "make pandas dataframe for seaborn"
    g_plts.data_framer(L2)
    "make choropleth numpy array"
    g_plts.choropleth_array()
    "make choropleth"
    g_plts.choropleth_plot("Numbers of Agents", "Proportion Observed","Aggregate")
    "make boxplot"
    g_plts.boxplot("Grid Square Size", "Grand Median L2s", "Aggregate")
    
if __name__ == "__main__":
    #ex1_depickle()
    ex2_depickle()
    