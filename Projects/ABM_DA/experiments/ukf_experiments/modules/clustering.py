#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test code for stationsim trajectory clustering.

https://medium.com/isiway-tech/gps-trajectories-clustering-in-python-2f5874204a53
"""

from ukf_ex1 import ex1_pickle_name, hx1, obs_key_func
import numpy as np
from ukf_plots import L2s
import sys
sys.path.append("../../../stationsim")
from ukf2 import pickle_main


import geopy.distance
from dipy.segment.clustering import QuickBundles


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm


from dipy.segment.metric import Metric 
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.segment.metric import ResampleFeature 

import pandas as pd

def load_model(n, prop):
    f_name = ex1_pickle_name(n, prop)
    source = "../test_pickles/"
    do_pickle = True
    
    "try loading class_dicts first. If no dict then try class instance."
    try:
        u  = pickle_main("dict_" + f_name, source, do_pickle)
    except:
        u  = pickle_main(f_name, source, do_pickle)
     
    return u

def array_into_list(positions, nan_array):
    
    return [positions[~np.isnan(nan_array[:,2*i]),2*i:2*(i+1)] for i in range(positions.shape[1]//2)]


def extract_truths(u):
    
    obs,preds,truths, nan_array =  u.data_parser()
    obs *= nan_array[::u.sample_rate,u.ukf_params["index2"]]
    truths *= nan_array
    preds *= nan_array
    
    truth_list = array_into_list(truths, nan_array)
    obs_list = array_into_list(obs, nan_array[0::u.ukf_params["sample_rate"],:])
    preds_list = array_into_list(preds[0::u.ukf_params["sample_rate"],:], 
                                       nan_array[0::u.ukf_params["sample_rate"],:])

    return truth_list, obs_list, preds_list

class GPSDistance(Metric):  
     "Computes the average GPS distance between two streamlines. " 
     def __init__(self): 
         super(GPSDistance, self).__init__(ResampleFeature(nb_points = 256))
         
     def are_compatible(self, shape1, shape2):
         return len(shape1) == len(shape2)
     
     def dist(self, v1, v2):
         
         x = [geopy.distance.vincenty([p[0][0], p[0][1]], [p[1][0], p[1][1]]).km for p in list(zip(v1,v2))]
         currD = np.mean(x)
         return currD
     

def produce_Clusters(truth_list, thresh):
    
    feature = ResampleFeature(nb_points=24)
    metric = AveragePointwiseEuclideanMetric(feature=feature) 
    qb = QuickBundles(threshold = thresh, metric = metric)
    clusters = qb.cluster(truth_list)
    
    return qb, clusters

def cluster_search(agent_id, clusters):
    
    cluster_number = 0
    for cluster in clusters:
        if agent_id in cluster.indices:
            break
        else:
            cluster_number +=1
            
    return cluster_number

    
    "check which cluster an agent is in"
    
def cluster_plot(truth_list, clusters):

    #produce discrete colourbar
    n_clusters = len(clusters)
        
    cmap = cm.jet  # define the colormap
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap_indices = len(cmaplist)//n_clusters
    
    # define the bins for colourbar
    bounds = np.linspace(0, n_clusters, n_clusters + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    mappable = cm.ScalarMappable(norm, cmap)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))  # setup the plot

    for i,truth in enumerate(truth_list):
        tag = cluster_search(i,clusters)
        plt.plot(truth[:,0], truth[:,1],linewidth = 6, color=cmaplist[tag*cmap_indices])
        
    #set bound of corridor
    plt.xlim([0,200])
    plt.ylim([0,100])
    
    # create a second axes for the colorbar
    ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(mappable = mappable, cax = ax2, cmap = cmap, norm = norm, 
                 boundaries = bounds, ticks = bounds-0.5)
    
    cbar.set_ticklabels(bounds)
    #plt.colorbar(mappable, ax =ax2, cmap=cmap, norm=norm,
    #    spacing='proportional', ticks=bounds, boundaries=bounds)
    
    ax.set_title('Agent Clustering on StationSim')
    #ax2.set_ylabel('Cluster')
 
    
def extract_L2s(u):
    
    obs,preds,truths, nan_array =  u.data_parser()
    obs *= nan_array[::u.sample_rate,u.ukf_params["index2"]]
    truths *= nan_array
    preds *= nan_array
    
    errors = L2s(truths, preds)
    
    errors = np.nanmedian(errors, axis=0)

    return errors

def agent_error_frame(u, clusters):
    
    """ build a data frame of
    
    agent_id, agent median error, cluster size, observed?
    """
    n = u.model_params["pop_total"]
    index = u.ukf_params["index"]
    
    ids = np.arange(0, n)
    
    errors = extract_L2s(u)
    
    which_cluster = [cluster_search(i,clusters) for i in ids]
    cluster_size = [len(clusters[i]) for i in which_cluster]
    
    observed = [False]*n
    for i in index:
        observed[i] = True
        
    error_frame = pd.DataFrame([errors, cluster_size, observed]).T
    error_frame.columns = ["median agent error", "prediction cluster size", "Observed"]
    
    f = plt.figure(figsize = (8, 8))
    l1 = plt.scatter(error_frame.iloc[index]["median agent error"], 
                error_frame.iloc[index]["prediction cluster size"], color = "green")
    not_index = error_frame.loc[~error_frame.index.isin(index)].index
    l2 = plt.scatter(error_frame.iloc[not_index]["median agent error"], 
                error_frame.iloc[not_index]["prediction cluster size"], color = "red")  
    
    plt.legend([l1,l2],["observed","unobserved"])
    return error_frame

if __name__ == "__main__":       
    
    
    """
    n population
    prop proportion observed
    thresh quickbundle tuning parameter. lower implies tighter clustering. 
        low enough thresh converges to clusters of size 1
    """
    
    n = 30
    prop = 0.5
    thresh = 3
    u = load_model(n, prop)
    truth_list, obs_list, preds_list = extract_truths(u)
    qb, clusters = produce_Clusters(truth_list, thresh)
    cluster_plot(truth_list, clusters)
    
    error_frame = agent_error_frame(u, clusters)
