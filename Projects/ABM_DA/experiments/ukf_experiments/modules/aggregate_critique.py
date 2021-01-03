#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots of kalman gains and assimilated innovations for aggregate critique.
@author: medrclaa
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
"if running this file on its own. this will move cwd up to ukf_experiments."
if os.path.split(os.getcwd())[1] != "ukf_experiments":
    os.chdir("..")
    
sys.path.append("../../stationsim")
sys.path.append("modules")

from ukf2 import pickle_main
from ukf_ex2 import *
from ukf_ex1 import *

def gain_element_plot(gains, ylabel):
    
    """plot elementwise trajectories of kalman gain (k*mu) for analysis
    
    """
    f = plt.figure()
    
    if len(gains.shape)>1:
        for i in range(gains.shape[0]):
            for j in range(gains.shape[1]):
                #if j%2==1:
                plt.plot(gains[i,j,]) 
    else:
        plt.plot(gains)
    plt.xlabel("number of assimilations over time")
    plt.ylabel(ylabel)


def data_extraction(u):
    
    """ pull kalman gains and mus from ukf instance for analysis
    
    """
    ks = np.dstack(u.ukf.ks)
    mus = np.dstack(u.ukf.mus)
    
    kmus = []
    for i in range(ks.shape[2]):
        kmus.append(np.matmul(mus[:,:,i], ks[:,:,i].T))
    
    kmus = np.dstack(kmus)    
    
    return ks, mus, kmus

def main(f_name):
    

    u = pickle_main(f_name, "pickles/", True)
    
    ks, mus, kmus = data_extraction(u)
    gain_element_plot(ks[0,0,:], "single kalman gain")
    gain_element_plot(ks, "ks")
    gain_element_plot(mus, "mus")
    gain_element_plot(kmus, "kmus")
    
    return ks, mus, kmus
    
if __name__ == "__main__":
    
    n = 10
    bin_size = 50
    prop = 1.0
    
    f_name = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"  
    #f_name = f"ukf_agents_{n}_prop_{prop}.pkl"
    
    ks, mus, kmus = main(f_name)

    k = ks[:,:,10]
    mu = mus[:,:,10]
    kmu = kmus[:,:,10]
    