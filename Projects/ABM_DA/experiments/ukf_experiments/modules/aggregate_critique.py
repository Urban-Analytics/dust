#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:50:52 2020

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

def gain_element_plot(gains):
    
    """plot elementwise trajectories of kalman gain (k*mu) for analysis
    
    """
    f = plt.figure()
    
    for i in range(gains.shape[0]):
        for j in range(gains.shape[1]):
            plt.plot(gains[i,j,:])    


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
    gain_element_plot(ks)
    gain_element_plot(mus)
    gain_element_plot(kmus)
    
    return ks, mus, kmus
    
if __name__ == "__main__":
    
    n = 20
    bin_size = 25
    prop = 1.0
    
    f_name = f"agg_ukf_agents_{n}_bin_{bin_size}.pkl"  
    #f_name = f"ukf_agents_{n}_prop_{prop}.pkl"
    
    ks, mus, kmus = main(f_name)

    k = ks[:,:,100]
    mu = mus[:,:,100]
    kmu = kmus[:,:,100]
    