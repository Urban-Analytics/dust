#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:42:17 2019

@author: rob
"""
"used in HiddenPrints"
import os
import sys
import numpy as np

"used in fx to restore stepped model"
from copy import deepcopy  

class HiddenPrints:
    
    """stop repeat printing from stationsim 
    We get a lot of `iterations : X` prints as it jumps back 
    and forth over every 100th step. This stops that.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def fx(x, **fx_kwargs):
    """Transition function for the StationSim
    This seems long winded but means only 1 stationsim instance is needed.
    Uses much less memory and is generally more efficient than
    running the full 2n + 1 stationsims particularly when multiprocessing.
    
    - Copies current base model
    - Replaces positions with some sigma points
    - Step replaced model forwards one time point
    - record new stationsim positions as forecasted sigmapoint
    - loop over this in ukf function to get family of forecasted 
      sigmas
    
    Parameters
    ------
    x : array_like
        Sigma point of measured state `x`
    **fx_args
        arbitrary arguments for transition function fx
        
    Returns
    -----
    state : array_like
        predicted measured state for given sigma point
    """   
    
    #f = open(f"temp_pickle_model_ukf_{self.time1}","rb")
    #model = pickle.load(f)
    #f.close()
    base_model = fx_kwargs["base_model"]
    model = deepcopy(base_model)
    if x is not None:
        model.set_state(state = x, sensor="location")    
    with HiddenPrints():
        model.step() #step model with print suppression
    state = model.get_state(sensor="location")
    
    return state
