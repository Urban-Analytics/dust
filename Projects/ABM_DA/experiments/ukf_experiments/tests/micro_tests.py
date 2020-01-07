#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Jan 15:33:05 2020

@author: rob

Testing for the micro level defined as tests for agents' basic elements and 
interactions from their perspective. Aim to htest that each individual testable
element is functioning as desired. I.E, testfunctional and structural defects
in a testable element. For example:
    
    - test agent building blocks like behaviour knowledge etc.
    - general deterministic/white box functions
    - building blocks of the environment e.g. non agents
    - test agents lifetime output
    - test an agent achieves something in a considerable time frame unver
        varying IC.
    - test interactions between elements
    - test quality properties of agents (e.g. number of behaviours scheduled at a 
    specific time.)
"""

import sys
sys.path.append("..")
from ukf_modules.default_ukf_configs import model_params, ukf_params

sys.path.append("../../../stationsim")
from stationsim_model import Model


    
test_model = Model(**model_params)


class test_UKF():
    
    def __init__(self, model):
        self.model = model
        
    
    