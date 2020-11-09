#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:51:09 2019

@author: rob

These are the default stationsim and UKF filter parameters used in all experiments.
We append these dictionaries for the indivual experiment as necssary
"""

"""
width - corridor width
height - corridor height

gates_in - how many entrances
gates_out - how many exits
gate_space- how wide are the exits 
gate_speed- mean entry speed for agents

speed_min - minimum agents speed to prevent ridiculuous iteration numbers
speed_mean - desired mean of normal distribution of speed of agents
speed_std - as above but standard deviation
speed_steps - how many ractchet levels of speed between min and max for each agent

separation - parameter to determine collisions in cKDTree. Similar to agent size
wiggle - how far an agent moves laterally to avoid queueing

step_limit - how many model steps to do before stopping.

do_ bools for saving plotting and animating data. 
"""
model_params = {

'width': 740,
'height': 700,

'gates_in': 3,
'gates_out': 2,
'gates_space': 1,
'gates_speed': 1,

'speed_min': .2,
'speed_mean': 5,
'speed_std': 1,
'speed_steps': 3,

'separation': 5,
'max_wiggle': 1,

'step_limit': 3600,

'do_history': True,
'do_print': True,
}

"""
sample_rate - how often to update kalman filter. higher number gives smoother predictions
do_batch - do batch processing on some pre-recorded truth data. not working yet
bring_noise - add noise to measurements?
noise - standard deviation of added Gaussian noise

a - alpha between 1 and 1e-4 typically determines spread of sigma points.
however for large dimensions may need to be even higher
b - beta set to 2 for gaussian. determines trust in prior distribution.
k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
not 100% sure what kappa does. think its a bias parameter.
"""

ukf_params = {      

'sample_rate' : 5,
"bring_noise" : True,
"noise" : 0.5,
"do_batch" : False,

"a": 0.3,
"b": 2,
"k": 0,

"fx_kwargs_update_function": None,
"fx_update_args": [],
"hx_kwargs_update_function": None,
"hx_update_args": [],
"obs_key_kwargs_update_function": None,

"record": True,

}

"""Default colour scheme for ukf plots given 4 current types of observation."""
marker_attributes = {
"markers" : {-1: "o", 0 : "X", 1: "^", 2 : "s"},
"colours" : {-1: "black" , 0 : "orangered", 1: "yellow", 2 : "skyblue"},
"labels" :  {-1: "Pseudo-True Position", 0 : "Unobserved", 
             1: "Aggregated", 2 : "Observed"}

}
