#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:44:23 2019

Take 1 on pytesting for ukf

Its very hard to accurately assert a highly random ABM is doing exactly the 
right thing.
I.E. Impossible to fix the input and get an expected output without 
heavy pickling which defeats the point.

For now im making sure the predict/update steps are working as intended.

@author: RC
"""
import sys
sys.path.append("../..")
sys.path.append("../../stationsim")

from stationsim.ukf import ukf_ss,plots
import numpy as np
from stationsim_model import Model

#%%
"define ABM and UKF parameters"

"""
width - corridor width
height - corridor height
pop_total -population total
entrances - how many entrances
entrance speed- mean entry speed for agents
exits - how many exits
exit_space- how wide are exits 
speed_min - minimum agents speed to prevent ridiculuous iteration numbers
speed_mean - desired mean of normal distribution of speed of agents
speed_std - as above but standard deviation
speed_steps - how many levels of speed between min and max for each agent
separation - agent radius to determine collisions
wiggle - wiggle distance
batch_iterations - how many model steps to do as a maximum
3 do_ bools for saving plotting and animating data. 
"""

model_params = {
			'pop_total': 5,

			'width': 200,
			'height': 100,

			'gates_in': 3,
			'gates_out': 2,
			'gates_space': 1,
			'gates_speed': 1,

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
"""
Sensor_Noise - how reliable are measurements H_x. lower value implies more reliable
Process_Noise - how reliable is prediction fx lower value implies more reliable
sample_rate - how often to update kalman filter. higher number gives smoother predictions

prop - proportion of agents observed. this is a floor function that rounds the proportion 
DOWN to the nearest intiger number of agents. 1 is all <1/pop_total is none

bring_noise: add noise to true ukf paths
noise: variance of said noise (0 mean)
do_batch - do batch processing on some pre-recorded truth data.
"""

filter_params = {      
   
"Sensor_Noise":  1, 
"Process_Noise": 1, 
'sample_rate': 5,
"prop": 0.5,
"bring_noise":True,
"noise":0.5,
"do_batch":False,

}

"""
a - alpha between 1 and 1e-4 typically determines spread of sigma points.
however for large dimensions may need to be even higher
b - beta set to 2 for gaussian. determines trust in prior distribution.
k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
not 100% sure what kappa does. think its a bias parameter.
!! might be worth making an interactive notebook that varies these. for fun
"""

ukf_params = {

"a":1,
"b":2,
"k":0,

}

base_model = Model(**model_params)
u = ukf_ss(model_params,filter_params,ukf_params,base_model)
u.init_ukf(ukf_params)  

#%%

def state_test(u):
    """Make sure the ukf state is still in tact.
    """
    
    "check an array of correct size"
    assert type(u.ukf.x) == np.ndarray
    assert np.shape(u.ukf.x) == (model_params["pop_total"]*2,)
    
    "check all agents within boundaries"
    assert np.nanmin(u.ukf.x[0::2]) >= 0
    assert np.nanmin(u.ukf.x[1::2]) >= 0
    assert np.nanmax(u.ukf.x[0::2]) <= model_params["width"] 
    assert np.nanmax(u.ukf.x[1::2]) <= model_params["height"] 
    
    "check some agent is still in the model (no exponential decay/divergence)"
    assert  ~np.all(np.isnan(u.ukf.x))
    
    "check ukf covariance is positive semi definite (PSD)"
    assert np.all(np.linalg.eigvals(u.ukf.P)>0)

class Test_ukf():
    """
    testing for ukf class 
    """
    
    def test_Sigmas(self):
        """test ukf.Sigmas 
        in: all 1s mean and identity covariance of 
            
        out: sigma point array 
        """
        n=model_params["pop_total"]
        n2 = 2*n
        "fake mean"
        x = np.ones(n2)
        "fake covariance to divide by g to get normalised intigers in sigmas"
        P = np.eye((n2))
        
        "fake sigma array"
        test = np.ones((n2,2*n2+1))
        test[:,1:n2+1] +=P #'upper' confidence sigmas
        test[:,n2+1:] -=P #lower sigmas 
        
        np.testing.assert_almost_equal(u.ukf.Sigmas(mean=x,P=P/u.ukf.g ),test)

class Test_ukf_ss(object):    
    
    def test_fx(self):
        """test transition function works and isnt mangling state space
        """
        
        "jump model forwards 10 steps"
        for _ in range(10):
            np.random.seed(seed = 8)
            u.base_model.step()
            u.ukf.predict()
        
        "test state space"
        state_test(u)
        assert np.shape(u.fx(u.ukf.x)) == (2*u.pop_total,)
                                                    
    def test_hx(self):
        """test measurement function works and isnt mangling state space       
        """
        "assimilate"
        state = u.base_model.get_state(sensor="location") #observed agents stat
        u.ukf.update(z=state[u.index2]) #update UKF
        
        "test state space"
        state_test(u)
        assert np.shape(u.hx(state)) == (2*u.sample_size,)
               
class Test_ukf_plots():       #
    
    def test_data_parser(self):
        obs,preds,full_preds,truth,nan_array= u.data_parser(True)

    pass
    
    
    
    
    