#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:44:23 2019

Take 1 on pytesting for ukf files

@author: RC
"""
import sys
sys.path.append("../..")
sys.path.append("../../stationsim")

from stationsim.ukf import ukf_ss
import numpy as np
from stationsim.stationsim_model import Model
import pickle

#%%


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




   
def pickle_saver(f_name,instance):
    f = open(f_name,"wb")
    pickle.dump(instance,f)
    f.close()

def pickle_loader(f_name):
    f = open(f_name,"rb")
    instance = pickle.load(f)
    f.close()
    return instance
    
f_name = f"test_model.pkl"

try:
    base_model = pickle_loader(f_name)

except:
    np.random.seed(seed = 8)
    base_model = Model(**model_params)
    pickle_saver(f_name,base_model)



    



#%%
#@pytest.mark.numpyfile
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
        "fake covariance without g to get normalised intigers in sigmas"
        P = np.eye((n2))
        
        "fake sigma array"
        test = np.ones((n2,2*n2+1))
        test[:,1:n2+1] +=P #'upper' confidence sigmas
        test[:,n2+1:] -=P #lower sigmas 
        
        np.testing.assert_almost_equal(u.ukf.Sigmas(mean=x,P=P/u.ukf.g ),test)
    

 
u = ukf_ss(model_params,filter_params,ukf_params,base_model)

np.random.seed(seed = 8)
index = np.sort(np.random.choice(model_params["pop_total"],u.sample_size,replace=False))
index2 = np.empty((2*index.shape[0]),dtype=int)
index2[0::2] = 2*index
index2[1::2] = (2*index)+1

u.index=index
u.index2=index2

u.init_ukf(ukf_params)  
   
class Test_ukf_ss(object):    
    
    
    def test_fx(self):
        """
        feed in some nice data and see if fx does what its supposed too
        
        
        how the hell do I test this its really random. numpy seeding does nothing.
        maybe feed it a fake fx?
        """
        
        
        for _ in range(5):
            np.random.seed(seed = 8)
            u.base_model.step()
            u.ukf.predict()
          
        np.testing.assert_almost_equal(u.ukf.x,np.array([ 8.37650321, 74.47490245,  4.10396988, 49.52422656,  2.5149293 ,
       73.67671124,  0.89609883, 75.1622166 ,  8.67836308, 24.58534842]))
        np.testing.assert_array_almost_equal(u.ukf.P,np.array([[ 7.46452460e+00, -3.97589037e-01, -1.38175491e-02,
                                                                -1.68085231e-02, -3.43079747e+00, -1.33106495e-01,
                                                                -3.83715897e-02, -4.21119503e-01,  8.99761429e-03,
                                                                 8.28269836e-04],
                                                               [-3.97589037e-01,  5.79604425e+00, -2.29531497e-03,
                                                                 3.22337101e-03,  5.71831228e-01, -4.34614298e-02,
                                                                 2.09130858e-02,  4.57365803e-02, -6.10019230e-03,
                                                                 1.29494588e-04],
                                                               [-1.38175491e-02, -2.29531497e-03,  3.12353851e+00,
                                                                -3.43118069e-01,  1.83632457e-01, -4.52757688e-01,
                                                                 5.71735221e-02, -1.88483892e-01,  1.26493872e-02,
                                                                -8.59193034e-04],
                                                               [-1.68085231e-02,  3.22337101e-03, -3.43118069e-01,
                                                                 6.01037477e+00,  6.37099000e-02, -2.79441186e-01,
                                                                 4.04457713e-02,  1.76240565e-01, -2.86816435e-03,
                                                                 1.87503837e-04],
                                                               [-3.43079747e+00,  5.71831228e-01,  1.83632457e-01,
                                                                 6.37099000e-02,  7.63967312e+00, -3.29132406e-01,
                                                                 8.66340408e-01,  3.34933483e-01,  5.64473931e-03,
                                                                -1.09514772e-03],
                                                               [-1.33106495e-01, -4.34614298e-02, -4.52757688e-01,
                                                                -2.79441186e-01, -3.29132406e-01,  8.13161816e+00,
                                                                 4.05712756e-02, -1.11787210e+00, -7.99111031e-01,
                                                                -6.12073526e-01],
                                                               [-3.83715897e-02,  2.09130858e-02,  5.71735221e-02,
                                                                 4.04457713e-02,  8.66340408e-01,  4.05712756e-02,
                                                                 3.12489636e+00,  1.83163559e-02, -9.23518680e-03,
                                                                -5.09053303e-03],
                                                               [-4.21119503e-01,  4.57365803e-02, -1.88483892e-01,
                                                                 1.76240565e-01,  3.34933483e-01, -1.11787210e+00,
                                                                 1.83163559e-02,  6.25739883e+00, -1.83696470e-01,
                                                                -1.93889908e-01],
                                                               [ 8.99761429e-03, -6.10019230e-03,  1.26493872e-02,
                                                                -2.86816435e-03,  5.64473931e-03, -7.99111031e-01,
                                                                -9.23518680e-03, -1.83696470e-01,  5.08422051e+00,
                                                                 4.13091135e-02],
                                                               [ 8.28269836e-04,  1.29494588e-04, -8.59193034e-04,
                                                                 1.87503837e-04, -1.09514772e-03, -6.12073526e-01,
                                                                -5.09053303e-03, -1.93889908e-01,  4.13091135e-02,
                                                                 5.70572237e+00]]))
                                                                
                                                                            
    
    def test_hx(self):
        """
        feed in some nice data and see if hx does what its supposed too
        """

        state = u.base_model.get_state(sensor="location") #observed agents stat
        u.ukf.update(z=state[u.index2]) #update UKF
        
        np.testing.assert_almost_equal(u.ukf.x,np.array([ 9.42400682, 74.31540774,  4.13564958, 49.56283482,  0.29853465,
       72.46802175,  0.6330997 , 75.36719009,  6.80767485, 24.52619221]))
        np.testing.assert_array_almost_equal(u.ukf.P,np.array([[ 6.09446148e+00, -1.71162894e-01,  4.60746505e-02,
                                                                 3.75198449e-04, -3.98212417e-01, -2.92768672e-02,
                                                                 3.07772195e-01, -3.21340643e-01, -1.97931318e-03,
                                                                -2.60160834e-03],
                                                               [-1.71162894e-01,  5.75813322e+00, -1.55517683e-02,
                                                                -1.69322860e-03,  6.60918667e-02, -2.51192395e-03,
                                                                -3.62568880e-02,  2.04994269e-02, -1.39256893e-03,
                                                                -1.90595636e-04],
                                                               [ 4.60746505e-02, -1.55517683e-02,  3.09758616e+00,
                                                                -3.58218473e-01,  1.93680940e-02, -4.95798630e-02,
                                                                 4.23413291e-02, -2.52103087e-01, -4.41942817e-03,
                                                                -4.62320676e-03],
                                                               [ 3.75198449e-04, -1.69322860e-03, -3.58218473e-01,
                                                                 6.00131608e+00,  6.19729150e-03, -3.09596839e-02,
                                                                 3.62769992e-02,  1.38187867e-01, -4.52465272e-03,
                                                                -2.76903840e-03],
                                                               [-3.98212417e-01,  6.60918667e-02,  1.93680940e-02,
                                                                 6.19729150e-03,  8.84093451e-01, -4.24381397e-03,
                                                                 1.00580520e-01,  3.39165682e-02, -4.47115065e-04,
                                                                -4.03534608e-04],
                                                               [-2.92768672e-02, -2.51192395e-03, -4.95798630e-02,
                                                                -3.09596839e-02, -4.24381397e-03,  8.88383985e-01,
                                                                 8.01886710e-03, -1.27988713e-01, -1.45873188e-02,
                                                                -1.00987280e-02],
                                                               [ 3.07772195e-01, -3.62568880e-02,  4.23413291e-02,
                                                                 3.62769992e-02,  1.00580520e-01,  8.01886710e-03,
                                                                 3.03742886e+00, -6.51127564e-03, -5.57946475e-04,
                                                                -7.33671490e-06],
                                                               [-3.21340643e-01,  2.04994269e-02, -2.52103087e-01,
                                                                 1.38187867e-01,  3.39165682e-02, -1.27988713e-01,
                                                                -6.51127564e-03,  6.08656000e+00, -4.67603432e-02,
                                                                -4.03028376e-02],
                                                               [-1.97931318e-03, -1.39256893e-03, -4.41942817e-03,
                                                                -4.52465272e-03, -4.47115065e-04, -1.45873188e-02,
                                                                -5.57946475e-04, -4.67603432e-02,  8.33726988e-01,
                                                                -3.07261536e-04],
                                                               [-2.60160834e-03, -1.90595636e-04, -4.62320676e-03,
                                                                -2.76903840e-03, -4.03534608e-04, -1.00987280e-02,
                                                                -7.33671490e-06, -4.03028376e-02, -3.07261536e-04,
                                                                 8.49953688e-01]]))
                                                                