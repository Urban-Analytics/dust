#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: RC

Attempt at monte carlo sigma point kalman filter (SPKF) augment using rejection 
sampling/approximate bayesian computation (ABC).

Dubbing it the ABCUKF.

The aim here is to convert aggregate counts into a sample of agent positions
 for processing in an ensemble filtering technique such as the enkf/ukf.

General gist of the algorithm is as follows:

-Assume we have a run a stationsim model observed via a grid of counts. 
 After two time steps we gain two measurements at time steps 1 and 2 respectively 
 (y_1 and y_2).

- We only know which squares our agents are in and wish to estimate their GPS 
  positions using approximate bayesian computation (ABC).

- To do this, we randomly generate a vector of agents positions normally
  distributed about x and P. We process this through some transition function f
  which also gives a normally distributed vector of positions.
    
- We then put this through a measurement function converting it into counts
  z_2 using some distance metric. If z_2 is within some tolerance of the
  true counts y_2 we accept it as belonging to an ensemble of potential agent
  positions.

- We repeat the previous two steps until a suitable ensemble population has 
  been reached.

- We then use this ensemble to predict the current state of our agents through
 any potential sigma point filter (ukf,pf,enkf).

- When we get our measurement at time step 3. We can then step our ensemble again,
  measure the distance between z_3 and y_3, keep any suitable members and 
  replenish the sample as necessary using futher ABC.

- We repeat the above 5 steps as necessary until StationSim finishes.
"""
import numpy as np
from scipy.stats import multivariate_normal as mvn

import sys
sys.path.append("../../../stationsim")
from stationsim_model import Model

sys.path.append("../modules")

from ukf_fx import fx
import default_ukf_configs as configs
from poly_functions import grid_poly, poly_count
import datetime
from ukf_plots import ukf_plots

"""list of functions needed and their purposes
Sigma Point generator
some weightings
rskf class
rskf_ss class
"""

def hx2(state,**hx_kwargs):
    
    
    """Convert each sigma point from noisy gps positions into actual measurements
    
    - take some desired state vector of all agent positions
    - count how many agents are in each of a list of closed polygons using poly_count
    - return these counts as the measured state equivalent
    
    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted
    
    **hx_args
        generic hx kwargs
    Returns
    ------
    counts : array_like
        forecasted `counts` of how many agents in each square to 
        compare with actual counts
    """
    poly_list = hx_kwargs["poly_list"]
    
    counts = poly_count(poly_list, state)

    return counts

def L1(v1, v2):

    """ Calculate L1 (Manhattan) distance between two vectors of counts.

    Parameters
    ------

    v1, v2 : list

        lists `v1` and `v2` of counts for how many agents are in each of a 
        set of polygons.

    Returns
    ------
    dist : float
        `dist` L1 distance between v1 and v2
    """

    v1 = np.array(v1)
    v2 = np.array(v2)
    dist = np.sum(np.abs(v1-v2))
    return dist


def rejection_sampler(v1, v2, distance_metric, tol, **metric_kwargs):

    """ Test if v1 and v2 are within some proximity of each other given an 
    arbitrary distance metric.

    Parameters
    ------

    v1, v2 : list
            lists `v1` and `v2` of counts for how many agents are in each of a 
            set of polygons. Typically v2 is the true data and v1 is some 
            randomly generated sample to compare.

    distance_metric : func
        metric of distance between two vectors. Typically L1 metric for 
        vectors of counts.

    tol : float
        tolerance `tol` indicating whether to reject v1 given its proximity to v2.
        lower tol implies tighter restrictions and potentially more time 
        generating a sample.

    metric_kwargs : kwargs
        `metric_kwargs` any kwargs for the distance metric

    Returns
    ------
    keep : bool
        whether to `keep` v1 or not.
    """

    dist = distance_metric(v1, v2, **metric_kwargs)
    if dist <= tol:
        accept = True
    else:
        accept = False
    return accept

def add_Sigmas(sigmas, x, p, sample_size):
    
    """Repopulate sigmas list until we have sample_size points
    
    
    Parameters
    ------
    sigmas: array_like
        `sigmas` list of active sigma points

    x, p : array_like
        state mean `x` and covariance `p`
        
    Returns 
    ------ 
        repopulated list of `sigmas` with sample_size points
    """
    
    distribution= mvn(x, p)
    
    while len(sigmas) < sample_size:
        new_point = distribution.rvs(1)
        sigmas.append(new_point)

  
def importance_sample_weights(x, p, sigmas):
    
    """ weight sigmas based on proximity to 
    
    """
    
    distribution = mvn(x,p)
    mean_density = distribution.pdf(x)
    
    weights = []
    for item in sigmas:
        weights.append(distribution.pdf(item)/mean_density)
    
    "convert to array and normalise."
    weights = np.array(weights)
    weights /= np.sum(weights)
    
    return weights
    
    
    
class forwards_abcukf():

    """forwards version of the algorithm
    
    populate sigmas with current x and p
    propagate through f
    measure through h
    compare with truth when it comes in
    reject sigmas outside of tolerance
    some x/P estimate with accepted sample e.g. unscented
    repopulate and propagate sigmas again until next observation
    
    This method is 'forwards' due to the flow of data being entirely
    forwards in time. We have our sigma points ready at the next time step
    ready to receive the observations. This method is susceptible to sample
    degeneracy as with the particle filter but is actually real time as compared
    to the backwards method
    
    """

    def __init__(self, kf_params):
        
        for key in kf_params.keys():
            setattr(self, key, kf_params[key])
            
        self.x = self.base_model.get_state(sensor="location")
       
        self.sigmas = []
        
    
        
    def predict(self, **fx_kwargs):
        
        """ predict step of forwards abcukf
        
        generate sigma points if there arent any.
        propagate poinrs forwards.
        wait for observations
        """
        add_Sigmas(self.sigmas, self.x, self.p, self.sample_size)
        

        
        sigmas = np.vstack(self.sigmas).T
        sigmas = np.apply_along_axis(fx, 0, sigmas, **self.fx_kwargs)
        

        self.sigmas = sigmas.T.tolist()        
        self.x = np.mean(sigmas, axis = 1)
        self.p = np.cov(sigmas) + self.q
        
    
    
    def update(self, z, **hx_kwargs):
        
        """ update step of backwards abcukf
        
        accept any sigma points within tolerance
        calculate some proximity weighting (L2 to a central sigma?)
        
        """
        
        "propagate sigmas through h. remove any outside of tolerance."
        
        for i, item in enumerate(self.sigmas):
            y = self.hx(item, **self.hx_kwargs)
            if not rejection_sampler(y, z, L1, self.tol):
                del self.sigmas[-i]
                
        "calculate mean and covariance estimates."
        
        sigmas = np.array(self.sigmas)
        self.x = np.mean(sigmas, axis = 0)
        c = np.cov(sigmas.T)
        
                
    
class backwards_abcukf():
    
    """ Backwards version of abcukf
    
    Given new measurements we propagate forwards sigma points as follows:

    wait until new measurements at time t+1
    
    using latent state estimate x and P at time t generate a single sigma point
    
    propgate this through f and h
    
    if it is suitably close to the new measurement keep it
    
    repeat this until a suitable sample has been attained
    
    once this sample is attained estimate x/P at time t+1
    
    wait for the next observation and repeat.
    
    This is the backwards method as the observation flows backwards
    from time t+1 to t. We do not generate sigma points until we have the 
    observation. This method will be considerably slower and is not 
    truly real time. However we have a greatly reduced chance of sample 
    degeneracy.
    
    Perhaps a combination of the two would be useful.
    
    Returns
    -------

    """
    def __init__(self):
        pass
    
        
    def backwards(self, sigmas, z, f, h, ):

        
        pass

class backandforwards():
    
    "perhaps combine the above two into a seesaw type method"
        
        
class abcukf_ss():

    def __init__(self, model_params, kf_params):
        
        self.model_params = model_params
        self.kf_params = kf_params
        
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in kf_params.keys():
            setattr(self, key, kf_params[key])
    
        self.obs = []  # actual sensor observations
        self.ukf_histories = []  #ukf predictions
        self.forecasts=[] #pure stationsim predictions
        self.truths = []  # noiseless observations
        
        self.init_kf(model_params, kf_params)
        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
            
    def init_kf(self, model_params, kf_params):
        
        """initiate abckf
        
        """
        
        self.kf = forwards_abcukf(kf_params)
        
        
    def predict(self):
        
        """prediction step
        
        """
                
        self.kf.predict() 
        self.forecasts.append(self.kf.x)
        self.base_model.step()
        self.truths.append(self.base_model.get_state(sensor="location"))
            
    def update(self, step):    
        
        if step%self.sample_rate == 0:
            
            state = self.base_model.get_state(sensor="location")
            noise_array=np.ones(self.pop_total*2)
            noise_array[np.repeat([agent.status!=1 for 
                                   agent in self.base_model.agents], 2)]=0
            noise_array*=np.random.normal(0, self.noise, self.pop_total*2)
            state+=noise_array
                
            "convert full noisy state to actual sensor observations"
            state = self.kf.hx(state, **self.hx_kwargs)
                
            self.kf.update(state, **self.hx_kwargs)
            
            if self.obs_key_func is not None:
                key = self.obs_key_func(state,**self.obs_key_kwargs)
                "force inactive agents to unobserved"
                key *= [agent.status%2 for agent in self.base_model.agents]
                self.obs_key.append(key)

            self.ukf_histories.append(self.kf.x) #append histories
            self.obs.append(state)
    
    def main(self):
        """main function for applying ukf to gps style station StationSim
        -    initiates ukf
        -    while any agents are still active
            -    predict with ukf
            -    step true model
            -    update ukf with new model positions
            -    repeat until all agents finish or max iterations reached
        -    if no agents then stop
        
        """
                
        for step in range(self.step_limit-1):
            
            "forecast next StationSim state and jump model forwards"
            self.predict()
            "assimilate forecasts using new model state."
            self.update(step)
            
            finished = self.base_model.pop_finished == self.pop_total
            if finished: #break condition
                break
            
            #elif np.nansum(np.isnan(self.ukf.x)) == 0:
            #    print("math error. try larger values of alpha else check fx and hx.")
            #    break
          

        self.time2 = datetime.datetime.now()#timer
        print(self.time2-self.time1)



if __name__ == "__main__":
    
    n = 30
    bin_size = 25
    model_params = configs.model_params
    model_params["pop_total"] = n
    
    base_model = Model(**configs.model_params)
    poly_list = grid_poly(model_params["width"], model_params["height"], bin_size)
    
    kf_params = {
        
        "n" : n,        
        "bin_size" : bin_size,
        "sample_rate" : 20, 
        "noise": 5,
        
        "p" : np.eye(2*n),
        "q" : np.eye(2*n), 
        "r" : np.eye(2*len(poly_list)),
        
        "sample_size" : 50,
        
        "base_model" : base_model ,
        
        "fx" : fx,
        "hx" : hx2, 
        "fx_kwargs" : {"base_model" : base_model}, 
        "hx_kwargs" : {"poly_list": poly_list},
        "obs_key_func" : None,
        'tol' : 10,
        
        }
    
    abckf = abcukf_ss(model_params, kf_params)
    abckf.main()
    
    a = np.vstack(abckf.truths)
    b = np.vstack(abckf.ukf_histories)
    res = a[::kf_params["sample_rate"], :] - b

    plts = ukf_plots(abckf, "","",False, False, {"markers":"","colours":"","labels":"", })    
    
    plts.path_plots(a, "truth", poly_list)
    plts.path_plots(b, "prediction", poly_list)


    
"""
import matplotlib.pyplot as plt
for i in range(asd2.shape[1]):
    traj = asd2[:, i]
    x =  np.array(traj[0][0])
    y = -1 * np.array(traj[0][1])
    plt.plot(x,y)
"""    
    