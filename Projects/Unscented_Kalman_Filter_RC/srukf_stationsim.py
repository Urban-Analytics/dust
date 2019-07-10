# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019

@author: RC

first attempt at a square root UKF class
class built into 5 steps
-init
-Prediction SP generation
-Predictions
-Update SP generation
-Update

SR filter generally the same as regular UKF for efficiency 
but more numerically stable wrt rounding errors 
and preserving PSD covariance matrices

based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
"""


import numpy as np
from math import floor
import matplotlib.pyplot as plt
import datetime
from copy import deepcopy

import pickle
import os

from StationSim_Wiggle import Model
from ukf_plots import plots,animations
from srukf import srukf

plt.style.use("dark_background")

class srukf_ss:
    def __init__(self,model_params,filter_params,srukf_params):
        """
        *_params - loads in parameters for the model, station sim filter and general UKF parameters
        base_model - initiate stationsim 
        pop_total - population total
        number_of_iterations - how many steps for station sim
        sample_rate - how often to update the kalman filter. intigers greater than 1 repeatedly step the station sim forward
        sample_size - how many agents observed if prop is 1 then sample_size is same as pop_total
        index and index 2 - indicate which agents are being observed
        srukf_histories- placeholder to store ukf trajectories
        time1 - initial time used to calculate run time 
        """
        #call params
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
        self.srukf_params = srukf_params
        
        self.base_model = Model(model_params) #station sim

        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        #number of batch iterations
        self.number_of_iterations = model_params['batch_iterations']
        self.sample_rate = self.filter_params["sample_rate"]
        #how many agents being observed
        if self.filter_params["do_restrict"]==True: 
            self.sample_size= floor(self.pop_total*self.filter_params["prop"])
        else:
            self.sample_size = self.pop_total
            
        #random sample of agents to be observed
        self.index = np.sort(np.random.choice(self.model_params["pop_total"],
                                                     self.sample_size,replace=False))
        self.index2 = np.empty((2*self.index.shape[0]),dtype=int)
        self.index2[0::2] = 2*self.index
        self.index2[1::2] = (2*self.index)+1
        
        self.srukf_histories = []
   
        self.time1 = datetime.datetime.now()#timer

    def fx(self,x,**fx_args):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially gradient * v *dt assuming no collisions
        with some algebra to get it into cartesian plane.
        
        to generalise this to every agent need to find the "ideal move" for each I.E  assume no collisions
        will make class agent_ideal which essential is the same as station sim class agent but no collisions in move phase
        
        New fx here definitely works as intended.
        I.E predicts lerps from each sigma point rather than using
        sets of lerps at 1 point (the same thing 5 times)
        in:
            x-prior state of base_model before step
            fx_args - generic kwargs
        out:
            prediction of base_model state at next step given prior positions x
            used to propagate sigmas points
        """
        #!!need to properly test deepcopy vs pickle both here for now. 
        #f = open(f"temp_pickle_model_srukf_{self.time1}","rb")
        #model = pickle.load(f)
        #f.close()
        model = deepcopy(self.base_model)
        #model = deepcopy(self.base_model)
        #state = model.agents2state()
        #model.state2agents(state = state)    
        #model.step()
        #state = model.agents2state()
        #return state[self.index2]
    
        model.state2agents(state = x)    
        model.step()
        state = model.agents2state()
        return state
   
    def hx(self,state,**hx_args):
        """
        Measurement function for agent. I.E. where the agents are recorded to be
        returns observed agents.
        in:
            full state space
        out: 
            observed subset of full state
        """
        #state = state[self.index2]
        
        return state[self.index2]
    
    def init_srukf(self):
        state = self.base_model.agents2state(self)
        Q = np.eye(len(state))
        R = np.eye(len(self.index2))
        self.srukf = srukf(srukf_params,state,self.fx,self.hx,Q,R)
        
        self.srukf_histories.append(self.srukf.x)
    
    
    def main(self):
        """
        main function for ukf station sim
        -initiates srukf
        -predict with srukf
        -step true model
        -update srukf with new model positions
        -repeat until model ends or max iterations reached
        """
        np.random.seed(seed = 8)#seeding if  wanted else hash it
        #np.random.seed(seed = 7)#seeding if  wanted else hash it

        
        self.init_srukf() 
        for _ in range(self.number_of_iterations-1):
            if _%100 ==0: #progress bar
                print(f"iterations: {_}")
                

            #f_name = f"temp_pickle_model_srukf_{self.time1}"
            #f = open(f_name,"wb")
            #pickle.dump(self.base_model,f,pickle.HIGHEST_PROTOCOL)
            #f.close()
            
            
            
            self.srukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            
            #os.remove(f"temp_pickle_model_srukf_{self.time1}")
            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state() #observed agents states
                self.srukf.update(z=state[self.index2]) #update UKF
                self.srukf.x[sr.srukf.x<0]=0
                x = self.srukf.x
                if np.sum(np.isnan(x))==x.shape[0] or np.sum(np.abs(x-self.srukf_histories[-1]))>1e3:
                    print(np.sum(np.abs(x-self.srukf_histories[-1])))
                    print("linalg error: increasing alpha")
                    print(" ")
                    srukf_params["abort"]=True
                    break
                self.srukf_histories.append(self.srukf.x) #append histories

                "debug point placeholder because spyder is dumb and you cant stop at empty lines"
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        time2 = datetime.datetime.now()#timer
        print(time2-self.time1)
        
    def data_parser(self,do_fill):
        """
        extracts data into numpy arrays
        in:
            do_fill - If false when an agent is finished its true position values go to nan.
            If true each agents final positions are repeated in the truthframe 
            until the end of the whole model.
            This is useful for various animating but is almost always kept False.
            Especially if using average error metrics as finished agents have practically 0 
            error and massively skew results.
        out:
            a - actual agents positions
            b - ukf predictions of said agent positions
        """
        sample_rate = self.sample_rate

        a2 = {}
        for k,agent in  enumerate(self.base_model.agents):
            a2[k] =  agent.history_loc
        max_iter = max([len(value) for value in a2.values()])
        b2 = np.vstack(self.srukf_histories)
        
        a= np.zeros((max_iter,self.pop_total*2))*np.nan
        b= np.zeros((max_iter,b2.shape[1]))*np.nan
        
  
        for i in range(int(a.shape[1]/2)):
            a3 = np.vstack(list(a2.values())[i])
            a[:a3.shape[0],(2*i):(2*i)+2] = a3
            if do_fill:
                a[a3.shape[0]:,(2*i):(2*i)+2] = a3[-1,:]

        
        for j in range(int(b.shape[0]//sample_rate)):
            b[j*sample_rate,:] = b2[j,:]
            
        "all agent observations"
        
        return a,b



if __name__ == "__main__":
    """
            a - alpha scaling parameter determining how far apart sigma points are spread. Typically between 1e-4 and 1
            b - beta scaling paramater incorporates prior knowledge of distribution of state space. b=2 optimal for gaussians
            k - kappa scaling parameter typically 0 for state space estimates and 3-dim(x) for parameter estimation
            init_x- initial state space
    """
    """
            width - corridor width
            height - corridor height
            pop_total -population total
            entrances - how many entrances
            entrance speed- mean entry speed for agents
            exits - how many exits
            exit_space- how wide are exits 
            speed_min - minimum agents speed to prevent ridiculuous iteration numbers
            speed_desire_mean - desired mean of normal distribution of speed of agents
            speed_desire_std - as above but standard deviation
            separation - agent radius to determine collisions
            wiggle - wiggle distance
            batch_iterations - how many model steps to do as a maximum
            3 do_ bools for saving plotting and animating data. 
    """
    model_params = {
            
            'width': 200,
            'height': 100,
            'pop_total': 20,
            'entrances': 3,
            'entrance_space': 2,
            'entrance_speed': 1,
            'exits': 2,
            'exit_space': 1,
            'speed_min': .1,
            'speed_desire_mean': 1,
            'speed_desire_std': 1,
            'separation': 4,
            'wiggle': 1,
            'batch_iterations': 10000,
            'do_save': True,
            'do_plot': False,
            'do_ani': False
            }
        
    filter_params = {         
            """
            Sensor_Noise - how reliable are measurements H_x. lower value implies more reliable
            Process_Noise - how reliable is prediction fx lower value implies more reliable
            sample_rate - how often to update kalman filter. higher number gives smoother predictions
            do_restrict - restrict to a proportion prop of the agents being observed
            do_animate - do animations of agent/wiggle aggregates
            do_wiggle_animate
            do_density_animate
            do_pair_animate
            prop - proportion of agents observed. 1 is all <1/pop_total is none
            heatmap_rate - after how many updates to record a frame
            bin_size - square sizes for aggregate plots,
            do_batch - do batch processing on some pre-recorded truth data.
            """
            
            "Sensor_Noise":  1, # how reliable are measurements H_x. lower value implies more reliable
            "Process_Noise": 1, #how reliable is prediction fx lower value implies more reliable
            'sample_rate': 1,   #how often to update kalman filter. higher number gives smoother (maybe oversmoothed) predictions
            "do_restrict": True, #"restrict to a proportion prop of the agents being observed"
            "do_animate": False,#"do animations of agent/wiggle aggregates"
            "do_wiggle_animate": False,
            "do_density_animate":False,
            "do_pair_animate":False,
            "prop": 0.9,#proportion of agents observed. 1 is all <1/pop_total is none
            "heatmap_rate": 2,# "after how many updates to record a frame"
            "bin_size":10,
            "do_batch":False,
            "plot_unobserved":True
            }
    """
            a - alpha between 1 and 1e-4 typically determines spread of sigma points.
            b - beta set to 2 for gaussian. determines trust in prior distribution.
            k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
                not 100% sure what kappa does. think its a bias parameter.
            !! might be worth making an interactive notebook that varies these. for fun
    """
    srukf_params = {
            "a":0.1,
            "b":2,
            "k":0,
            "d_rate" : 10,
            "abort" : False

            }
    alphas= [1,2,5,10,15,25,50,100,1000]
    
    for i in range(len(alphas)):
        srukf_params["abort"]=False
        sr = srukf_ss(model_params,filter_params,srukf_params)
        try:   
            srukf_params["a"]=alphas[i]
            print(f"alpha: {alphas[i]}")
            sr.main()
            if not srukf_params["abort"]:
                actual,preds= sr.data_parser(True)
                
                if filter_params["plot_unobserved"] or filter_params["prop"]<1:
                    distances,t_mean = plots.diagnostic_plots(sr,actual,preds,False,False)
                distances2,t_mean2 = plots.diagnostic_plots(sr,actual,preds,True,False)
                break
    
        except np.linalg.LinAlgError:
            print("linalg error: increasing alpha")

        finally:
            srukf_params["a"]*=10
    if srukf_params["abort"]: #final check for everything failed
        print ("no suitable alpha found")
        print("math error. try larger values of alpha else check fx and hx.")
