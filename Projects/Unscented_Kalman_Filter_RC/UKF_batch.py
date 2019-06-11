
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:28:35 2019
@author: rob
UKF with one state and sampling rate addition
"""


import os
os.chdir("/home/rob/dust/Projects/RC_Scripts/")
import numpy as np
from math import floor
from StationSim import Model, Agent
from filterpy.kalman import MerweScaledSigmaPoints as MSSP
from filterpy.kalman import UnscentedKalmanFilter as UNKF
from filterpy.common import Q_discrete_white_noise as QDWN
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import datetime
import imageio
import matplotlib.cm as cm
import re
sqrt2 = np.sqrt(2)
plt.style.use("dark_background")

class UKF:
    """
    individually assign a UKF wrapper to each agent 
    update each agent individually depending on whether it is currently active within the model
    whether KF updates is determined by the "activity matrix" sourced at each time step from agent.active propertysqrt2 = np.sqrt(2)
    if active, update KF using standard steps
    this allows for comphrehensive histories to be stored in KF objects
    
    """
    
    def __init__(self,Model,model_params,filter_params):
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
        self.pop_total = self.model_params["pop_total"] #number of agents
        self.number_of_iterations = model_params['batch_iterations'] #number of batch iterations
        self.base_model = Model(model_params) #station sim
        self.UKF_histories = []
        self.UKF_Ps = []

        self.time_id = 0
        self.finished = 0 #number of finished agents
        
        self.sample_rate = self.filter_params["sample_rate"]
        if self.filter_params["do_restrict"]==True: 
            self.sample_size= floor(self.pop_total*self.filter_params["prop"])
        else:
            self.sample_size = self.pop_total
            
        self.index = np.sort(np.random.choice(self.model_params["pop_total"],
                                              self.sample_size,replace=False))
        self.index2 = np.empty((2*self.index.shape[0]),dtype=int)
        self.index2[0::2] = 2*self.index
        self.index2[1::2] = (2*self.index)+1

    def F_x(self,x,dt):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially gradient * v *dt assuming no collisions
        with some algebra to get it into cartesian plane.
        
        to generalise this to every agent need to find the "ideal move" for each I.E  assume no collisions
        will make class agent_ideal which essential is the same as station sim class agent but no collisions in move phase
        
        New fx here definitely works as intended.
        I.E predicts lerps from each sigma point rather than using
        sets of lerps at 1 point (the same thing 5 times)
        """
        loc = np.empty((len(x),))
        sample_agents = [self.base_model.agents[j] for j in self.index]
        for i,agent in enumerate(sample_agents):
            loc1 = agent.loc_desire
            loc2 = x[2*i:(2*i)+2]
            reciprocal_distance = sqrt2 / sum(abs(loc1 - loc2))  # lerp5: profiled at 6.41μs
            loc[2*i:(2*i)+2] = loc2 + agent.speeds[-1] * (loc1 - loc2) * reciprocal_distance
            
        
        
        return loc
        
        
    def H_x(self,z):
        """
        
        Measurement function for agent.
        !im guessing this is just the output from base_model.step
        """

        return z
    
    
    
    def init_ukf(self):
        
        """
        either updates or initialises UKF else ignores agent depending on activity status
        """
        
        "init UKF"
        
        
        state = self.base_model.agents2state(self)
        state = state[self.index2]
        sigmas = MSSP(n=len(state),alpha=0.01,beta=.2,kappa=1) #sigmapoints
        self.ukf =  UNKF(dim_x=len(state),dim_z=len(state),fx = self.F_x, hx=self.H_x, dt=1, points=sigmas)#init UKF
        self.ukf.x = state #initial state
        self.ukf.R = np.eye(len(state))*self.filter_params["Sensor_Noise"] #sensor noise. larger noise implies less trust in sensor and to favour prediction
        self.ukf.P = np.eye(len(state))*self.filter_params["Process_Noise"]
        
        for i in range(int(len(state)/2)):  #! initialise process noise as blocked structure of discrete white noise. cant think of any better way to do this
            i1 = 2*i
            i2 = (2*(i+1))
        #    self.ukf.Q[i1:i2,i1:i2] =  QDWN(2,dt=1,var=2*self.filter_params["Process_Noise"])  
        #    self.ukf.Q[i1:i2,i1:i2] = np.array([[2,1],[1,2]])*self.filter_params["Process_Noise"]

        self.ukf.Q = np.eye(len(state))*self.filter_params["Process_Noise"]
        self.UKF_histories.append(self.ukf.x)


    
    def main(self):
        time1 = datetime.datetime.now()
       
        
        truth = np.load(self.filter_params["batch_file"])
        truth = truth[:,self.index2]
        truth_list = [truth[j,:] for j in range(truth.shape[0])]#pull rows into lists
        #!! add sample rate reduction later
        
        self.init_ukf()#init UK
        "this function is atrocious dont use it"
        #self.UKF_histories,self.UKF_Ps = self.ukf.batch_filter(truth_list)
        
        "instead"

        for _,z in enumerate(truth_list):
            if _%100==0:
                print(f"iterations: {_}")
            self.ukf.predict()
            self.time_id+=1
            if  self.time_id%self.filter_params["sample_rate"]==0:
                self.ukf.update(z)
                self.base_model.step()
                self.UKF_histories.append(self.ukf.x)
                self.UKF_Ps.append(self.ukf.P)
            
        time2 = datetime.datetime.now()
        print(time2-time1)    
       
    def data_parser(self,do_fill):
        #!! with nans and 
        sample_rate = self.sample_rate
        "partial true observations"
        a = {}
        "UKF predictions for observed above"
        b = np.vstack(self.UKF_histories)
        for k,index in enumerate(self.index):
            a[k] =  self.base_model.agents[index].history_loc
        
        max_iter = max([len(value) for value in a.values()])

        a2= np.zeros((max_iter,self.sample_size*2))*np.nan
        b2= np.zeros((max_iter,self.sample_size*2))*np.nan
        
        #!!possibly change to make NAs at end be last position
        for i in range(int(a2.shape[1]/2)):
            a3 = np.vstack(list(a.values())[i])
            a2[:a3.shape[0],(2*i):(2*i)+2] = a3
            if do_fill:
                a2[a3.shape[0]:,(2*i):(2*i)+2] = a3[-1,:]
        
        
        for j in range(int(b2.shape[0]//sample_rate)):
            b2[j*sample_rate,:] = b[j,:]
            
        "all agent observations"
        a_full = {}
        
        for k,agent in  enumerate(self.base_model.agents):
            a_full[k] =  agent.history_loc
            
        max_iter = max([len(value) for value in a_full.values()])
        a2_full= np.zeros((max_iter,self.pop_total*2))*np.nan
        
        for i in range(int(a2_full.shape[1]/2)):
            a3_full = np.vstack(list(a_full.values())[i])
            a2_full[:a3_full.shape[0],(2*i):(2*i)+2] = a3_full
            if do_fill:
                a2_full[a3_full.shape[0]:,(2*i):(2*i)+2] = a3_full[-1,:]

        return a2,b2,a2_full
    
    def plots(self):
        a ,b,a_full = self.data_parser(False)
  
        plt.figure()
        for j in range(int(model_params["pop_total"]*self.filter_params["prop"])):
            plt.plot(a[:,2*j],a[:,(2*j)+1])    
            plt.title("True Positions")

        plt.figure()
        for j in range(int(model_params["pop_total"]*self.filter_params["prop"])):
            plt.plot(b[::self.sample_rate,2*j],b[::self.sample_rate,(2*j)+1])    
            plt.title("KF predictions")

            
        
            
        "find mean error between agent and prediction"
        c = {}
        c_means = []
        
       
        
        for i in range(int(b.shape[1]/2)):
            a_2 =   a[:,(2*i):(2*i)+2] 
            b_2 =   b[:,(2*i):(2*i)+2] 
    

            c[i] = []
            for k in range(a_2.shape[0]):
                if np.any(np.isnan(a_2[k,:])) or np.any(np.isnan(b_2[k,:])):
                    c[i].append(np.nan)
                else:                       
                    c[i].append(dist.euclidean(a_2[k,:],b_2[k,:]))
                
            c_means.append(np.nanmean(c[i]))
        
        c = np.vstack(c.values())
        time_means = np.nanmean(c,axis=0)
        plt.figure()
        plt.plot(time_means[::self.sample_rate])
        plt.axhline(y=0,color="r")
        plt.title("MAE over time")
    
            
            
        index = np.where(c_means == np.nanmax(c_means))[0][0]
        print(index)
        a1 = a[:,(2*index):(2*index)+2]
        b1 = b[:,(2*index):(2*index)+2]
        
        plt.figure()
        plt.plot(a1[:,0],a1[:,1],label= "True Path")
        plt.plot(b1[::self.sample_rate,0],b1[::self.sample_rate,1],label = "KF Prediction")
        plt.legend()
        plt.title("Worst agent")
                  

        return c_means,time_means
if __name__ == "__main__":
    np.random.seed(seed = 8)#seeding if  wanted else hash it
    model_params = {
                    'width': 200,
                    'height': 100,
                    'pop_total': 300,
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
                    "Sensor_Noise": 1,
                    "Process_Noise": 1,
                    'sample_rate': 1,
                    "do_restrict": True,
                    "prop": 0.01,
                    "heatmap_rate": 5,
                    #'batch_file': "truth.npy"

                    'batch_file': f"""ACTUAL_TRACKS_{model_params["pop_total"]}_0.npy"""

                    }
        
    


    runs = 1
    for i in range(runs):
       
        U = UKF(Model, model_params,filter_params)
        U.main()
        
        if runs==1 and model_params["do_save"] == True:   
            c_mean,t_mean = U.plots()
        a,b,a_full = U.data_parser(True)
        #pop = model_params["pop_total"]
        #np.save(f"UKF_TRACKS_{pop}_{i}",a)
        #np.save(f"ACTUAL_TRACKS_{pop}_{i}",b)




