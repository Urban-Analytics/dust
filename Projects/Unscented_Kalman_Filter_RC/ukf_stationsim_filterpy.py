#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:28:35 2019
@author: rob
Sequential/Batch UKF for station sim
!! used for 
"""


import os
os.chdir("/home/rob/dust/Projects/Unscented_Kalman_Filter_RC") #linux root
import numpy as np
from math import floor
import pickle
import datetime
from copy import deepcopy

from filterpy.kalman import MerweScaledSigmaPoints as MSSP
from filterpy.kalman import UnscentedKalmanFilter as UNKF
from filterpy.common import Q_discrete_white_noise as QDWN

import matplotlib.pyplot as plt
plt.style.use("dark_background")

from StationSim_Wiggle import Model,Agent
from ukf_plots import plots,animations

sqrt2 = np.sqrt(2)

class UKF:
    """
    individually assign a UKF wrapper to each agent 
    update each agent individually depending on whether it is currently active within the model
    whether KF updates is determined by the "activity matrix" sourced at each time step from agent.active propertysqrt2 = np.sqrt(2)
    if active, update KF using standard steps
    this allows for comphrehensive histories to be stored in KF objects
    
    """
    
    def __init__(self,Model,model_params,filter_params):
        """various inits for parameters,storage, indexing and more"""
        #call params
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
        
        #these are used alot i figure its cleaner to call them here
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
        
        #frame counts for animations
        self.frame_number = 1
        self.wiggle_frame_number = 1
        #recording 2dhistograms for wiggle densities
        
        self.UKF_histories = []
        self.Ps = []
        self.densities = []
        self.kf_densities = []
        self.wiggle_densities= {}
        self.diff_densities = []

    
    def F_x(self,x,dt,time1):
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
        #maybe call this once before ukf.predict() rather than 5? times. seems slow        
        f = open(f"temp_pickle_model_ukf_{time1}","rb")
        model = pickle.load(f)
        f.close()
        
        model.state2agents(state = x)    
        model.step()
        state = model.agents2state()
        return state
   
    def H_x(self,state):
        """
        Measurement function for agent.
        !!im guessing this is just the output from base_model.step
        take full state return those observed and NaNs otherwise
        
        """
        #state = state[self.index2]
        #mask = np.ones_like(state)
        #mask[self.index2]=False
        #z = state[np.where(mask==0)]
        
        z = state[self.index2]
        
        return z
    
    """
    initialises UKFs various parameter
    sigmas(points) - sigmas points to be used. numerous types to be found
    heremost common VDM sigmas are used.
    Q,R -noise structures for transition and measurement functions respectively
    P- Initial guess/estimate for covariance structure of state space.
    x - current states
    
    !!search for better structures for P and Q or use adaptive filtering 
    """   
    
    def init_ukf(self):
        state = self.base_model.agents2state(self)
        #state = state[self.index2] #observed agents only
        sigmas = MSSP(n=len(state),alpha=1,beta=.2,kappa=1) #sigmapoints
        self.ukf =  UNKF(dim_x=len(state),dim_z=len(state)
                        ,fx = self.F_x, hx=self.H_x
                        , dt=1, points=sigmas)#init UKF
        self.ukf.x = state #initial state
        #sensor noise. larger noise implies less trust in sensor and to favour prediction
        self.ukf.R = np.eye(len(self.index2))*self.filter_params["Sensor_Noise"]
        #initial guess for state space uncertainty
        self.ukf.P = np.eye(len(state))
        
        """
        various structures for Q. either block diagonal structure
        I.E assume agents independent but xy states within agents are not
        or standard diagonal and all state elements are independent
        """
        #for i in range(int(len(state)/2)):  #! initialise process noise as blocked structure of discrete white noise. cant think of any better way to do this
        #    i1 = 2*i
        #    i2 = (2*(i+1))
        #    self.ukf.Q[i1:i2,i1:i2] =  QDWN(2,dt=1,var=self.filter_params["Process_Noise"])  
            #self.ukf.Q[i1:i2,i1:i2] = np.array([[2,1],[1,2]])*self.filter_params["Process_Noise"]
            
        self.ukf.Q = np.eye(len(state))*self.filter_params["Process_Noise"]
        self.UKF_histories.append(self.ukf.x)
        
    

    

            
    """
    main function runs UKF over stationsim
    """
    def main_sequential(self):
        time1 = datetime.datetime.now()
        np.random.seed(seed = 8)#seeding if  wanted else hash it
        self.base_model = Model(self.model_params) #station sim
        f_name = f"base_model_{self.pop_total}"
        f = open(f_name,"wb")
        pickle.dump(self.base_model,f)
        f.close()
        self.init_ukf() #init UKF        

        for _ in range(self.number_of_iterations-1): #cycle over batch iterations. tqdm gives progress bar on for loop.
            if _%100 ==0: #progress bar
                print(f"iterations: {_}")
                                
            #!! redo this with pickles seems cleaner?
            
            f_name = f"temp_pickle_model_ukf_{time1}"
            f = open(f_name,"wb")
            pickle.dump(self.base_model,f)
            f.close()
                            
            self.ukf.predict(self.base_model,time1 = time1) #predict where agents will jump
            os.remove(f"temp_pickle_model_ukf_{time1}")
            self.base_model.step() #jump stationsim agents forwards
            
            #if _%self.filter_params["heatmap_rate"] == 0 :  #take frames for wiggles instead similar to heatmap above
            #    if self.filter_params["do_wiggle_animate"]:
            #        plots.wiggle_heatmap(self)
                    
            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state()[self.index2] #observed agents states
                self.ukf.update(z=state) #update UKF
                self.UKF_histories.append(self.ukf.x) #append histories
                self.Ps.append(self.ukf.P)

                #if _%self.filter_params["assimilation_rate"]==0:
                #    self.ukf.x = self.base_model.agents2state()
            
               
                
                if self.base_model.pop_finished == self.pop_total: #break condition
                    if self.filter_params["do_animate"]:
                        plots.heatmap(self)
                    if self.filter_params["do_animate"]:
                        animations.animate(self,"output","heatmap")           
                    if self.filter_params["do_wiggle_animate"]:
                        animations.animate(self,"output_wiggle","wiggle")
                    if self.filter_params["do_density_animate"]:
                        plots.density_frames(self)
                        animations.animate(self,"output_diff","diff")
                    if self.filter_params["do_pair_animate"]:
                        plots.pair_frames(self)
                        animations.animate(self,"output_pairs","pairs")
                    
                    break
        
        time2 = datetime.datetime.now()
        print(time2-time1)
        return        
    
    def main_batch(self):
        """
        main ukf function for batch comparing some truth data against new predictions
        
        """
        time1 = datetime.datetime.now()#timer
       
        
        truth = np.load(f"ACTUAL_TRACKS_{self.pop_total}_0.npy")
        f_name = f"base_model_{self.pop_total}"
        f = open(f_name,"rb")
        self.base_model = pickle.load(f)
        f.close()
        
        
        truth = truth[1:,:]#cut start
        truth = truth[::self.filter_params["sample_rate"],self.index2]
        truth_list = [truth[j,:] for j in range(truth.shape[0])]#pull rows into lists
        np.random.seed(seed = 8)#seeding if  wanted else hash it
        self.init_ukf()#init UK

        for _,z in enumerate(truth_list):
            if _%100==0:
                print(f"iterations: {_}")
            
            f_name = f"temp_pickle_model_ukf"
            f = open(f_name,"wb")
            pickle.dump(self.base_model,f)
            f.close()
            
            
            self.ukf.predict(self.base_model)
            os.remove("temp_pickle_model_ukf")
            self.base_model.step()
            if  _%self.filter_params["sample_rate"]==0:
                #look into residual z and see if can process as shorter or use nans?
                self.ukf.update(z)
                self.UKF_histories.append(self.ukf.x)
                self.Ps.append(self.ukf.P)
            

        time2 = datetime.datetime.now()#end timer
        print(f"time taken: {time2-time1}")    #print elapsed time
        
    """extract data from lists of agents into more suitable frames"""
    
    def data_parser(self,do_fill):
        #!! with nans and 
        sample_rate = self.sample_rate
        "partial true observations"
        "UKF predictions for observed above"

        a2 = {}
        for k,agent in  enumerate(self.base_model.agents):
            a2[k] =  agent.history_loc
        max_iter = max([len(value) for value in a2.values()])
        b2 = np.vstack(self.UKF_histories)
        
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
    
    
    def cov2cor(self,cov):
        """
        function to convert square cov to corr matrix
        """
        n = cov.shape[0]
        variances = np.diag(cov)
        deviations = np.sqrt(variances)
        
        for i in range(n): 
            cov[i,:]/= deviations[i]
            cov[:,i]/= deviations[i]

    
        
    
    
if __name__ == "__main__":
    #np.random.seed(seed = 8)#seeding if  wanted else hash it
    model_params = {
                    'width': 200,
                    'height': 100,
                    'pop_total': 10,
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
                    "Sensor_Noise":  1, # how reliable are measurements H_x. lower value implies more reliable
                    "Process_Noise": 1, #how reliable is prediction F_x lower value implies more reliable
                    'sample_rate': 1,   #how often to update kalman filter. higher number gives smoother (maybe oversmoothed) predictions
                    "assimilation_rate": 50, #every assimilate_rate iterations 
                    "do_restrict": True, #"restrict to a proportion prop of the agents being observed"
                    "do_animate": True,#"do animations of agent/wiggle aggregates"
                    "do_wiggle_animate": False,
                    "do_density_animate":False,
                    "do_pair_animate":False,
                    "prop": 0.5,#proportion of agents observed. 1 is all <1/pop_total is none
                    "heatmap_rate": 2,# "after how many updates to record a frame"
                    "bin_size":10,
                    "do_batch":False
                    }
        
    

    
    runs = 1
    for i in range(runs):
        if not filter_params["do_batch"]:
            U = UKF(Model, model_params,filter_params)
            U.main_sequential()
            
            if runs==1 and model_params["do_save"] == True:   #plat results of single run

            save=True
            if save:
                a,b = U.data_parser(True)
                pop = model_params["pop_total"]
                np.save(f"ACTUAL_TRACKS_{pop}_{i}",a)
                np.save(f"UKF_TRACKS_{pop}_{i}",b)
                #entrance_times = np.array([agent.time_start for agent in U.base_model.agents])
                #np.save(f"{pop}_entrance_times",entrance_times)
                
            if filter_params["prop"]<1:
                c_mean,t_mean = plots.diagnostic_plots(U,False,True)
            c_mean2,t_mean2 = plots.diagnostic_plots(U,True,True)

            
            

        else:
            U = UKF(Model, model_params,filter_params)
            U.main_batch()
            a,b = U.data_parser(False)

            if runs==1 and model_params["do_save"] == True:   #plat results of single run
                c_mean,t_mean = plots.plots()






