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

SR filter generally the same as regular filters for efficiency 
but more numerically stable wrt rounding errors 
and preserving PSD covariance matrices

based on
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6179312
"""
import os
os.chdir("/home/rob/dust/Projects/RC_Scripts/")
import numpy as np
from StationSim_UKF import Model, Agent
import scipy

model_params = {
            'width': 200,
            'height': 100,
            'pop_total': 2,
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
            'batch_iterations': 10_000,
            'do_save': True,
            'do_plot': True,
            'do_ani': False
        }


class SRUKF():
    
    def __init__(self,model_params):
        """this needs to:
            - init x0, S_0,S_v and S_n
           
            
        """
        self.model_params = model_params
        self.base_model = Model(self.model_params)
        self.x0 =  Model.agents2state(self.base_model)      
        self.P = np.eye(2*model_params["pop_total"])*2
        self.Q = np.eye(2*model_params["pop_total"])
        self.h = 1.7320508 #root 3
        #self.wm = np.zeros(((2*self.x0.shape)+1,1))
        #self.wc = np.zeros(((2*self.x0.shape)+1,1))

    def P_Sigmas(self):
        """
        central differenced sigmas as per the paper.
        very similar formula to Merwe's Sigmas but based on different background
        (Stirling's Formula and various Numeric Analysis Techniques)
        takes in :
            -current state estimate (x0)
            -current process and sensor covariance matrices (P,Q)
            
        """
        self.x0 = np.append(self.x0,np.zeros((2*self.model_params["pop_total"],)))
        self.sqrtP = scipy.linalg.cholesky(self.P) #square roots of process/noise covariances
        self.sqrtQ = scipy.linalg.cholesky(self.Q)
        self.S =  scipy.linalg.block_diag(self.sqrtP,self.sqrtQ)#blocking of above covariances
        
        self.psigmas = np.zeros(((2*self.x0.shape[0])+1,self.x0.shape[0])) #calculating sigmas using x0,h and S
        self.psigmas[0,:]= self.x0
        for i in range(self.x0.shape[0]):
            self.psigmas[i+1,:] = self.x0 + self.h*self.S[:,i]
            self.psigmas[self.x0.shape[0]+i+1,:] = self.x0 - self.h*self.S[:,i]

        return self.psigmas
    
    def F_x(self,psigmas):
        """
        (non-)linear transition function taking current state space and predicting 
        innovation
        """
        fpsigmas = np.zeros(psigmas.shape)
        for j, agent in enumerate(self.base_model.agents):
            fpsigmas[0,(2*j):(2*j)+2] = agent.ideal_location
            loc1 = agent.loc_desire
            for i in range(1,int((psigmas.shape[0]/2)-1)):
                loc2 = psigmas[i,(2*j):(2*j)+2]
                reciprocal_distance = 1.41421 / sum(abs(loc1 - loc2))  # lerp5: profiled at 6.41μs
                fpsigmas[i,(2*j):(2*j)+2] = loc2 + agent.speed_desire * (loc1 - loc2) * reciprocal_distance
                
                
                loc2 = psigmas[self.model_params["pop_total"]+i-1,(2*j):(2*j)+1]
                reciprocal_distance = 1.41421 / sum(abs(loc1 - loc2))  # lerp5: profiled at 6.41μs
                fpsigmas[i+self.model_params["pop_total"],(2*j):(2*j)+2] = loc2 + agent.speed_desire * (loc1 - loc2) * reciprocal_distance
            
        return fpsigmas

    def predict(self):
        """
        predict transitions
        calculate estimates of new mean in the usual way
        calculate predicted covariance using qr decomposition
        """
        
        return
    
    def U_Sigmas(self):
        "new sigma points involving predicted prior (hatS)"
        return    
    
    def H_x(self,x):
        """
        measurement function converting state space into same dimensions 
        as measurements to assess residuals
        """
        return
    
    def update(self):
        """
        calculate residuals
        calculate kalman gain
        merge prediction with measurements in the Kalman style using
        cholesky update function
        """
        return
        
    def batch(self):
        """
        batch
        """

        sigmas = self.P_Sigmas()
        fsigmas = self.F_x(sigmas)
        return fsigmas
    
    
s = SRUKF(model_params)
sigmas = s.P_Sigmas()
fsigmas = s.F_x(sigmas)