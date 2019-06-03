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
from StationSim_UKF import Model, Agent
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
        self.UKF = None #dictionary of KFs for each agent
        self.UKF_histories = []
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
        self.frame_number = 1


    def F_x(x,dt,self):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially gradient * v *dt assuming no collisions
        with some algebra to get it into cartesian plane.
        
        to generalise this to every agent need to find the "ideal move" for each I.E  assume no collisions
        will make class agent_ideal which essential is the same as station sim class agent but no collisions in move phase
        """
        sample_agents = [self.base_model.agents[j] for j in self.index]
        loc = [agent.ideal_location for agent in sample_agents ]
        loc = np.array(loc)
        loc = loc.flatten()
        return loc
        
        
    def H_x(location,z):
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
        sigmas = MSSP(n=len(state),alpha=.01,beta=.2,kappa=1) #sigmapoints
        self.ukf =  UNKF(dim_x=len(state),dim_z=len(state),fx = self.F_x, hx=self.H_x, dt=1, points=sigmas)#init UKF
        self.ukf.x = state #initial state
        self.ukf.R = np.eye(len(state))*self.filter_params["Sensor_Noise"] #sensor noise. larger noise implies less trust in sensor and to favour prediction
        self.ukf.P = np.eye(len(state))*self.filter_params["Process_Noise"]
        
        #for i in range(int(len(state)/2)):  #! initialise process noise as blocked structure of discrete white noise. cant think of any better way to do this
        #    i1 = 2*i
        #    i2 = (2*(i+1))
        #    self.ukf.Q[i1:i2,i1:i2] =  QDWN(2,dt=1,var=self.filter_params["Process_Noise"])  
        #    self.ukf.Q[i1:i2,i1:i2] = np.array([[2,1],[1,2]])*self.filter_params["Process_Noise"]

        self.ukf.Q = np.eye(len(state))*self.filter_params["Process_Noise"]*self.filter_params["sample_rate"]
        self.UKF_histories.append(self.ukf.x)

    
    def data_parser(self):
        sample_rate = self.sample_rate
        a = {}
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
            a2[a3.shape[0]:,(2*i):(2*i)+2] = a3[-1,:]


        for j in range(int(b2.shape[0]//sample_rate)):
            b2[j*sample_rate,:] = b[j,:]

        return a2,b2
    
    def plots(self):
        a ,b = self.data_parser()
  
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

    def heatmap(self):
        os.chdir("/home/rob/dust/Projects/RC_Scripts/output")
        locs = [agent.location for agent in self.base_model.agents]
        locs = np.vstack(locs)
        
        #os.chdir("/home/rob/dust/Projects/RC_Scripts/output")
        f = plt.figure()
        plt.hist2d(locs[:,0],locs[:,1],normed = False,bins = 10,
                   range = np.array([[0,self.model_params["width"]],[0,self.model_params["height"]]])
                   ,animated= True,cmap=cm.Spectral,cmin =1e-10 )      
        plt.xlabel("Corridor width")
        plt.ylabel("Corridor height")
        plt.colorbar()
        #plt.show()
        file = f"{self.frame_number}"
        f.savefig(file)
        os.chdir("/home/rob/dust/Projects/RC_Scripts")
        plt.close()
        self.frame_number+=1
    
    
    """
    getting files to sort numerically. can probably resolve this by naming png files better.
    """
    
    
    def animate(self):
        def tryint(s):
            try:
                return int(s)
            except:
                return s
        
        def alphanum_key(s):
            """ Turn a string into a list of string and number chunks.
                "z23a" -> ["z", 23, "a"]
            """
            return [tryint(c) for c in re.split('([0-9]+)', s) ]        
            
    
        
        files = os.listdir('output')
        files.sort(key=alphanum_key) #sorting files numerically
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread('output/{}'.format(filename)))
        imageio.mimsave('outputGIF.mp4', images,fps=10)
        
        self.clear_output_folder()

        
    def save_histories(self):
        a,b = self.data_parser()
        return a,b
    
    
    
    def batch(self):
        time1 = datetime.datetime.now()
        self.init_ukf() #init UKF
        #self.F_x(dt=1,agents = self.base_model.agents)
        

        for _ in range(self.number_of_iterations-1): #cycle over batch iterations. tqdm gives progress bar on for loop.
            if _%100 ==0:
                print(f"iterations: {_}")
                
            if _%self.filter_params["heatmap_rate"] == 0 :    
                self.heatmap()
                
            self.base_model.step() #jump agents forward using actual position
            self.ukf.predict(self) 

            if self.base_model.time_id%self.sample_rate == 0:
                
                state = self.base_model.agents2state()[self.index2]
                self.ukf.update(z=state)
                self.UKF_histories.append(self.ukf.x)
                    
                f = []
                for j in range(self.pop_total):
                    f.append(self.base_model.agents[j].active)
                g=np.array(f)
                
                if np.sum(g==2) == self.pop_total:
                    self.animate()
                    break
        self.wiggle_counts = [agent.wiggle_count for agent in U.base_model.agents]
        
        time2 = datetime.datetime.now()
        print(time2-time1)
        return
    
       
    def clear_output_folder(self):
       folder = 'output'
       for the_file in os.listdir(folder):
           file_path = os.path.join(folder, the_file)
           try:
               if os.path.isfile(file_path):
                   os.unlink(file_path)
           except Exception as e:
               print(e)
    
    
    
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
                    "Sensor_Noise": 5,
                    "Process_Noise": 2,
                    'sample_rate': 5,
                    "do_restrict": True,
                    "prop": 0.1,
                    "heatmap_rate": 5
                    }
        
    


    runs = 1
    for i in range(runs):
       
        U = UKF(Model, model_params,filter_params)
        U.batch()
        
        if runs==1 and model_params["do_save"] == True:   
            c_mean,t_mean = U.plots()
        a,b = U.save_histories()
        pop = model_params["pop_total"]
        np.save(f"UKF_TRACKS_{pop}_{i}",a)
        np.save(f"ACTUAL_TRACKS_{pop}_{i}",b)






