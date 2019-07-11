# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019

@author: RC


UKF filter using own function rather than filterpys
arc adapted so saves files for downloading from remote
 and post-hoc plotting using arc_ukf_plots.py
"""


import numpy as np
from math import floor
import matplotlib.pyplot as plt
import datetime
import pickle


from StationSim_Wiggle import Model
from ukf_plots import plots,animations
from ukf import ukf

plt.style.use("dark_background")

class ukf_ss:
    def __init__(self,model_params,filter_params,ukf_params):
        """various inits for parameters,storage, indexing and more"""
        #call params
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
        self.ukf_params = ukf_params
        
        self.base_model = Model(self.model_params) #station sim

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
        
        self.ukf_histories = []
   
        self.time1 =  datetime.datetime.now()#timer

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
        """
        #maybe call this once before ukf.predict() rather than 5? times. seems slow        
        f = open(f"temp_pickle_model_ukf_{self.time1}","rb")
        model = pickle.load(f)
        f.close()
        
        model.state2agents(state = x)    
        model.step()
        state = model.agents2state()
        return state
   
    def hx(self,state,**hx_args):
        """
        Measurement function for agent.
        !!im guessing this is just the output from base_model.step
        take full state return those observed and NaNs otherwise
        
        """
        #state = state[self.index2]
        #mask = np.ones_like(state)
        #mask[self.index2]=False
        #z = state[np.where(mask==0)]
        
        state = state[self.index2]
        
        return state
    
    def init_ukf(self):
        state = self.base_model.agents2state(self)
        Q = np.eye(self.pop_total*2)
        R = np.eye(len(self.index2))
        self.ukf = ukf(ukf_params,state,self.fx,self.hx,Q,R)
        
        self.ukf_histories.append(self.ukf.x)
    
    
    def main(self):
        np.random.seed(seed = 8)#seeding if  wanted else hash it
        #np.random.seed(seed = 7)#seeding if  wanted else hash it

        
        self.init_ukf() 
        for _ in range(self.number_of_iterations-1):
            if _%100 ==0: #progress bar
                print(f"iterations: {_}")
                

            f_name = f"temp_pickle_model_ukf_{self.time1}"
            f = open(f_name,"wb")
            pickle.dump(self.base_model,f)
            f.close()
            
            
            self.ukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            os.remove(f"temp_pickle_model_ukf_{self.time1}")

            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state() #observed agents states
                self.ukf.update(z=state[self.index2]) #update UKF
                self.ukf_histories.append(self.ukf.x) #append histories
                
                x = self.ukf.x
                if np.isnan(x[0]):
                    print("math error. try larger values of alpha else check fx and hx.")
                    break
                ""
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        time2 = datetime.datetime.now()#timer
        print(time2-self.time1)
        
    def data_parser(self,do_fill):
        #!! with nans and 
        sample_rate = self.sample_rate
        "partial true observations"
        "UKF predictions for observed above"

        a2 = {}
        for k,agent in  enumerate(self.base_model.agents):
            a2[k] =  agent.history_loc
        max_iter = max([len(value) for value in a2.values()])
        b2 = np.vstack(self.ukf_histories)
        
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
                    "Process_Noise": 1, #how reliable is prediction fx lower value implies more reliable
                    'sample_rate': 1,   #how often to update kalman filter. higher number gives smoother (maybe oversmoothed) predictions
                    "do_restrict": True, #"restrict to a proportion prop of the agents being observed"
                    "do_animate": False,#"do animations of agent/wiggle aggregates"
                    "do_wiggle_animate": False,
                    "do_density_animate":False,
                    "do_pair_animate":False,
                    "prop": 0.5,#proportion of agents observed. 1 is all <1/pop_total is none
                    "heatmap_rate": 2,# "after how many updates to record a frame"
                    "bin_size":10,
                    "do_batch":False
                    }
    
    ukf_params = {
            "a":1,#alpha between 1 and 1e-4 typically
            "b":2,#beta set to 2 for gaussian 
            "k":0,#kappa usually 0 for state estimation and 3-dim(state) for parameters
            "d_rate" : 10,

            }
    
    
    u = ukf_ss(model_params,filter_params,ukf_params)
    u.main()
    a,b= u.data_parser(True)
    
    np.save(f"actual_{u.pop_total}",a)
    np.save(f"ukf_{u.pop_total}",b)
    np.save(f"index_{u.pop_total}",u.index2)
    
    pop = model_params["pop_total"]
    f_name = f"pickle_model_ukf_{pop}"
    f = open(f_name,"wb")
    pickle.dump(u,f)
    f.close()
    
    
    
    #scp medrclaa@arc3.leeds.ac.uk:actual_100.npy /home/rob/
    #scp medrclaa@arc3.leeds.ac.uk:ukf_100.npy /home/rob/
    #scp medrclaa@arc3.leeds.ac.uk:index_100.npy /home/rob/

    