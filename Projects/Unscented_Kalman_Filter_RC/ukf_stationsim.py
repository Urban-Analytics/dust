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

UKF filter using own function rather than filterpys

based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
"""

#import pip packages
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import datetime
import pickle
from multiprocessing import Pool
#import local packages
from StationSim_Wiggle import Model
from ukf_plots import plots,animations
from ukf import ukf
#for dark plots. purely an aesthetic choice.
plt.style.use("dark_background")

class ukf_ss:
    """
    UKF for station sim using UKF filter "ukf.py" programmed by RC.
    """
    def __init__(self,model_params,filter_params,ukf_params):
        """
        *_params - loads in parameters for the model, station sim filter and general UKF parameters
        base_model - initiate stationsim 
        pop_total - population total
        number_of_iterations - how many steps for station sim
        sample_rate - how often to update the kalman filter. intigers greater than 1 repeatedly step the station sim forward
        sample_size - how many agents observed if prop is 1 then sample_size is same as pop_total
        index and index 2 - indicate which agents are being observed
        ukf_histories- placeholder to store ukf trajectories
        time1 - initial time used to calculate run time 
        """
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
        For station sim this is essentially a placeholder step with
        varying initial locations depending on each sigma point.
        

        """
        """
        call placeholder base_model created in main via pickling
        this is done to essentially rollback the basemodel as stepping 
        it is non-invertible (afaik).
        """
        f = open(f"temp_pickle_model_ukf_{self.time1}","rb")
        model = pickle.load(f)
        f.close()
        
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
        state = state[self.index2]
        
        return state
    
    def init_ukf(self):
        "initialise ukf with initial state and covariance structures."
        state = self.base_model.agents2state(self)
        Q = np.eye(self.pop_total*2)
        R = np.eye(len(self.index2))
        P = np.eye(self.pop_total)
        self.ukf = ukf(ukf_params,state,self.fx,self.hx,P,Q,R)
        self.ukf_histories.append(self.ukf.x) #
    
    
    def main(self):
        """
        main function for ukf station sim
        -initiates ukf
        -predict with ukf
        -step true model
        -update ukf with new model positions
        -repeat until model ends or max iterations reached
        
        """
        np.random.seed(seed = 8)#seeding if  wanted else hash it
        #np.random.seed(seed = 7)# another seed if  wanted else hash it

        
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
            

            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state() #observed agents states
                self.ukf.update(z=state[self.index2]) #update UKF
                self.ukf_histories.append(self.ukf.x) #append histories
                
                x = self.ukf.x
                if np.sum(np.isnan(x))==x.shape[0]:
                    print("math error. try larger values of alpha else check fx and hx.")
                    break
                ""
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
            'pop_total': 50,
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
            
    filter_params = {      
           
            "Sensor_Noise":  1, 
            "Process_Noise": 1, 
            'sample_rate': 1,
            "do_restrict": True, 
            "do_animate": False,
            "do_wiggle_animate": False,
            "do_density_animate":False,
            "do_pair_animate":False,
            "prop": 0.1,
            "heatmap_rate": 2,
            "bin_size":10,
            "do_batch":False
            }
    """
            a - alpha between 1 and 1e-4 typically determines spread of sigma points.
            b - beta set to 2 for gaussian. determines trust in prior distribution.
            k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
                not 100% sure what kappa does. think its a bias parameter.
            !! might be worth making an interactive notebook that varies these. for fun

            """
    ukf_params = {
            
            "a":1,
            "b":2,
            "k":0,
            "d_rate" : 10, #data assimilotion rate every "d_rate model steps recalibrate UKF positions with truth

            }
    
    """run and extract data"""
    u = ukf_ss(model_params,filter_params,ukf_params)
    u.main()
    actual,preds= u.data_parser(True)
            
    """plots"""
    if filter_params["prop"]<1:
        distances,t_mean = plots.diagnostic_plots(u,actual,preds,False,False)
    distances2,t_mean2 = plots.diagnostic_plots(u,actual,preds,True,False)