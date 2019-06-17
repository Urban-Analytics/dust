#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:28:35 2019
@author: rob
Sequential/Batch UKF for station sim
!! used for 
"""


import os
os.chdir("/home/rob/dust/Projects/Unscented_Kalman_Filter_RC")
import numpy as np
from math import floor
from StationSim_Wiggle import Model,Agent
from filterpy.kalman import MerweScaledSigmaPoints as MSSP
from filterpy.kalman import UnscentedKalmanFilter as UNKF
from filterpy.common import Q_discrete_white_noise as QDWN
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import datetime
import imageio
import matplotlib.cm as cm
from copy import deepcopy
import pickle
import matplotlib.colors as col
from custom_norm import DivergingNorm, DoubleDivergingNorm
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

    
    def F_x(self,x,dt,model1,model2):
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
        model2.step()
        state = model2.agents2state()
        state=state[self.index2]
        model2 = model1
        return state
   
    def H_x(location,z):
        """
        Measurement function for agent.
        !!im guessing this is just the output from base_model.step
        """
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
        state = state[self.index2] #observed agents only
        sigmas = MSSP(n=len(state),alpha=1,beta=.2,kappa=1) #sigmapoints
        self.ukf =  UNKF(dim_x=len(state),dim_z=len(state)
                        ,fx = self.F_x, hx=self.H_x
                        , dt=1, points=sigmas)#init UKF
        self.ukf.x = state #initial state
        #sensor noise. larger noise implies less trust in sensor and to favour prediction
        self.ukf.R = np.eye(len(state))*self.filter_params["Sensor_Noise"]
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
        
    
    """extract data from lists of agents into more suitable frames"""
    
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
    
    """diagnostic plots plotting truth paths, kf paths, MAE over time 
    and worst performing agent"""
    
    def plots(self):
        a ,b,a_full = self.data_parser(False)
        if self.filter_params["do_batch"]:
            pop = self.model_params["pop_total"]
            a = np.load(f"ACTUAL_TRACKS_{pop}_0.npy")
            a = a[:,self.index2]
  
        plt.figure()
        for j in range(int(model_params["pop_total"]*self.filter_params["prop"])):
            plt.plot(a[:,(2*j)],a[:,(2*j)+1])    
            plt.title("True Positions")

        plt.figure()
        for j in range(int(model_params["pop_total"]*self.filter_params["prop"])):
            plt.plot(b[::self.sample_rate,2*j],b[::self.sample_rate,(2*j)+1])    
            plt.title("KF predictions")

            
        
            
        """MAE metric. 
        finds mean average euclidean error at each time step and per each agent"""
        c = {}
        c_means = []
        
       
        
        for i in range(int(a.shape[1]/2)):
            a_2 =   a[:,(2*i):(2*i)+2] 
            b_2 =   b[:,(2*i):(2*i)+2] 
    

            c[i] = []
            for k in range(floor(np.min([a.shape[0],b.shape[0]]))):
                if np.any(np.isnan(a_2[k,:])) or np.any(np.isnan(b_2[k,:])):
                    c[i].append(np.nan)
                else:                       
                    c[i].append(dist.euclidean(a_2[k,:],b_2[k,:]))
                
            c_means.append(np.nanmean(c[i]))
        
        c = np.vstack(list(c.values()))
        time_means = np.nanmean(c,axis=0)
        plt.figure()
        plt.plot(time_means[::self.sample_rate])
        plt.axhline(y=0,color="r")
        plt.title("MAE over time")
    
            
        """find agent with highest MAE and plot it.
        mainly done to check something odd isnt happening"""
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


    """
    3 functions for animations of agents/wiggle locations/counts
    """
    def heatmap(self):
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        sample_agents = self.base_model.agents
        locs = [agent.location for agent in sample_agents]
        locs = np.vstack(locs)
        bins = self.filter_params["bin_size"]
        width = self.model_params["width"]
        height = self.model_params["height"]


        f = plt.figure()
        ax = f.add_subplot(111)
        
        plt.scatter(locs[:,0],locs[:,1],color="cyan")
        ax.set_ylim(0,height)
        ax.set_xlim(0,width)        
        hist,xb,yb = np.histogram2d(locs[:,0],locs[:,1],
                                    range = [[0,width],[0,height]],
                                    bins = [2*bins,bins],density=True)
        hist *= bins**2
        hist= hist.T
        hist = np.flip(hist,axis=0)

        extent = [0,width,0,height]
        plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                   ,cmap = cm.Spectral ,extent=extent
                   ,norm=DivergingNorm(vmin=1e-10,vcenter=0.1,vmax=1))
        
        ticks = np.array([0.001,0.1,0.2,0.5,1.0])
        cbar = plt.colorbar(fraction=0.046,pad=0.04,shrink=0.71,
                            ticks = ticks,space="uniform")
        plt.clim(0,1)
        cbar.set_alpha(1)
        cbar.draw_all()
            
        plt.xlabel("Corridor width")
        plt.ylabel("Corridor height")
        cbar.set_label("Agent Density (x100%)") 
        number = str(self.frame_number).zfill(5)
        file = f"output/heatmap{number}"
        f.savefig(file)
        plt.close()
        self.frame_number+=1


    def wiggle_heatmap(self):
            #sample_agents = [self.base_model.agents[j] for j in self.index]
            #swap if restricting observed wiggles
            sample_agents = self.base_model.agents
            wiggles = np.array([agent.wiggle for agent in sample_agents])
            #are you having a wiggle m8
            index = np.where(wiggles==1)
            non_index = np.where(wiggles==0)
            #sort locations
            locs = [agent.location for agent in sample_agents]
            locs = np.vstack(locs)
            non_locs = locs[non_index,:][0,:,:]
            locs = locs[index,:][0,:,:]
            #initiate figure /axes
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            bins = self.filter_params["bin_size"]
            width = self.model_params["width"]
            height = self.model_params["height"]

            #plot non-wigglers and set plot size
            plt.scatter(non_locs[:,0],non_locs[:,1],color="cyan")
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)
            cmap = cm.Spectral

            #check for any wigglers and plot the 2dhist 
            if np.sum(wiggles)!=0:
                plt.scatter(locs[:,0],locs[:,1],color="magenta")
                hist,xb,yb = np.histogram2d(locs[:,0],locs[:,1],
                                            range = [[0,width],[0,height]],
                                            bins = [2*bins,bins],density=True)  #!! some formula for bins to make even binxbin squares??  
                hist *= bins**2
                hist= hist.T
                hist = np.flip(hist,axis=0)
                self.wiggle_densities[self.wiggle_frame_number] = hist
                
                extent = [0,width,0,height]
                im=plt.imshow(np.ma.masked_where(hist==0,hist)
                           ,cmap = cmap,extent=extent,
                           norm=DivergingNorm(vmin=1e-10,vcenter=0.11,vmax=1))
                
            #if no wiggles plot a "ghost histogram" to maintain frame structure  
            else:
                #ghost histogram with one entry and (1,1)
                hist,xb,yb = np.histogram2d(np.array([1]),np.array([1]),
                                            range = [[0,width],[0,height]],
                                            bins = [bins,bins],density=True)   
               
                extent = [0,width,0,height]
                #plot ghost hist with no opacity (alpha=0) to make it invisible
                im=plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                           ,cmap = cm.Spectral ,extent=extent,alpha=0
                           ,norm=DivergingNorm(vmin=1e-10,vcenter=0.1,vmax=1))
            
            #colourbar and various plot fluff
            ticks = np.array([0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
            #!! numbers adjusted by trial and error for 200x100 field. 
            #should probably generalise this and the bin structure at some point
            cbar = plt.colorbar(im,fraction=0.046,pad=0.04,shrink=0.71,
                                ticks = ticks)
            plt.clim(0,1)
            cbar.set_alpha(1)
            cbar.draw_all()
            
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            cbar.set_label("Wiggle Density (x100%)")
            
            number = str(self.wiggle_frame_number).zfill(5)
            file = f"output_wiggle/wiggle{number}"
            f.savefig(file)
            plt.close()
            self.wiggle_frame_number+=1
    

    """animates folder of frames"""
    def animate(self,file,name):
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}GIF.mp4', images,fps=10)
        
        self.clear_output_folder(file)
        

    """clears animated frames after being animated"""
    def clear_output_folder(self,file_name):
       folder = file_name
       for the_file in os.listdir(folder):
           file_path = os.path.join(folder, the_file)
           try:
               if os.path.itime_idsfile(file_path):
                   os.unlink(file_path)
           except Exception as e:
               print(e)
            
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
                
            if _%self.filter_params["heatmap_rate"] == 0 :  #take frames for animations every heatmap_rate loops
                if self.filter_params["do_animate"]:
                    self.heatmap()
            #!! redo this with pickles seems cleaner?
            model1=model2=deepcopy(self.base_model)
            self.ukf.predict(self, model1=model1,model2=model2) #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            
            if _%self.filter_params["heatmap_rate"] == 0 :  #take frames for wiggles instead similar to heatmap above
                if self.filter_params["do_wiggle_animate"]:
                    self.wiggle_heatmap()
                    
            if self.base_model.time_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.agents2state()[self.index2] #observed agents states
                self.ukf.update(z=state) #update UKF
                self.UKF_histories.append(self.ukf.x) #append histories

                
               
                
                if self.base_model.pop_finished == self.pop_total: #break condition
                    #if self.filter_params["do_animate"]:
                    #    self.animate("output","heatmap")           
                    #if self.filter_params["do_wiggle_animate"]:
                    #    self.animate("output_wiggle","wiggle")
                    self.density_frames()
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
            model1=deepcopy(self.base_model)
            model2=deepcopy(self.base_model)
            self.ukf.predict(model1=model1,model2=model2)
            self.base_model.step()
            if  _%self.filter_params["sample_rate"]==0:
                self.ukf.update(z)
                self.UKF_histories.append(self.ukf.x)
                self.Ps.append(self.ukf.P)
            
        time2 = datetime.datetime.now()#end timer
        print(f"time taken: {time2-time1}")    #print elapsed time
        
        
    def density_frames(self):
        "snapshots of densities"
        a,b,a_full = U.data_parser(False)
        bins = self.filter_params["bin_size"]
        width = self.model_params["width"]
        height = self.model_params["height"]
        #generate full from observed
        first_time =True
        if first_time:
            for _ in range(a.shape[0]):
                hista,xb,yb = np.histogram2d(a[_,::2],a[_,1::2],
                                        range = [[0,width],[0,height]],
                                        bins = [2*bins,bins],density=True)
                hista *= bins**2
                hista= hista.T
                hista = np.flip(hista,axis=0)
                self.densities.append(hista)
        
        for _ in range(b.shape[0]):
            histb,xb,yb = np.histogram2d(b[_,0::2],b[_,1::2],
                                    range = [[0,width],[0,height]],
                                    bins = [2*bins,bins],density=True)
            histb *= bins**2
            histb= histb.T
            histb = np.flip(histb,axis=0)
            self.kf_densities.append(histb)
    
        for _ in range(len(self.densities)):
           self.diff_densities.append(np.abs(self.densities[_]-self.kf_densities[_]))
           
        c = np.dstack(self.diff_densities)
        
        for _ in range(1,c.shape[2]):
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            bins = self.filter_params["bin_size"]
            width = self.model_params["width"]
            height = self.model_params["height"]
            #plot non-wigglers and set plot size
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)

            cmap = cm.Spectral
            cmap.set_bad(color="black")
            #check for any wigglers and plot the 2dhist 
            if np.sum(c[:,:,_])!=0:
                hista = c[:,:,_]
                extent = [0,width,0,height]
                im=plt.imshow(np.ma.masked_where(hista==0,hista)
                           ,cmap = cmap,extent=extent,norm=DoubleDivergingNorm(vcenter=0.05))
                
            #if no wiggles plot a "ghost histogram" to maintain frame structure  
            else:
                #ghost histogram with one entry and (1,1)
                hist,xb,yb = np.histogram2d(np.array([-1]),np.array([-1]),
                                            range = [[0,width],[0,height]],
                                            bins = [bins,bins],density=True)   
               
                extent = [0,width,0,height]
                #plot ghost hist with no opacity (alpha=0) to make it invisible
                im=plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                              ,cmap = cmap ,extent=extent,alpha=1
                              ,norm=DoubleDivergingNorm(vmin=-0.5,vcenter=0.05,vmax=0.5))
            #colourbar and various plot fluff
            ticks = np.array([-0.5,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.5])
            #!! numbers adjusted by trial and error for 200x100 field. 
            #should probably generalise this and the bin structure at some point
            cbar = plt.colorbar(im,fraction=0.046,pad=0.04,shrink=0.71,
                                ticks = ticks,spacing="proportional")
            plt.clim(-0.5,0.5)
            cbar.set_alpha(1)
            cbar.draw_all()
            
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            cbar.set_label("Wiggle Density (x100%)")
            
            number =  str(_).zfill(5)
            file = f"output_diff/wiggle{number}"
            f.savefig(file)
            plt.close()
            
        self.animate("output_diff","diff") 
        self.clear_output_folder("output_diff")
            
            
            
            

    
    
if __name__ == "__main__":
    #np.random.seed(seed = 8)#seeding if  wanted else hash it
    model_params = {
                    'width': 200,
                    'height': 100,
                    'pop_total': 100,
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
                    "do_restrict": True, #"restrict to a proportion prop of the agents being observed"
                    "do_animate": False,#"do animations of agent/wiggle aggregates"
                    "do_wiggle_animate":False,
                    "prop": 0.333,#proportion of agents observed. 1 is all <1/pop_total is none
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
                c_mean,t_mean = U.plots()
            
            
            save=True
            if save:
                a,b,a_full = U.data_parser(True)
                pop = model_params["pop_total"]
                np.save(f"ACTUAL_TRACKS_{pop}_{i}",a_full)
                np.save(f"PARTIAL_TRACKS_{pop}_{i}",a)
                np.save(f"UKF_TRACKS_{pop}_{i}",b)
                entrance_times = np.array([agent.time_start for agent in U.base_model.agents])
                np.save(f"{pop}_entrance_times",entrance_times)

        else:
            U = UKF(Model, model_params,filter_params)
            U.main_batch()
            a,b,a_full = U.data_parser(False)

            if runs==1 and model_params["do_save"] == True:   #plat results of single run
                c_mean,t_mean = U.plots()




