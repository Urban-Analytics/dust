# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019
@author: RC

The Unscented Kalman Filter (UKF) designed to be hyper efficient alternative to similar 
Monte Carlo techniques such as the Particle Filter.
based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
class built into 5 steps
-init
-Prediction SP generation
-Predictions
-Update SP generation
-Update

NOTE: To avoid confusion 'observation/obs' are observed stationsim data
not to be confused with the 'observed' boolean 
determining whether to look at observed/unobserved agent subset
(maybe change the former the measurements etc.)

NOTE: __main__ here is now deprecated. use ukf notebook in experiments folder

As of 01/09/19 dependencies are: 
    pip install imageio
    pip install ffmpeg
    pip install scipy
    pip install filterpy
"""

#for filter
import numpy as np
from math import floor, log10, ceil
import matplotlib.pyplot as plt
import datetime
import multiprocessing
from copy import deepcopy
import os 
import sys 
import pickle 

#due to import errors from other directories

try:
    sys.path.append("../../../stationsim")
    from stationsim_model import Model
except:
    pass

#for plots

#from seaborn import kdeplot  # no longer used
import matplotlib.gridspec as gridspec #for nested plots in matplotlib pair_frames_stack
import imageio #for animations
from shutil import rmtree #used to keep animatons frames in order


# used for plotting covariance ellipses for each agent. 
# from filterpy.stats import covariance_ellipse  
# from scipy.stats import norm #easy l2 norming  
# from math import cos, sin

plt.rcParams.update({'font.size':20})  # make plot font bigger
#%%
class HiddenPrints:
    """stop repeating printing from stationsim
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python

    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

"""general ukf class"""
class ukf:
    """main ukf class with aggregated measurements
    
    Parameters
    ------
    ukf_params : dict
        dictionary of ukf parameters `ukf_params`
    init_x : array_like
        Initial ABM state `init_x`
    poly_list : list
        list of polygons `poly_list`
    fx,hx: method
        transitions and measurement functions `fx` `hx`
    P,Q,R : array_like
        Noise structures `P` `Q` `R`
    
    """
    
    def __init__(self,ukf_params,init_x,fx,hx,P,Q,R):
        """
        x - state
        n - state size 
        P - state covariance
        fx - transition function
        hx - measurement function
        lam - lambda paramter function of tuning parameters a,b,k
        g - gamma parameter function of tuning parameters a,b,k
        wm/wc - unscented weights for mean and covariances respectively.
        Q,R -noise structures for fx and hx
        xs,Ps - lists for storage
        """
        
        #init initial state
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = P
        #self.P = np.linalg.cholesky(self.x)
        self.fx=fx
        self.hx=hx
        
        #init further parameters based on a through el
        self.lam = ukf_params["a"]**2*(self.n+ukf_params["k"]) - self.n #lambda paramter calculated viar
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        #init weights based on paramters a through el
        main_weight =  1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-ukf_params["a"]**2+ukf_params["b"])

    
            
        self.Q=Q
        self.R=R

        self.xs = []
        self.Ps = []

    def Sigmas(self,mean,P):
        """sigma point calculations based on current mean x and covariance P
        
        Parameters
        ------
        mean , P : array_like
            mean `x` and covariance `P` numpy arrays
            
        Returns
        ------
        sigmas : array_like
            sigma point matrix
        
        """
        
     
        sigmas = np.ones((self.n,(2*self.n)+1)).T*mean
        sigmas=sigmas.T
        sigmas[:,1:self.n+1] += self.g*P #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*P #'lower' confidence sigmas
        return sigmas 

    def predict(self,**fx_args):
        """Transitions sigma points forwards using markovian transition function plus noise Q
        
        - calculate sigmas using prior mean and covariance P.
        - forecast sigmas X- for next timestep using transition function Fx.
        - unscented mean for foreacsting next state.
        - calculate interim Px
        - pass these onto next update function
        
        Parameters
        ------
        **fx_args
            generic arguments for transition function f
        """
        
        #calculate NL projection of sigmas
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.P)) #calculate current sigmas using state x and UT element S
        "numpy apply along axis or multiprocessing options"
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
        #p = multiprocessing.Pool()
        #nl_sigmas = np.vstack(p.map(self.fx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        #p.close()
        wnl_sigmas = nl_sigmas*self.wm
            
        xhat = np.sum(wnl_sigmas,axis=1)#unscented mean for predicitons
        
        """
        should be a faster way of doing this
        covariance estimation for prior P as a sum of the outer products of 
        (sigmas - unscented mean) weighted by wc
        """
        
        " quadratic form calculation of (cross/within) variances to replace old for loop"
        Pxx = np.matmul(np.matmul((nl_sigmas.transpose()-xhat).T,np.diag(self.wc)),(nl_sigmas.transpose()-xhat))+self.Q

        #Pxx =  self.wc[0]*np.outer((nl_sigmas[:,0].T-xhat),(nl_sigmas[:,0].T-xhat))+self.Q
        #for i in range(1,len(self.wc)): 
        #    Pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)
            
        self.P = Pxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z,**hx_args):     
        """ update forecasts with measurements to get posterior assimilations
        
        Does numerous things in the following order
        - nudges X- sigmas with calculated Px from predict
        - calculate measurement sigmas Y = h(X)
        - calculate unscented means of Y
        -calculate Py Pxy,K using R
        - calculate x update
        - calculate P update
        
        Parameters
        ------
        z : array_like
            measurements from sensors `z`
        **hx_args
            arbitrary arguments for measurement function
        """
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.P)) #update using Sxx and unscented mean
        #nl_sigmas = np.apply_along_axis(self.hx,0,sigmas)
        p = multiprocessing.Pool()
        nl_sigmas = np.vstack(p.map(self.hx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        p.close()
        wnl_sigmas = nl_sigmas*self.wm

        """
        unscented estimate of posterior mean using said posterior sigmas
        """
        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        
        
        "similar weighted estimates as Pxx for cross covariance and posterior covariance"
        "now (faster?) with quadratic form"
        
        Pyy = np.matmul(np.matmul((nl_sigmas.transpose()-yhat).T,np.diag(self.wc)),(nl_sigmas.transpose()-yhat))+self.R

        #Pyy =  self.wc[0]*np.outer((nl_sigmas[:,0].transpose()-yhat),(nl_sigmas[:,0].transpose()-yhat))+self.R
        #for i in range(1,len(self.wc)):
        #    Pyy += self.wc[i]*np.outer((nl_sigmas[:,i].transpose()-yhat),(nl_sigmas[:,i].transpose()-yhat))
        
        Pxy = np.matmul(np.matmul((sigmas.transpose()-self.x).T,np.diag(self.wc)),(nl_sigmas.transpose()-yhat))

        #Pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),(nl_sigmas[:,0].transpose()-yhat))
        #for i in range(1,len(self.wc)):
        #    Pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),(nl_sigmas[:,i].transpose()-yhat))
            
        "kalman gain"
        K = np.matmul(Pxy,np.linalg.inv(Pyy))
 
        #update xhat
        self.x = self.x + np.matmul(K,(z-yhat))
        self.P = self.P - np.matmul(K,np.matmul(Pyy,K.T))
        self.Ps.append(self.P)
        self.xs.append(self.x)
        
        
        
    def batch(self):
        """
        batch function maybe build later
        """
        return

class ukf_ss:
    """UKF for station sim using ukf filter class.
    
    Parameters
    ------
    model_params,filter_params,ukf_params : dict
        loads in parameters for the model, station sim filter and general UKF parameters
        `model_params`,`filter_params`,`ukf_params`
    poly_list : list
        list of polygons `poly_list`
    base_model : method
        stationsim model `base_model`
    """
    def __init__(self,model_params,filter_params,ukf_params,base_model):
        """
        *_params - loads in parameters for the model, station sim filter and general UKF parameters
        base_model - initiate stationsim 
        pop_total - population total
        number_of_iterations - how many steps for station sim
        sample_rate - how often to update the kalman filter. intigers greater than 1 repeatedly step the station sim forward
        sample_size - how many agents observed if prop is 1 then sample_size is same as pop_total
        index and index 2 - indicate which agents are being observed
        ukf_histories- placeholder to store ukf trajectories
        time1 - start gate time used to calculate run time 
        """
        # call params
        self.model_params = model_params #stationsim parameters
        self.filter_params = filter_params # ukf parameters
        self.ukf_params = ukf_params
        self.base_model = base_model #station sim
        
        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        # number of batch iterations
        self.number_of_iterations = model_params['step_limit']
        self.sample_rate = self.filter_params["sample_rate"]
        # how many agents being observed
        if self.filter_params["prop"]<1: 
            self.sample_size= floor(self.pop_total*self.filter_params["prop"])
        else:
            self.sample_size = self.pop_total
            
        #random sample of agents to be observed
        self.index = np.sort(np.random.choice(self.model_params["pop_total"],
                                                     self.sample_size,replace=False))
        self.index2 = np.empty((2*self.index.shape[0]),dtype=int)
        self.index2[0::2] = 2*self.index
        self.index2[1::2] = (2*self.index)+1
        
        self.ukf_preds=[] # fills in blanks between assimlations with pure stationsim. 
                          # good for animation. not good for error metrics use ukf_histories
        self.ukf_histories = [] # assimilated ukf positions no filled in section (this is predictions vs full predicitons above)
        self.full_Ps=[] #full covariances. again used for animations and not error metrics
        self.truths = [] # noiseless observations
        self.obs = []
        
        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
    def fx(self,x,**fx_args):
        """Transition function for the StationSim
        
        -Copies current base model
        -Replaces position with sigma point
        -Moves replaced model forwards one time point
        -record new stationsim positions as forecasted sigmapoint
        
        Parameters
        ------
        x : array_like
            Sigma point of measured state `x`
        **fx_args
            arbitrary arguments for transition function fx
            
        Returns
        -----
        state : array_like
            predicted measured state for given sigma point
        """   
            
        #f = open(f"temp_pickle_model_ukf_{self.time1}","rb")
        #model = pickle.load(f)
        #f.close()
        model = deepcopy(self.base_model)
        model.set_state(state = x,sensor="location")    
        with HiddenPrints():
            model.step() #step model with print suppression
        state = model.get_state(sensor="location")
        
        return state
   
    def hx(self,state,**hx_args):
        """Convert each sigma point from noisy gps positions into actual measurements
        
        -   omits pre-definied unobserved agents given by index/index2
        
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
        state = state[self.index2]
        
        return state
    
    def init_ukf(self,ukf_params):
        """initialise ukf with initial state and covariance structures.
        
        
        Parameters
        ------
        ukf_params : dict
            dictionary of various ukf parameters `ukf_params`
        
        """
        
        x = self.base_model.get_state(sensor="location")#initial state
        Q = np.eye(self.pop_total*2)#process noise
        R = np.eye(len(self.index2))#sensor noise
        P = np.eye(self.pop_total*2)#inital guess at state covariance
        self.ukf = ukf(ukf_params,x,self.fx,self.hx,P,Q,R)
    
    
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
        
        self.init_ukf(self.ukf_params) 
        for _ in range(self.number_of_iterations-1):

            #f_name = f"temp_pickle_model_ukf_{self.time1}"
            #f = open(f_name,"wb")
            #pickle.dump(self.base_model,f)
            #f.close()
            
            
            self.ukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            self.truths.append(self.base_model.get_state(sensor="location"))
            

                
            "DA update step and data logging"
            "data logged for full preds and only assimilated preds (just predict step or predict and update)"
            if _%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.get_state(sensor="location") #observed agents states
                "apply noise to active agents"
                if self.filter_params["bring_noise"]:
                    noise_array=np.ones(self.pop_total*2)
                    noise_array[np.repeat([agent.status!=1 for agent in self.base_model.agents],2)]=0
                    noise_array*=np.random.normal(0,self.filter_params["noise"],self.pop_total*2)
                    state+=noise_array
                    
                self.ukf.update(z=state[self.index2]) #update UKF
                
                self.ukf_histories.append(self.ukf.x) #append histories
                self.ukf_preds.append(self.ukf.x)
                status = np.repeat(np.array([float(agent.status) for agent in self.base_model.agents]),2)
                status[status!=1] = np.nan 
                self.obs.append(state*status)
                self.full_Ps.append(self.ukf.P)
                
                x = self.ukf.x
                if np.sum(np.isnan(x))==x.shape[0]:
                    print("math error. try larger values of alpha else check fx and hx.")
                    break
            else:
                "update full preds that arent assimilated"
                self.ukf_preds.append(self.ukf.x)
                self.full_Ps.append(self.ukf.P)

                ""
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        self.time2 = datetime.datetime.now()#timer
        print(self.time2-self.time1)
        
    def data_parser(self,do_fill):
        """extracts data into numpy arrays
        
        
        Returns
        ------
            
        a : array_like
            `a` noisy observations of agents positions
        b : array_like
            `b` ukf predictions of said agent positions
        c : array_like
            `c` if sampling rate >1 fills inbetween predictions with pure stationsim prediciton
                this is solely for smoother animations later
        d : 
            `d` true agent positions
            
        nan_array : array_like
            `nan_array` which entries of b are nan. good for plotting/metrics
        """
        sample_rate = self.sample_rate
        
        nan_array = np.ones(shape=(max([len(agent.history_locations) for agent in self.base_model.agents]),2*self.pop_total))*np.nan
        for i in range(self.pop_total):
            agent = self.base_model.agents[i]
            array = np.array(agent.history_locations)
            array[array==None] ==np.nan
            nan_array[:len(agent.history_locations),2*i:(2*i)+2] = array
        
        nan_array = ~np.isnan(nan_array)
        
            
        a2 =  np.vstack(self.obs) 
        b2 = np.vstack(self.ukf_histories)
        d = np.vstack(self.truths)

        
        a= np.zeros((d.shape[0],self.pop_total*2))*np.nan
        b= np.zeros((d.shape[0],b2.shape[1]))*np.nan
  

        
        for j in range(int(b.shape[0]//sample_rate)):
            a[j*sample_rate,:] = a2[j,:]
            b[j*sample_rate,:] = b2[j,:]
         

        if sample_rate>1:
            c= np.vstack(self.ukf_preds)

        
            
            "all agent observations"
        
            return a,b,c,d,nan_array
        else:
            return a,b,d,nan_array

#%%
class plots:
    """class for all plots used in UKF
    
    Parameters
    ------
    filter_class : class
        `filter_class` some finished ABM with UKF fitted to it 
    
    save_dir: string
        directory to save plots to. If using current directory "".
        e.g ukf experiments from stationsim "../experiments/ukf_experiments"
    """
    
    def __init__(self,filter_class,save_dir):
        "define which class to plot from"
        self.filter_class=filter_class
        self.save_dir = save_dir
    def plot_data_parser(self,a,b,observed):
        """takes data from ukf_ss data parser and preps it for plotting
        
        Parameters
        ------
        a,b : array_like
            `a` true measurements and `b` ukf predictions
        observed : string
            plotting observed or unobserved agents
            
        Returns
        ------
        a,b: array_like
            same measurements `a` and ukf `b` but filters for 
            observed/unobserved agents 
        plot_range : int
            how many agents are being plotted observed/unobserved.
            good for diagnostics with multiple tracks
        """

        filter_class = self.filter_class
        plot_range =floor(filter_class.model_params["pop_total"]*(filter_class.filter_params["prop"]))

        if observed:
                a = a[:,filter_class.index2]
                if len(filter_class.index2)<b.shape[1]:
                    b = b[:,filter_class.index2]

        else:      
                mask = np.ones(a.shape[1])
                mask[filter_class.index2]=0
                a = a[:,np.where(mask==1)][:,0,:]
                b = b[:,np.where(mask==1)][:,0,:]
                plot_range = filter_class.pop_total-plot_range
        return a,b,plot_range

        
    def L2s(self,a,b):
        """L2 distance errors between measurements and ukf predictions
        
        finds mean L2 (euclidean) distance at each time step and per each agent
        provides whole array of distances per agent and time
        and L2s per agent and time. 
        
        Parameters
        ------
        a,b: array_like
            `a` measurements and `b` ukf arrays to compare
            
        Returns
        ------
        
        c : array_like
            `c` matrix of  L2 distances between a and b over time and agents.
        index : array_like
            for sampling rates >1 need to omit some empty nan rows.
            this indicates which rows are kept. `index`
            isnt really useful I dont know why its here honestly
        agent_means,time_means : array_like
            column and row means for c respectively. finds the mean error per 
            `agent_means` agent and `time_means` time point
        
        """
        sample_rate =self.filter_class.filter_params["sample_rate"]
        c = np.ones(((a.shape[0]//sample_rate),int(a.shape[1]/2)))*np.nan
        

        index = np.arange(0,c.shape[0])*sample_rate

        #loop over each time per agent
        for i in range(len(index)):
                a2 = np.array([a[sample_rate*i,0::2],a[sample_rate*i,1::2]]).T
                b2 = np.array([b[sample_rate*i,0::2],b[sample_rate*i,1::2]]).T
                res = a2-b2
                c[i,:]=np.apply_along_axis(np.linalg.norm,1,res) # take L2 norm of rows to output vector of scalars
                    
        agent_means = np.nanmean(c,axis=0)
        time_means = np.nanmean(c,axis=1)
        
        return c,index,agent_means,time_means
        
    def diagnostic_plots(self,a,b,observed,save):
        """plots general diagnostics plots for efficacy of UKF
        
        Parameters
        ------
        a,b : array_like
            `a` measurements and `b` ukf arrays
        observed,save : bool
            plotting `observed` or unobserved?
            `save` plots?
        
        """
        if observed:
            obs_text = "Observed"
        else:
            obs_text="Unobserved"
            
        a,b,plot_range = self.plot_data_parser(a,b,observed)
      
        sample_rate =self.filter_class.filter_params["sample_rate"]
        
        #f=plt.figure(figsize=(12,8))
        #for j in range(floor(int(plot_range))):
        #    plt.plot(a[:,(2*j)],a[:,(2*j)+1],lw=3)  
        #    plt.xlim([0,self.filter_class.model_params["width"]])
        #    plt.ylim([0,self.filter_class.model_params["height"]])
        #    plt.xlabel("Corridor Width")
        #    plt.ylabel("Corridor Height")
        #    plt.title(f"{obs_text} True Positions")

        #g = plt.figure(figsize=(12,8))
        #for j in range(int(plot_range)):
        #    plt.plot(b[::sample_rate,2*j],b[::sample_rate,(2*j)+1],lw=3) 
        #    plt.xlim([0,self.filter_class.model_params["width"]])
        #    plt.ylim([0,self.filter_class.model_params["height"]])
        #    plt.xlabel("Corridor Width")
        #    plt.ylabel("Corridor Height")
        #    plt.title(f"{obs_text} KF Predictions")
            
      
        c,c_index,agent_means,time_means = self.L2s(a,b)
        
        #h = plt.figure(figsize=(12,8))
        #time_means[np.isnan(time_means)]=0
        #plt.plot(c_index,time_means,lw=5,color="k",label="Mean Agent L2")
        #for i in range(c.shape[1]):
        #    plt.plot(c_index,c[:,i],linestyle="-.",lw=3)
            
        #plt.axhline(y=0,color="k",ls="--",alpha=0.5)
        #plt.xlabel("Time (steps)")
        #plt.ylabel("L2 Error")
        #plt.title(obs_text+" ")
        #plt.title(f"{obs_text} L2s Over Time")
        #plt.legend()
        #"""find agent with highest L2 and plot it.
        #mainly done to check something odd isnt happening"""
        
        #index = np.where(agent_means == np.nanmax(agent_means))[0][0]
        #print(index)
        #a1 = a[:,(2*index):(2*index)+2]
        #b1 = b[:,(2*index):(2*index)+2]
       # 
       # i = plt.figure(figsize=(12,8))
       # plt.plot(a1[::sample_rate,0],a1[::sample_rate,1],label= "True Path",lw=3)
       # plt.plot(b1[::self.filter_class.sample_rate,0],b1[::self.filter_class.sample_rate,1],label = "KF Prediction",lw=3)
       # plt.legend()
       # plt.xlim([0,self.filter_class.model_params["width"]])
       # plt.ylim([0,self.filter_class.model_params["height"]])
       # plt.xlabel("Corridor Width")
       # plt.ylabel("Corridor Height")
       # plt.title(obs_text+" True Positions")
       # plt.title("Worst agent")
       # plt.title(f"{obs_text} Worst Agent")


        if observed:
                density_number = self.filter_class.sample_size
        else:
            density_number = self.filter_class.model_params["pop_total"]-self.filter_class.sample_size

        j = plt.figure(figsize=(12,8))
        plt.hist(agent_means,density=False,edgecolor="k")
        plt.xlabel("L2 Error")
        plt.ylabel(str(density_number)+ " " + obs_text+ " Agent Counts")
        #plt.title(obs_text+" Histogram of agent L2s")
        #kdeplot(agent_means,color="red",cut=0,lw=4,label="kde estimate")
        #plt.legend()

        if save:
            j.savefig(self.save_dir + f"{obs_text}_agent_hist.pdf")
            #f.savefig(self.save_dir +f"{obs_text}_obs.pdf")
            #g.savefig(self.save_dir +f"{obs_text}_kf.pdf")
            #h.savefig(self.save_dir +f"{obs_text}_l2.pdf")
            # i.savefig(self.save_dir +f"{obs_text}_worst.pdf")
            
        return c,time_means
    
    
        
    def pair_frames(self,a,b):
        """ pairwise animation of ukf predictions and true measurements over ABM run
        
        Parameters
        ------
        a,b : array_like
            `a` measurements and `b` ukf estimates
        
        """
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)
        
        os.mkdir(self.save_dir +"output_pairs")
        for i in range(a.shape[0]):
            a_s = [a_o[i,:],a_u[i,:]]
            b_s = [b_o[i,:], b_u[i,:]]
            f = plt.figure(figsize=(12,8))
            ax = plt.subplot(111)
            plt.xlim([0,width])
            plt.ylim([0,height])
            
            "plot true agents and dummies for legend"
            ax.scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o",edgecolors="k")
            ax.scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o",edgecolors="k")
            ax.scatter(-1,-1,color="orangered",label = "Observed Predictions",marker="P",edgecolors="k")
            ax.scatter(-1,-1,color="yellow",label = "Unobserved Predictions",marker="^",edgecolors="k")

            markers = ["P","^"]
            colours = ["orangered","yellow"]
            for j in range(len(a_s)):

                a1 = a_s[j]
                b1 = b_s[j]
                if np.abs(np.nansum(a1-b1))>1e-4: #check for perfect conditions (initial)
                    for k in range(int(a1.shape[0]/2)):
                        a2 = a1[(2*k):(2*k)+2]
                        b2 = b1[(2*k):(2*k)+2]          
                        if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                            x = [a2[0],b2[0]]
                            y = [a2[1],b2[1]]
                            ax.plot(x,y,color="k")
                            ax.scatter(b2[0],b2[1],color=colours[j],marker = markers[j],edgecolors="k")
            
            "put legend outside of plot"
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=2)
            "labelling"
            plt.xlabel("corridor width")
            plt.ylabel("corridor height")
            #plt.title("True Positions vs UKF Predictions")
            "save frame and close plot else struggle for RAM"
            number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
            file = self.save_dir+ f"output_pairs/pairs{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir +"output_pairs",
                            self.save_dir +f"pairwise_gif_{self.filter_class.pop_total}",24)

    def pair_frames_stack(self,a,b):
        """ pairwise animation with observerd/unobserved error over top
        
        ..deprecated:: 
            not really used this for a while its too small
            a nice gridspec excercise for anyone interested in nested plotting
        
        Parameters
        ------
        a,b : array_like
            `a` measurements and `b` ukf estimates
        
        """        
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)#uobs
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)#obs
        c,c_index,agent_means,time_means = self.L2s(a_o,b_o) #mses
        c2,c_index2,agent_means2,time_means2 = self.L2s(a_u,b_u) #mses
        time_means[np.isnan(time_means)]=0
        time_means2[np.isnan(time_means)]=0
        
        os.mkdir(self.save_dir +"output_pairs")
        for i in range(a.shape[0]):
            a_s = [a_o[i,:],a_u[i,:]]
            b_s = [b_o[i,:], b_u[i,:]]
            f=plt.figure()
            gs = gridspec.GridSpec(4,4)
            axes = [plt.subplot(gs[:2,:]),plt.subplot(gs[2:,:2]),plt.subplot(gs[2:,2:])]

            axes[0].set_xlim([0,width])
            axes[0].set_ylim([0,height])
            axes[1].set_xlim([0,a_u.shape[0]])
            axes[1].set_ylim([0,np.nanmax(time_means)*1.05])
            axes[2].set_xlim([0,a_u.shape[0]])
            axes[2].set_ylim([0,np.nanmax(time_means2)*1.05])         
            
            "plot true agents and dummies for legend"
            axes[0].scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o",edgecolors="k")
            axes[0].scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o",edgecolors="k")
            axes[0].scatter(-1,-1,color="orangered",label = "KF_Observed",marker="P",edgecolors="k")
            axes[0].scatter(-1,-1,color="yellow",label = "KF_Unobserved",marker="^",edgecolors="k")

            
            markers = ["P","^"]
            colours = ["orangered","yellow"]
            for j in range(len(a_s)):

                a1 = a_s[j]
                b1 = b_s[j]
                if np.abs(np.nansum(a1-b1))>1e-4: #check for perfect conditions (initial)
                    for k in range(int(a1.shape[0]/2)):
                        a2 = a1[(2*k):(2*k)+2]
                        b2 = b1[(2*k):(2*k)+2]          
                        if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                            x = [a2[0],b2[0]]
                            y = [a2[1],b2[1]]
                            axes[0].plot(x,y,color="k")
                            axes[0].scatter(b2[0],b2[1],color=colours[j],marker = markers[j],edgecolors="k")
            
            #box = axes[1].get_position()
            #axes[1].set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            
            axes[1].plot(c_index,time_means[:i],label="observed")
            axes[2].plot(c_index2,time_means2[:i],label="unobserved")

            axes[0].legend(bbox_to_anchor=(1.012, 1.2),
                      ncol=3,prop={'size':7})
            axes[0].set_xlabel("corridor width")
            axes[0].set_ylabel("corridor height")
            axes[1].set_ylabel("Observed L2")
            axes[1].set_xlabel("Time (steps)")
            axes[2].set_ylabel("Unobserved L2")
            axes[2].set_xlabel("Time (steps)")
            #axes[0].title("True Positions vs UKF Predictions")
            "zfill names files such that sort() does its job properly later"
            number =  str(i).zfill(ceil(log10(a.shape[0]))) 
            axes[0].text(0,1.05*height,"Frame Number: "+str(i))
            
            file = self.save_dir+f"output_pairs/pairs{number}"
            f.tight_layout()
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir +"output_pairs",
                           self.save_dir +f"pairwise_gif_{self.filter_class.pop_total}",24)
        
    def pair_frames_stack_ellipse(self,a,b):
        """ pairwise animation with observerd/unobserved error and covariance ellipses
        
        ..deprecated:: 
            not really used this for a while its too small
            a nice gridspec excercise for anyone interested in nested plotting
            should probably remove this to drop the filterpy dependency
        
        Parameters
        ------
        a,b : array_like
            `a` measurements and `b` ukf estimates
        
        """    
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        sample_rate=self.filter_class.filter_params["sample_rate"]
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)#obs
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)#uobs
        c,c_index,agent_means,time_means = self.L2s(a_o,b_o) #obs L2s
        c2,c_index2,agent_means2,time_means2 = self.L2s(a_u,b_u) #uobs L2s
        time_means[np.isnan(time_means)]=0
        time_means2[np.isnan(time_means)]=0

        os.mkdir(self.save_dir +"output_pairs")
        for i in range(a.shape[0]):
            a_s = [a_o[i,:],a_u[i,:]]
            b_s = [b_o[i,:], b_u[i,:]]
            f=plt.figure(figsize=(12,12))
            gs = gridspec.GridSpec(4,4)
            axes = [plt.subplot(gs[:2,:]),plt.subplot(gs[2:,:2]),plt.subplot(gs[2:,2:])]
            
             
            
            "plot true agents and dummies for legend"
            
            P = self.filter_class.full_Ps[i]
            agent_covs = []
            for j in range(int(a.shape[1]/2)):
                agent_covs.append(P[(2*j):(2*j)+2,(2*j):(2*j)+2])
            #axes[0].scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o")
            #axes[0].scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o")
            
            "placeholders for a consistent legend. make sure theyre outside the domain of plotting"
            axes[0].scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o",edgecolors="k")
            axes[0].scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o",edgecolors="k")            
            axes[0].scatter(-1,-1,color="orangered",label = "KF_Observed",marker="P",edgecolors="k")
            axes[0].scatter(-1,-1,color="yellow",label = "KF_Unobserved",marker="^",edgecolors="k")

            
            markers = ["P","^"]
            colours = ["orangered","yellow"]
            for j in range(len(a_s)):
                a1 = a_s[j]
                b1 = b_s[j]
                if np.abs(np.nansum(a1-b1))>1e-4: #check for perfect conditions (initial)
                    for k in range(int(a1.shape[0]/2)):
                        a2 = a1[(2*k):(2*k)+2]
                        b2 = b1[(2*k):(2*k)+2]          
                        if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                            x = [a2[0],b2[0]]
                            y = [a2[1],b2[1]]
                            axes[0].plot(x,y,color="k")
                            axes[0].scatter(b2[0],b2[1],color=colours[j],marker = markers[j],edgecolors="k")
                            plot_covariance((x[1],y[1]),agent_covs[k],ax=axes[0],edgecolor="skyblue",alpha=0.6,show_center=False)
            #box = axes[1].get_position()
            #axes[1].set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            
            "labelling"
            axes[2].set_xlim([0,a.shape[0]])
            axes[2].plot(c_index2[:(1+i//sample_rate)],time_means2[:(1+i//sample_rate)],label="unobserved")
            axes[2].set_xlim([0,a.shape[0]])
            axes[2].set_ylim([0,np.nanmax(time_means2)*1.05])  
            axes[2].set_ylabel("Unobserved L2")
            axes[2].set_xlabel("Time (steps)")


            axes[1].set_xlim([0,a_u.shape[0]])
            axes[1].set_ylim([0,np.nanmax(time_means)*1.05])
            axes[1].set_ylabel("Observed L2")
            axes[1].set_xlabel("Time (steps)")
            axes[1].plot(c_index[:(1+i//sample_rate)],time_means[:(1+i//sample_rate)],label="observed")
            

            axes[0].legend(bbox_to_anchor=(1.012, 1.2),ncol=3,prop={'size':14})
            axes[0].set_xlim([0,width])
            axes[0].set_ylim([0,height])
            axes[0].set_xlabel("corridor width")
            axes[0].set_ylabel("corridor height")
            

            #axes[0].title("True Positions vs UKF Predictions")
            "zfill file numbers such that sort() works later"
            number =  str(i).zfill(ceil(log10(a.shape[0]))) 
            axes[0].text(0,1.05*height,"Frame Number: "+str(i))

            file = self.save_dir+f"output_pairs/pairs{number}"
            
            f.tight_layout()
            f.savefig(file,bbox_inches="tight")
            plt.close()
        
        animations.animate(self,self.save_dir +"output_pairs",
                           self.save_dir +f"pairwise_gif_{filter_class.pop_total}",24)
                
    def pair_frames_single(self,a,b,frame_number,save):
        """ pairwise animation with observerd/unobserved error over top
        
        
        Parameters
        ------
        a,b : array_like
            `a` measurements and `b` ukf estimates
        
        frame_number : int
            `frame_number` which time frame to plot
            
        save: bool
            `save` plot?
        """    
        i=frame_number
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)
        
        
        a_s = [a_o[i,:],a_u[i,:]]
        b_s = [b_o[i,:], b_u[i,:]]
        f = plt.figure(figsize=(12,8))
        ax = plt.subplot(111)
        plt.xlim([0,width])
        plt.ylim([0,height])
        
        "plot true agents and dummies for legend"
        ax.scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o",edgecolors="k")
        ax.scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o",edgecolors="k")
        ax.scatter(-1,-1,color="orangered",label = "Observed Predictions",marker="P",edgecolors="k")
        ax.scatter(-1,-1,color="yellow",label = "Unobserved Predictions",marker="^",edgecolors="k")

        markers = ["P","^"]
        colours = ["orangered","yellow"]
        for j in range(len(a_s)):

            a1 = a_s[j]
            b1 = b_s[j]
            if np.abs(np.nansum(a1-b1))>1e-4: #check for perfect conditions (initial)
                for k in range(int(a1.shape[0]/2)):
                    a2 = a1[(2*k):(2*k)+2]
                    b2 = b1[(2*k):(2*k)+2]          
                    if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                        x = [a2[0],b2[0]]
                        y = [a2[1],b2[1]]
                        ax.plot(x,y,color="k")
                        ax.scatter(b2[0],b2[1],color=colours[j],marker = markers[j],edgecolors="k")
        
        "put legend outside of plot"
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  ncol=2)
        "labelling"
        plt.xlabel("corridor width")
        plt.ylabel("corridor height")
        #plt.title("True Positions vs UKF Predictions")
        "save frame and close plot else struggle for RAM"
        number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
        file = self.save_dir+f"ukf_pairs{number}"
        f.savefig(file)


"""should probably remove the following 2 functions if not using covariance ellipses"""
                
def _std_tuple_of(var=None, std=None, interval=None):
    """
    Convienence function for plotting. Given one of var, standard
    deviation, or interval, return the std. Any of the three can be an
    iterable list.
    Examples
    --------
    >>>_std_tuple_of(var=[1, 3, 9])
    (1, 2, 3)
    """

    if std is not None:
        if np.isscalar(std):
            std = (std,)
        return std


    if interval is not None:
        if np.isscalar(interval):
            interval = (interval,)

        return norm.interval(interval)[1]

    if var is None:
        raise ValueError("no inputs were provided")

    if np.isscalar(var):
        var = (var,)
    return np.sqrt(var) 




def plot_covariance(
        mean, cov=None, variance=1.0, std=None, interval=None,
        ellipse=None, title=None, axis_equal=False,
        show_semiaxis=False, show_center=True,
        facecolor=None, edgecolor=None,
        fc='none', ec='#004080',
        alpha=1.0, xlim=None, ylim=None,
        ls='solid',ax=None):
    """
    used to convert covariance matrix into covariance ellipses for each agent.
    shamelessly comandeered from filterpy.stats to remove dependancy
    
    filterpy.stats covariance ellipse plot function with added parameter for custom axis"""
    from matplotlib.patches import Ellipse
    import matplotlib.pyplot as plt

    if cov is not None and ellipse is not None:
        raise ValueError('You cannot specify both cov and ellipse')

    if cov is None and ellipse is None:
        raise ValueError('Specify one of cov or ellipse')

    if facecolor is None:
        facecolor = fc

    if edgecolor is None:
        edgecolor = ec

    if cov is not None:
        ellipse = covariance_ellipse(cov)

    if axis_equal:
        plt.axis('equal')

    if title is not None:
        plt.title(title)

    angle = np.degrees(ellipse[0])
    width = ellipse[1] * 2.
    height = ellipse[2] * 2.

    std = _std_tuple_of(variance, std, interval)
    for sd in std:
        e = Ellipse(xy=mean, width=sd*width, height=sd*height, angle=angle,
                    facecolor=facecolor,
                    edgecolor=edgecolor,
                    alpha=alpha,
                    lw=2, ls=ls)
        ax.add_patch(e)
    x, y = mean
    if show_center:
        ax.scatter(x, y, marker='+', color=edgecolor)

    if show_semiaxis:
        a = ellipse[0]
        h, w = height/4, width/4
        ax.plot([x, x+ h*cos(a+np.pi/2)], [y, y + h*sin(a+np.pi/2)])
        ax.plot([x, x+ w*cos(a)], [y, y + w*sin(a)])
       
        
class animations:
    """class to animate frames recorded by several functions into actual mp4s
    
    Parameters
    ------
    file,name : str
        which `file` of frames to animate
        what `name` to give the gif
    fps : int
        frames per second `fps` of mp4. higher fps means faster video
    """
    def animate(self,file,name,fps):
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}.mp4', images,fps=fps)
        rmtree(file)
        #animations.clear_output_folder(self,file)
        
#%%
if __name__ == "__main__":
    
    recall =True # recalling a pickled run or starting from scratch?
    do_pickle =True # if not recalling do you want to pickle this run so you can recall it?

    if not recall:
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
        
        """run and extract data"""
        n = model_params["pop_total"]
        prop = filter_params["prop"]
        noise = filter_params["noise"]
        
        base_model = Model(**model_params)
        u = ukf_ss(model_params,filter_params,ukf_params,base_model)
        u.main()
        if do_pickle:
            f_name = f"../experiments/ukf_experiments/test_ukf_pickle_{n}_{prop}_{noise}.pkl"
            f = open(f_name,"wb")
            pickle.dump(u,f)
            f.close()
            
            
    else:
        "file name to recall from certain parameters or enter own"

        n = 30
        prop = 0.5 
        noise = 0.5
        
        file_name = f"ukf_pickle_{n}_{prop}_{noise}.pkl"
        f = open("../experiments/ukf_experiments/"+file_name,"rb")
        u = pickle.load(f)
        f.close()
        filter_params = u.filter_params
        model_params = u.model_params
        
    if filter_params["sample_rate"]>1:
        print("partial observations. using interpolated predictions (full_preds) for animations.")
        print("ONLY USE preds FOR ANY ERROR METRICS")
    
    obs,preds,full_preds,truth,nan_array= u.data_parser(True)
    #truth[np.isnan(obs)]=np.nan #keep finished agents from skewing mean down
    preds[~nan_array]=np.nan #kill wierd tails of finished agents (remove this and see what happens)
    full_preds[~nan_array]=np.nan #kill wierd tails of finished agents (remove this and see what happens)
    truth[~nan_array]=np.nan
    """plots"""
    plts = plots(u,"")

    
    plot_save=False
    if filter_params["prop"]<1:
        distances,t_mean = plts.diagnostic_plots(truth,preds,False,plot_save)
    distances2,t_mean2 = plts.diagnostic_plots(truth,preds,True,plot_save)
    
    
    plts.pair_frames_single(truth,full_preds,100,False)

    "animate into pairwise plots gifs?"
    animate = False
    if animate:
        if filter_params["sample_rate"]==1:
            plts.pair_frames(truth,preds)
            #plts.heatmap(real) #these probably dont work anymore. feel free to try
            #plts.pair_frames_stack_ellipse(obs,preds)
    
        else:
            plts.pair_frames(truth,full_preds)
            #plts.pair_frames_stack_ellipse(obs,full_preds) #these probably dont work anymore. feel free to try
            #plts.heatmap(obs)
    
