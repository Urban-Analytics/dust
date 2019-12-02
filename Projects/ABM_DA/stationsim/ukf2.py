# -*- coding: utf-8 -*-

"""
Created on Thu May 23 11:13:26 2019
@author: RC

The Unscented Kalman Filter (UKF) designed to be hyper efficient alternative to similar 
Monte Carlo techniques such as the Particle Filter. This file aims to unify the old 
intern project files by combining them into one single filter. It also aims to be geared 
towards real data with a modular data input approach for a hybrid of sensors.

This is based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf

NOTE: To avoid confusion 'observation/obs' are observed stationsim data
not to be confused with the 'observed' boolean 
determining whether to look at observed/unobserved agent subset
(maybe change the former the measurements etc.)

NOTE: __main__ here is now deprecated. use ukf notebook in experiments folder

As of 01/09/19 dependencies are: 
    pip install imageio
    pip install imageio-ffmpeg
    pip install ffmpeg
    pip install scipy
    pip install filterpy
"""

#general packages used for filtering
"used for a lot of things"
import os
import sys 
import numpy as np

path = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.append(path)

"general timer"
import datetime 

"used to calculate sigma points in parallel."
import multiprocessing  

"used to save clss instances when run finished."
import pickle 

"import stationsim model"
sys.path.append("..")
from stationsim.stationsim_model import Model




sys.path.append("../experiments/ukf_experiments/")

from ukf_ex1 import omission_params
from ukf_ex2 import aggregate_params 
from ukf_plots import ukf_plots


"used for plotting covariance ellipses for each agent. not really used anymore"
# from filterpy.stats import covariance_ellipse  
# from scipy.stats import norm #easy l2 norming  
# from math import cos, sin

#%%

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
    
    def __init__(self, model_params, ukf_params, init_x, fx, hx, p, q, r):
        """
        x - state
        n - state size 
        p - state covariance
        fx - transition function
        hx - measurement function
        lam - lambda paramter function of tuning parameters a,b,k
        g - gamma parameter function of tuning parameters a,b,k
        wm/wc - unscented weights for mean and covariances respectively.
        q,r -noise structures for fx and hx
        xs,ps - lists for storage
        """
        
        #init initial state
        self.model_params = model_params
        self.ukf_params = ukf_params
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.p = p
        #self.P = np.linalg.cholesky(self.x)
        self.fx = fx
        self.hx = hx
        
        #init further parameters based on a through el
        self.lam = ukf_params["a"]**2*(self.n+ukf_params["k"]) - self.n #lambda paramter calculated viar
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        #init weights based on paramters a through el
        main_weight =  1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-ukf_params["a"]**2+ukf_params["b"])
    
        self.q = q
        self.r = r

        self.xs = []
        self.ps = []

    def Sigmas(self,mean,p):
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
        sigmas[:,1:self.n+1] += self.g*p #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*p #'lower' confidence sigmas
        return sigmas 

    def predict(self):
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
        
        "calculate MSSPs using x and P"
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.p)) #calculate current sigmas using state x and UT element S
        "numpy apply along axis or multiprocessing options"
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas,base_model)
        "old multiprocessing version. maybe better in higher dimensional cases"
        # p = multiprocessing.Pool()
        # nl_sigmas = np.vstack(p.map(self.fx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        # p.close()
        "weight sigmas"
        wnl_sigmas = nl_sigmas*self.wm
        "unscented desired state forecast"
        xhat = np.sum(wnl_sigmas,axis=1)#unscented mean for predicitons
        
        """
        should be a faster way of doing this
        covariance estimation for prior P as a sum of the outer products of 
        (sigmas - unscented mean) weighted by wc
        """
        
        " quadratic form calculation of (cross/within) variances to replace old for loop"
        pxx = np.matmul(np.matmul((nl_sigmas.transpose()-xhat).T,np.diag(self.wc)),
                        (nl_sigmas.transpose()-xhat))+self.q
        
        "old version. numpy probably faster than this for loop"
        #pxx =  self.wc[0]*np.outer((nl_sigmas[:,0].T-xhat),(nl_sigmas[:,0].T-xhat))+self.Q
        #for i in range(1,len(self.wc)): 
        #    pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)
            
        "update p and x with forecasts to assimilate in update"
        self.p = pxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z):     
        """ update forecasts with measurements to get posterior assimilations
        
        Does numerous things in the following order
        - nudges X- sigmas with calculated Px from predict
        - calculate measurement sigmas Y = h(X)
        - calculate unscented means of Y
        -calculate py pxy,K using r
        - calculate x update
        - calculate p update
        
        Parameters
        ------
        z : array_like
            measurements from sensors `z`
        **hx_args
            arbitrary kwargs for measurement function
        """
        
        "`nudge` sigmas based on now observed x and p"
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.p)) 
        
        "calculate measured state sigmas from desired sigmas using hx"
        "apply along axis version seems better than multiprocessing"
        nl_sigmas = np.apply_along_axis(self.hx,0,sigmas,self.model_params,self.ukf_params)
        
        "old multiprocessing version. might be faster for higher dim cases"
        # p = multiprocessing.Pool()
        # nl_sigmas = np.vstack(p.map(self.hx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        # p.close()
        
        "apply unscented weighting"
        wnl_sigmas = nl_sigmas*self.wm

        "measured state unscented mean"
        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        
        "unscented estimate for measured state covariance"
        "now using quadratic form"
        pyy = np.matmul(np.matmul((nl_sigmas.transpose()-yhat).T,np.diag(self.wc))
            ,(nl_sigmas.transpose()-yhat))+self.r

        "old version of pyy. no idea if its faster but numpy > this for loop probably"
        # pyy =  self.wc[0]*np.outer((nl_sigmas[:,0].transpose()-yhat),
        #     (nl_sigmas[:,0].transpose()-yhat))+self.R
        # for i in range(1,len(self.wc)):
        #     pyy += self.wc[i]*np.outer((nl_sigmas[:,i].transpose()-yhat),
        #       (nl_sigmas[:,i].transpose()-yhat))
        
        
        "quadratic form calculation for desired/measured cross covariance."
        pxy = np.matmul(np.matmul((sigmas.transpose()-self.x).T,np.diag(self.wc))
            ,(nl_sigmas.transpose()-yhat))

        "old version of pxy. no idea if its faster but numpy > this for loop probably"
        # pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),
        #    (nl_sigmas[:,0].transpose()-yhat))
        # for i in range(1,len(self.wc)):
        #     pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),
        #       (nl_sigmas[:,i].transpose()-yhat))
            
        "kalman gain. ratio of trust between forecast and measurements"
        k = np.matmul(pxy,np.linalg.inv(pyy))
 
        "update x and p. add to overall trajectories"
        self.x = self.x + np.matmul(k,(self.hx(z, self.model_params, self.ukf_params)-yhat))
        self.p = self.p - np.matmul(k,np.matmul(pyy,k.T))
        self.ps.append(self.p)
        self.xs.append(self.x)

    def batch(self):
        """
        batch function hopefully coming soon
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
    def __init__(self,model_params,ukf_params,base_model):
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
        self.ukf_params = ukf_params # ukf parameters
        self.base_model = base_model #station sim
        
        
        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        # number of batch iterations
        self.number_of_iterations = model_params['step_limit']
        self.sample_rate = self.ukf_params["sample_rate"]
        # how many agents being observed

            
        #random sample of agents to be observed

        
        
        """fills in blanks between assimlations with pure stationsim. 
        good for animation. not good for error metrics use ukf_histories
        """
        self.ukf_preds=[] 
        """assimilated ukf positions no filled in section
        (this is predictions vs full predicitons above)"""
        self.ukf_histories = []  
        self.full_ps=[]  # full covariances. again used for animations and not error metrics
        self.truths = []  # noiseless observations
        self.obs = []  # actual sensor observations
        self.obs_key = [] # which agents are observed (0 not, 1 agg, 2 gps)
        self.obs_key_func = ukf_params["obs_key_func"]  #

        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
    
    def init_ukf(self,ukf_params):
        """initialise ukf with initial state and covariance structures.
        
        
        Parameters
        ------
        ukf_params : dict
            dictionary of various ukf parameters `ukf_params`
        
        """
        
        x = self.base_model.get_state(sensor="location")#initial state
        p = ukf_params["p"]#inital guess at state covariance
        q = ukf_params["q"]
        r = ukf_params["r"]#sensor noise
        self.ukf = ukf(model_params, ukf_params,x,ukf_params["fx"],ukf_params["hx"],p,q,r)
    
    
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
            
            "forecast sigma points forwards to predict next state"
            self.ukf.predict() 
            "step model forwards"
            self.base_model.step()
            "add true noiseless values from ABM for comparison"
            self.truths.append(self.base_model.get_state(sensor="location"))
            
            "DA update step and data logging"
            "data logged for full preds and only assimilated preds (just predict step or predict and update)"
            if _%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.get_state(sensor="location") #observed agents states
                "apply noise to active agents"
                if self.ukf_params["bring_noise"]:
                    noise_array=np.ones(self.pop_total*2)
                    noise_array[np.repeat([agent.status!=1 for agent in self.base_model.agents],2)]=0
                    noise_array*=np.random.normal(0,self.ukf_params["noise"],self.pop_total*2)
                    state+=noise_array
                    
                    
                self.ukf.update(z=state) #update UKF
                key = self.obs_key_func(state,self.model_params, self.ukf_params)
                "force inactive agents to unobserved"
                key *= [agent.status%2 for agent in self.base_model.agents]
                self.obs_key.append(key)

                self.ukf_histories.append(self.ukf.x) #append histories
                self.ukf_preds.append(self.ukf.x)
                
                self.obs.append(self.ukf.hx(state, self.model_params, self.ukf_params))
                self.full_ps.append(self.ukf.p)
                     
                
                x = self.ukf.x
                if np.sum(np.isnan(x))==x.shape[0]:
                    print("math error. try larger values of alpha else check fx and hx.")
                    break
            else:
                "update full preds that arent assimilated"
                self.ukf_preds.append(self.ukf.x)
                self.full_ps.append(self.ukf.p)
                ""
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        self.time2 = datetime.datetime.now()#timer
        print(self.time2-self.time1)

    def data_parser(self):
        
        
        """extracts data into numpy arrays
        
        Returns
        ------
            
        a : array_like
            `a` noisy observations of agents positions
        b : array_like
            `b` ukf predictions of said agent positions
        c : array_like
            `c` if sampling rate >1 fills blank space in b with pure stationsim prediciton
                this is solely for smoother animations later
        d : 
            `d` true noiseless agent positions for post-hoc comparison
            
        nan_array : array_like
            `nan_array` which entries of b are nan. good for plotting/metrics            
        """
       
        nan_array = np.ones(shape=(max([len(agent.history_locations) for agent in self.base_model.agents]),2*self.pop_total))*np.nan
        for i in range(self.pop_total):
            agent = self.base_model.agents[i]
            array = np.array(agent.history_locations)
            array[array==None] ==np.nan
            nan_array[:len(agent.history_locations),2*i:(2*i)+2] = array
        
        nan_array = ~np.isnan(nan_array)
        
        """pull actual data. note a and b dont have gaps every sample_rate
        measurements. Need to fill in based on truths (d).
        """
        a =  np.vstack(self.obs) 
        b2 = np.vstack(self.ukf_histories)
        d = np.vstack(self.truths)
        obs_key2 = np.vstack(self.obs_key)
        
        "full 'd' size placeholders"
        b= np.zeros((d.shape[0],self.pop_total*2))*np.nan
        obs_key = np.zeros((d.shape[0],self.pop_total))*np.nan
        
        "fill in every sample_rate rows with ukf estimates and observation type key"
        for j in range(int(b.shape[0]//self.sample_rate)):
            b[j*self.sample_rate,:] = b2[j,:]
            obs_key[j*self.sample_rate,:] = obs_key2[j,:]

        """only need c if there are gaps in measurements >1 sample_rate
        else ignore
        """
        if self.sample_rate>1:
            c= np.vstack(self.ukf_preds)
            return a,b,c,d,obs_key,nan_array
        else:
            return a,b,d,obs_key,nan_array

#%%

if __name__ == "__main__":
    recall = False #recall previous run
    do_pickle = False #pickle new run
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
        bin_size - square sizes for aggregate plots,
        do_batch - do batch processing on some pre-recorded truth data.
        bring_noise - add noise to measurements?
        noise - variance of added noise
        
        a - alpha between 1 and 1e-4 typically determines spread of sigma points.
            however for large dimensions may need to be even higher
        b - beta set to 2 for gaussian. determines trust in prior distribution.
        k - kappa usually 0 for state estimation and 3-dim(state) for parameters.
            not 100% sure what kappa does. think its a bias parameter.
        !! might be worth making an interactive notebook that varies these. for fun
        """
        
        ukf_params = {      
               
                'sample_rate' : 5,
                "do_restrict" : True, 
                "bring_noise" : True,
                "noise" : 0.5,
                "do_batch" : False,
        
                "a": 1,
                "b": 2,
                "k": 0,
                
                }
        
        "unhash the necessary one"
        "omission version"
        #ukf_params = omission_params(model_params, ukf_params, 0.5)
        "aggregate version"
        ukf_params = aggregate_params(model_params, ukf_params,50)
        "lateral omission"
        #ukf_params = lateral_params(model_params, ukf_params)
        print(model_params)
        print(ukf_params)
        base_model = Model(**model_params)
        u = ukf_ss(model_params,ukf_params,base_model)
        u.main()
        
        obs,preds,full_preds,truth,obs_key,nan_array= u.data_parser()
        truth[~nan_array]=np.nan
        preds[~nan_array]=np.nan
        full_preds[~nan_array]=np.nan

        plts = ukf_plots(u,"../stationsim/")
        
        "animations"
        #plts.trajectories(truth)
        #if ukf_params["sample_rate"]>1:
        #    plts.pair_frames_animation(truth,full_preds,range(truth.shape[0]))
        #else:
        #    plts.pair_frames_animation(truth,preds)
            
        #plts.heatmap(truth,ukf_params,truth.shape[0])
        
        "single frame plots"
        
        plts.pair_frame(truth, preds, obs_key, 50)
        plts.heatmap_frame(truth,ukf_params,50)
        plts.error_hist(truth, preds, False)
        plts.path_plots(truth,preds, False)
    else:
        "file name to recall from certain parameters or enter own"
        n = 30
        noise= 0.5 
        bin_size = 25


        file_name = f"ukf_agg_pickle_{n}_{bin_size}_{noise}.pkl"
        f = open("../experiments/ukf_experiments/"+file_name,"rb")
        u = pickle.load(f)
        f.close()
        filter_params=u.filter_params
    
        poly_list =u.poly_list #generic square grid over corridor



