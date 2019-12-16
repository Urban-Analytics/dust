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

"used to save clss instances when run finished."
import pickle 

"import stationsim model"
"this append seems redundant but the notebooks need it. no idea why."
try:
    from stationsim_model import Model
except:
    sys.path.append("..")
    from stationsim.stationsim_model import Model

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
    
    def __init__(self, model_params, ukf_params, base_model, init_x, fx, hx, p, q, r):
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
        self.base_model = base_model
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



    def unscented_Mean(self, sigma_function, kf_function, *function_args):
        
        
        """calculate unscented transform estimate for forecasted/desired means
        
        -calculate sigma points defined by sigma_function 
            (usually MSSP or central differenceing)
            
        - apply kf_function to sigma points (usually transition function or 
            measurement function.) to get transformed sigma points
        
        - calculate weighted mean of transformed points to get unscented mean
        
        """
        
        sigmas = sigma_function(self.x,self.p)
        "calculate either forecasted sigmas X- or measured sigmas Y with f/h"
        nl_sigmas = np.apply_along_axis(kf_function,0,sigmas,*function_args)
        "calculate unscented mean using non linear sigmas and MSSP mean weights"
        xhat = np.dot(nl_sigmas,self.wm)#unscented mean for predicitons
        
        return sigmas, nl_sigmas, xhat
        
    def covariance(self, data1, mean1, weight, data2 = None,mean2 = None, addition = None):
        
        
        """within/cross-covariance between sigma points and their unscented mean.
        
        Note: CAN'T use numpy.cov here as it uses the regular mean 
            and not the unscented mean. Maybe theres a faster numpy version
        
        Define sigma point matrices X_{mxp}, Y_{nxp}, 
        some unscented mean vectors a_{mx1}, b_{nx1}
        and some vector of covariance weights wc.
        
        Also define a column subtraction operator COL(X,a) 
        such that we subtract a from every column of X elementwise.
        
        Using the above we calculate the cross covariance between two sets of 
        sigma points as 
        
        P_xy = COL(X-a) * W * COL(Y-b)^T
        
        Given some diagonal weight matrix W with diagonal entries wc and 0 otherwise.
        
        This is similar to the standard statistical covariance with the exceptions
        of a non standard mean and weightings.
        """
        
        "if no secondary data defined do within covariance. else do cross"
        
        if data2 is None and mean2 is None:
            data2 = data1
            mean2 = mean1
            
        """calculate component matrices for covariance by performing 
            column subtraction and diagonalising weights."""
        
        weighting = np.diag(weight)
        residuals = (data1.T - mean1).T
        residuals2 = (data2.T - mean2).T
        
        "calculate P_xy as defined above"
    
        covariance_matrix = np.linalg.multi_dot([residuals,weighting,residuals2.T])
        
        """old versions"""
        "old quadratic form version. made faster with multi_dot."
        #covariance_matrix = np.matmul(np.matmul((data1.transpose()-mean1).T,np.diag(weight)),
        #                (data1.transpose()-mean1))+self.q
        
        "numpy quadratic form far faster than this for loop"
        #covariance_matrix =  self.wc[0]*np.outer((data1[:,0].T-mean1),(data2[:,0].T-mean2))+self.Q
        #for i in range(1,len(self.wc)): 
        #    pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)
        
        "if some additive noise is involved (as with the Kalman filter) do it here"
        
        if addition is not None:
            covariance_matrix += addition
        
        return covariance_matrix
    
    def MSSP(self,mean,p):
        
        
        """sigma point calculations based on current mean x and covariance P
        
        Parameters
        ------
        mean , P : array_like
            mean `x` and covariance `P` numpy arrays
            
        Returns
        ------
        sigmas : array_like
            matrix of MSSPs with each column representing one point
        
        """
        P = np.linalg.cholesky(p)
        sigmas = np.ones((self.n,(2*self.n)+1)).T*mean
        sigmas=sigmas.T
        sigmas[:,1:self.n+1] += self.g*P #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*P #'lower' confidence sigmas
        return sigmas 

    def predict(self):
        
        
        """Transitions sigma points forwards using markovian transition function plus noise Q
        
        - calculate sigmas using prior mean and covariance P.
        - forecast sigmas X- for next timestep using transition function Fx.
        - unscented mean for foreacsting next state.
        - calculate interim mean state x and covariance P
        - pass these onto  update function
        
        """
        
        sigmas, nl_sigmas, xhat = self.unscented_Mean(self.MSSP, self.fx,self.base_model)
        pxx = self.covariance(nl_sigmas,xhat,self.wc,addition = self.q)
        
        self.p = pxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z):   
        
        
        """ update forecasts with measurements to get posterior assimilations
        
        - nudges X- sigmas with new pxx from predict
        - calculate measurement sigmas Y = h(X-)
        - calculate unscented mean of Y, yhat
        - calculate measured state covariance pyy sing r
        - calculate cross covariance between X- and Y and Kalman gain (pxy, K)
        - update x and P
        - store x and P updates in lists (xs, ps)
        
        Parameters
        ------
        z : array_like
            measurements from sensors `z`
        """
        
        sigmas, nl_sigmas, yhat = self.unscented_Mean(self.MSSP, self.hx, 
                                                      self.model_params,self.ukf_params)
        pyy =self.covariance(nl_sigmas,yhat, self.wc, addition=self.r)
        pxy = self.covariance(sigmas,self.x, self.wc, nl_sigmas, yhat)
        k = np.matmul(pxy,np.linalg.inv(pyy))
 
        "i dont know why `self.x += ...` doesnt work here"
        self.x = self.x + np.matmul(k,(z-yhat))
        self.p = self.p - np.linalg.multi_dot([k, pyy, k.T])
        
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
        """assimilated ukf positions no filled in section
        (this is predictions vs full predicitons above)"""
        
        self.obs = []  # actual sensor observations
        self.ukf_histories = []  
        self.forecasts=[] 
        self.truths = []  # noiseless observations

        self.full_ps=[]  # full covariances. again used for animations and not error metrics
        self.obs_key = [] # which agents are observed (0 not, 1 agg, 2 gps)
        self.obs_key_func = ukf_params["obs_key_func"]  #defines what type of observation each agent has

        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
    
    def init_ukf(self,ukf_params):
        
        
        """initialise ukf with initial state and covariance structures.
        
        set:
            - initial state
            - noise structures p,q,r
            - ABM base_model, fx/hx functions and their args
        
        Parameters
        ------
        ukf_params : dict
            dictionary of various ukf parameters `ukf_params`
        
        """
        
        x = self.base_model.get_state(sensor="location")#initial state
        p = ukf_params["p"]#inital guess at state covariance
        q = ukf_params["q"]
        r = ukf_params["r"]#sensor noise
        self.ukf = ukf(self.model_params, ukf_params, self.base_model, x, ukf_params["fx"], ukf_params["hx"], p, q, r)
    
    def ss_Predict(self):
        
        
        """ Forecast step of UKF for stationsim.
        
        - forecast state using UKF (unscented transform)
        - update forecasts list
        - jump base_model forwards to forecast time
        - 
        """
        "forecast sigma points forwards to predict next state"
        self.ukf.predict() 
        self.forecasts.append(self.ukf.x)
        "step model forwards"
        self.base_model.step()
        "add true noiseless values from ABM for comparison"
        self.truths.append(self.base_model.get_state(sensor="location"))
        "append raw ukf forecasts of x and p"

    def ss_Update(self,step):
        
        
        """ Update step of UKF for stationsim.
        
        - measure state from base_model.
        - add some gaussian noise.
        - assimilate ukf with noise state
        - record each agents observation type.
        - append lists of ukf assimilations and observations
        
        """
        if step%self.sample_rate == 0:
            state = self.base_model.get_state(sensor="location")
            "apply noise to active agents"
            if self.ukf_params["bring_noise"]:
                noise_array=np.ones(self.pop_total*2)
                noise_array[np.repeat([agent.status!=1 for agent in self.base_model.agents],2)]=0
                noise_array*=np.random.normal(0,self.ukf_params["noise"],self.pop_total*2)
                state+=noise_array
                
            "convert full noisy state to actual sensor observations"
            state = self.ukf.hx(state, self.model_params, self.ukf_params)
                
            self.ukf.update(state)
            
            if self.obs_key_func is not None:
                key = self.obs_key_func(state,self.model_params, self.ukf_params)
                "force inactive agents to unobserved"
                key *= [agent.status%2 for agent in self.base_model.agents]
                self.obs_key.append(key)

            self.ukf_histories.append(self.ukf.x) #append histories
            self.obs.append(state)
                 
        
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
        
        "initialise UKF"
        self.init_ukf(self.ukf_params) 
        for step in range(self.number_of_iterations-1):
            
            "forecast next StationSim state and jump model forwards"
            self.ss_Predict()
            "assimilate forecasts using new model state."
            self.ss_Update(step)
            
            finished = self.base_model.pop_finished == self.pop_total
            if finished: #break condition
                break
            
            #elif np.nansum(np.isnan(self.ukf.x)) == 0:
            #    print("math error. try larger values of alpha else check fx and hx.")
            #    break
          

        self.time2 = datetime.datetime.now()#timer
        print(self.time2-self.time1)

    def data_parser(self):
        
        
        """extracts data into numpy arrays
        
        Returns
        ------
            
        obs : array_like
            `obs` noisy observations of agents positions
        preds : array_like
            `preds` ukf predictions of said agent positions
        forecasts : array_like
            `forecasts` just the first ukf step at every time point
            useful for comparison in experiment 0
        truths : 
            `truths` true noiseless agent positions for post-hoc comparison
            
        nan_array : array_like
            `nan_array` stationsim gets stuck when an agent finishes it's run. good for plotting/metrics            
        """
       
       
            
        """pull actual data. note a and b dont have gaps every sample_rate
        measurements. Need to fill in based on truths (d).
        """
        obs =  np.vstack(self.obs) 
        preds2 = np.vstack(self.ukf_histories)
        truths = np.vstack(self.truths)
        
        
        "full 'd' size placeholders"
        preds= np.zeros((truths.shape[0],self.pop_total*2))*np.nan
        
        "fill in every sample_rate rows with ukf estimates and observation type key"
        for j in range(int(preds.shape[0]//self.sample_rate)):
            preds[j*self.sample_rate,:] = preds2[j,:]

        nan_array = np.ones(shape = truths.shape)*np.nan
        for i, agent in enumerate(self.base_model.agents):
            "find which rows are  NOT (None, None). Store in index. "
            array = np.array(agent.history_locations)
            index = ~np.equal(array,None)[:,0]
            "set anything in index to 1. I.E which agents are still in model."
            nan_array[index,2*i:(2*i)+2] = 1

        return obs,preds,truths,nan_array
          
       
    def obs_key_parser(self):
        """extract obs_key
        
        """
        obs_key2 = np.vstack(self.obs_key)
        shape = np.vstack(self.truths).shape[0]
        obs_key = np.zeros((shape,self.pop_total))*np.nan
        
        for j in range(int(shape//self.sample_rate)):
            obs_key[j*self.sample_rate,:] = obs_key2[j,:]
        
        return obs_key
        
def pickler(instance, source, f_name):
    "save ukf run as a pickle"
    f = open(source + f_name,"wb")
    pickle.dump(instance,f)
    f.close()

def depickler(source, f_name):
    "load a ukf pickle"
    f = open(source+f_name,"rb")
    u = pickle.load(f)
    f.close()
    return u





