# -*- coding: utf-8 -*-

"""
The Unscented Kalman Filter (UKF) designed to be hyper efficient alternative to similar 
Monte Carlo techniques such as the Particle Filter. This file aims to unify the old 
intern project files by combining them into one single filter. It also aims to be geared 
towards real data with a modular data input approach for a hybrid of sensors.

This is based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf

This file has no main. use ukf notebook/modules in experiments folder

"""

#general packages used for filtering

import os
import numpy as np #numpy
import datetime # for timing experiments
import pickle # for saving class instances
from scipy.stats import chi2 # for adaptive ukf test
import glob

def unscented_Mean(sigmas, wm, kf_function, **function_kwargs):
    
    
    """calculate unscented transform estimate for forecasted/desired means
    
    -calculate sigma points using sigma_function 
        (e.g Merwe Scaled Sigma Points (MSSP) or 
        central difference sigma points (CDSP)) 
    - apply kf_function to sigma points (usually transition function or 
        measurement function.) to get an array of transformed sigma points.
    - calculate weighted mean of transformed points to get unscented mean.
    
    Parameters
    ------
    sigmas, wm : array_like
        `sigmas` array of sigma points  and `wm` mean weights
        
    kf_function : function
        `sigma_function` function defining type of sigmas used and
        `kf_function` defining whether to apply transition of measurement 
        function (f/h) of UKF.
    
    **function_kwargs : kwargs
        `function_kwargs` keyword arguments for kf_function. Varies depending
        on the experiment so good to be general here. This must be in dictionary 
        form e.g. {"test":1} then called as necessary in kf_function as 
        test = function_kwargs["test"] = 1
        
        This allows for a generalised set of arguements for any desired
        function. Typically we use Kalman filter measurement and transition functions.
        Any function could be used as long as it takes some vector of 1d agent inputs and
        returns a 1d output.
        
    Returns 
    ------
    
    sigmas, nl_sigmas, xhat : array_like
        raw `sigmas` from sigma_function, projected non-linear `nl_sigmas`, 
        and the unscented mean of `xhat` of said projections.
        
    """
    
    "calculate either forecasted sigmas X- or measured sigmas Y with f/h"
    nl_sigmas = np.apply_along_axis(kf_function,0,sigmas, **function_kwargs)
    "calculate unscented mean using non linear sigmas and MSSP mean weights"
    xhat = np.dot(nl_sigmas, wm)#unscented mean for predicitons
    
    return nl_sigmas, xhat

    
def covariance(data1, mean1, weight, data2 = None, mean2 = None, addition = None):
    
    
    """within/cross-covariance between sigma points and their unscented mean.
    
    Note: CAN'T use numpy.cov here as it uses the regular mean 
    and not the unscented mean. as far as i know there isnt a
    numpy function for this
    
    This is the mathematical formula. Feel free to ignore
    --------------------------------------
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
    --------------------------------------
    
    - put weights in diagonal matrix
    - calculate resdiual matrix/ices as each set of sigmas minus their mean
    - calculate weighted covariance as per formula above
    - add additional noise if required e.g. process/sensor noise
    
    Parameters
    ------
    
    data1, mean1` : array_like
        `data1` some array of sigma points and their unscented mean `mean1` 
        
    data2, mean2` : array_like
        `data2` some OTHER array of sigma points and their unscented mean `mean2` 
        can be same as data1, mean1 for within sample covariance.
        
    `weight` : array_like
        `weight` sample covariance weightings
    
    addition : array_like
        `addition` some additive noise for the covariance such as 
        the sensor/process noise.
        
    Returns 
    ------
    
    covariance_matrix : array_like
        `covariance_matrix` unscented covariance matrix used in ukf algorithm
        
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

def MSSP(mean,p,g):
    
    """sigma point calculations based on current mean x and covariance P
    
    - calculate square root of P 
    - generate empty sigma frame with each column as mean
    - keep first point the same
    - for next n points add the ith column of sqrt(P)
    - for the final n points subtract the ith column of sqrt(P)
    
    Parameters
    ------
    mean , p : array_like
        state mean `x` and covariance `p` numpy arrays
        
    g : float
        `g` sigma point scaling parameter. larger g pushes outer sigma points
        further from the centre sigma.
        
    Returns
    ------
    sigmas : array_like
        matrix of MSSPs with each column representing one sigma point
    
    """
    n = mean.shape[0]
    s = np.linalg.cholesky(p)
    sigmas = np.ones((n,(2*n)+1)).T*mean
    sigmas=sigmas.T
    sigmas[:,1:n+1] += g*s #'upper' confidence sigmas
    sigmas[:,n+1:] -= g*s #'lower' confidence sigmas
    return sigmas 

#%%

class ukf:
    
    """main ukf class for assimilating sequential data from some ABM
    
    Parameters
    ------
    model_params, ukf_params : dict
        dictionary of model `model_params` and ukf `ukf_params` parameters
    
    base_model : cls
        `base_model` initalised class instance of desired ABM.
    """
    
    def __init__(self, model_params, ukf_params, base_model):
        
        
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
        "full parameter dictionaries and ABM"
        self.model_params = model_params
        self.ukf_params = ukf_params
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])
            
        self.base_model = base_model
        
        "pull parameters from dictionary"
        self.x = self.base_model.get_state(sensor="location") #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension


        "MSSP sigma point scaling parameters"
        self.lam = self.a**2*(self.n+self.k) - self.n 
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        "unscented mean and covariance weights based on a, b, and k"
        main_weight =  1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-self.a**2+self.b)

        self.xs = []
        self.ps = []
    
    def predict(self, **fx_kwargs):
        
        
        """Transitions sigma points forwards using markovian transition function plus noise Q
        
        - calculate sigmas using prior mean and covariance P.
        - forecast sigmas X- for next timestep using transition function Fx.
        - unscented mean for foreacsting next state.
        - calculate interim mean state x and covariance P
        - pass these onto  update function
        
        Parameters
        ------
        fx_kwargs : dict
            keyword arguments for transition function of ABM.
            This is step for stationsim and requires the current stationsim
            instance as an arguement to run forwards as an ensemble.
        """
        
        sigmas = MSSP(self.x, self.p, self.g)
        nl_sigmas, xhat = unscented_Mean(sigmas, self.wm, self.fx,
                                         **self.fx_kwargs )
        self.sigmas = nl_sigmas
        
        pxx = covariance(nl_sigmas,xhat,self.wc,addition = self.q)
        
        self.p = pxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z, **hx_kwargs):   
        
        
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
        hx_kwargs : dict
        
        """
        
        nl_sigmas, yhat = unscented_Mean(self.sigmas, self.wm,
                                         self.hx, **self.hx_kwargs)
        pyy =covariance(nl_sigmas, yhat, self.wc, addition=self.r)
        pxy = covariance(self.sigmas, self.x, self.wc, nl_sigmas, yhat)
        k = np.matmul(pxy,np.linalg.inv(pyy))
 
        "i dont know why `self.x += ...` doesnt work here"
        x = self.x + np.matmul(k,(z-yhat))
        p = self.p - np.linalg.multi_dot([k, pyy, k.T])
        
        
        """adaptive ukf augmentation. one for later."""
        adaptive = False
        if adaptive:
            mu = np.array(z)- np.array(self.hx(self.sigmas[:,0],
                                               self.model_params,self.ukf_params))
            if np.sum(np.abs(mu))!=0:
                x, p = self.fault_test(z, mu, pxy, pyy, self.x, self.p, k, yhat)    
            
        "update mean and covariance states"
        self.ps.append(self.p)
        self.xs.append(self.x)

    def batch(self):
        """
        batch function hopefully coming soon
        """
        return

    def fault_test(self,z, mu, pxy, pyy, x, p, k, yhat):
        
        
        """ adaptive UKF augmentation
        
        -check chi squared test
        -if fails update q and r.
        -recalculate new x and p
        
        """
        sigma = np.linalg.inv((pyy+ self.r))
        psi = np.linalg.multi_dot([mu.T, sigma, mu])
        critical = chi2.ppf(0.8, df = mu.shape[0]) #critical rejection point
        print(psi, critical)
        if psi <= critical :
            
            "accept null hypothesis. keep q,r"
            pass
        
        else:
            
            "nudge q and r according to estimates. recalculate x and p."
            eps = z - self.hx(x, **self.hx_kwargs)            
            sigmas = MSSP(x,p,self.g)
            syy = covariance(self.hx(sigmas, **self.hx_kwargs), yhat,self.wc)
            delta_1 =  1 - (self.a*critical)/psi
            delta = np.max(self.delta0,delta_1)
            phi_1 =  1 - (self.b*critical)/psi
            phi = np.max(self.phi0, phi_1)
            
            self.q = (1-phi)*self.q + phi*np.linalg.multi_dot([k, mu, mu.T, k.T])
            self.r = (1-delta)*self.r + delta*(np.linalg.multi_dot([eps, eps.T]) + syy)
            
            print("noises updated")
            "correct estimates using new noise"
            pyy  = syy + self.r
            k = np.matmul(pxy,np.linalg.inv(pyy))
            x = self.x + np.matmul(k,(z-yhat))
            p = self.p - np.linalg.multi_dot([k, pyy, k.T])
            
        return x, p
        
     
    
#%%
class ukf_ss:
    
    
    """UKF for station sim using ukf filter class.
    
    Parameters
    ------
    model_params,filter_params,ukf_params : dict
        loads in parameters for the model, station sim filter and general UKF parameters
        `model_params`,`filter_params`,`ukf_params`

    base_model : method
        stationsim model `base_model`
    """
    
    def __init__(self, model_params, ukf_params, base_model):
        
        
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
        
        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])
        
       

        """lists for various data outputs
        observations
        ukf assimilations
        pure stationsim forecasts
        ground truths
        list of covariance matrices
        list of observation types for each agents at one time point
        """
        self.obs = []  # actual sensor observations
        self.ukf_histories = []  
        self.forecasts=[] 
        self.truths = []  # noiseless observations

        self.full_ps=[]  # full covariances. again used for animations and not error metrics
        self.obs_key = [] # which agents are observed (0 not, 1 agg, 2 gps)

        "timer"
        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
    
    @classmethod
    def set_random_seed(cls, seed=None):
        """Set a new numpy random seed
        :param seed: the optional seed value (if None then get one from os.urandom)
        """
        new_seed = int.from_bytes(os.urandom(4), byteorder='little') if seed == None else seed
        np.random.seed(new_seed)
        
    def init_ukf(self,ukf_params,seed = None):
        
        
        """initialise ukf with initial state and covariance structures.
       
        Parameters
        ------
        ukf_params : dict
            dictionary of various ukf parameters `ukf_params`
        seed : float
            fixed `seed` for testing. 
        
        Returns
        ------
        self.ukf : class
            `ukf` class intance for stationsim
        """
        if seed != None:
            self.set_random_seed(seed)
        
        self.ukf = ukf(self.model_params, ukf_params, self.base_model)
    
    def ss_Predict(self):
        
        
        """ Forecast step of UKF for stationsim.
        
        - forecast state using UKF (unscented transform)
        - update forecasts list
        - jump base_model forwards to forecast time
        - update truths list with new positions
        """
        
        self.ukf.predict() 
        self.forecasts.append(self.ukf.x)
        self.base_model.step()
        self.truths.append(self.base_model.get_state(sensor="location"))

    def ss_Update(self,step,**hx_kwargs):
        
        
        """ Update step of UKF for stationsim.
        - if step is a multiple of sample_rate
            - measure state from base_model.
            - add some gaussian noise to active agents.
            - apply measurement funciton h to project noisy 
                state onto measured state
            - assimilate ukf with projected noisy state
            - calculate each agents observation type with obs_key_func.
            - append lists of ukf assimilations and model observations
        - else do nothing
        """
        
        if step%self.sample_rate == 0:
            
            state = self.base_model.get_state(sensor="location")
            noise_array=np.ones(self.pop_total*2)
            noise_array[np.repeat([agent.status!=1 for 
                                   agent in self.base_model.agents], 2)]=0
            noise_array*=np.random.normal(0, self.noise, self.pop_total*2)
            state+=noise_array
                
            "convert full noisy state to actual sensor observations"
            state = self.ukf.hx(state, **hx_kwargs)
                
            self.ukf.update(state, **hx_kwargs)
            
            if self.obs_key_func is not None:
                key = self.obs_key_func(state,**self.obs_key_kwargs)
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
        for step in range(self.step_limit-1):
            
            "forecast next StationSim state and jump model forwards"
            self.ss_Predict()
            "assimilate forecasts using new model state."
            self.ss_Update(step, **self.hx_kwargs)
            
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
        obs_key = np.zeros((shape, self.pop_total))*np.nan
        
        for j in range(int(shape//self.sample_rate)):
            obs_key[j*self.sample_rate, :] = obs_key2[j, :]
        
        return obs_key
        
def pickler(instance, pickle_source, f_name):
    
    
    """save ukf run as a pickle
    
    Parameters
    ------
    instance : class
        finished ukf_ss class `instance` to pickle. defaults to None 
        such that if no run is available we load a pickle instead.
        
    f_name, pickle_source : str
        `f_name` name of pickle file and `pickle_source` where to load 
        and save pickles from/to

    """
    
    f = open(pickle_source + f_name,"wb")
    pickle.dump(instance,f)
    f.close()

def depickler(pickle_source, f_name):
    
    
    """load a ukf pickle
    
    Parameters
    ------
    pickle_source : str
        `pickle_source` where to load and save pickles from/to

    instance : class
        finished ukf_ss class `instance` to pickle. defaults to None 
        such that if no run is available we load a pickle instead.
    """
    f = open(pickle_source+f_name,"rb")
    u = pickle.load(f)
    f.close()
    return u

    

class class_dict_to_instance(ukf_ss):
    
    
    """ build a complete ukf_ss instance from a pickled class_dict.
    
    This class simply inherits the ukf_ss class and adds attributes according
    to some dictionary 
    """
    
    def __init__(self, dictionary):
        
        
        """ take base ukf_ss class and load in attributes for a finished
        ABM run defined by dictionary
        """
        
        for key in dictionary.keys():
            "for items in dictionary set self.key = dictionary[key]"
            setattr(self, key, dictionary[key])
        
def pickle_main(f_name, pickle_source, do_pickle, instance = None):
    
    
    """main function for saving and loading ukf pickles
    
    NOTE THE FOLLOWING IS DEPRECATED IT NOW SAVES AS CLASS_DICT INSTEAD FOR 
    VARIOUS REASONS
    
    - check if we have a finished ukf_ss class and do we want to pickle it
    - if so, pickle it as f_name at pickle_source
    - else, if no ukf_ss class is present, load one with f_name from pickle_source 
        
    IT IS NOW
    
    - check if we have a finished ukf_ss class instance and do we want to pickle it
    - if so, pickle instance.__dict__ as f_name at pickle_source
    - if no ukf_ss class is present, load one with f_name from pickle_source 
    - if the file is a dictionary open it into a class instance for the plots to understand
    - if it is an instance just load it as is.
    
           
    Parameters
    ------
    f_name, pickle_source : str
        `f_name` name of pickle file and `pickle_source` where to load 
        and save pickles from/to
    
    do_pickle : bool
        `do_pickle` do we want to pickle a finished run?
   
    instance : class
        finished ukf_ss class `instance` to pickle. defaults to None 
        such that if no run is available we load a pickle instead.
    """
    
    if do_pickle and instance is not None:
        
        "if given an instance. save it as a class dictionary pickle"
        print(f"Pickling file to dict_{f_name}")
        pickler(instance.__dict__, pickle_source, f_name)
        return
    
    else:
        file = depickler(pickle_source, f_name)
        print(f"Loading pickle {f_name}")
        "try loading the specified file as a class dict. else an instance."
        if type(file) == dict:
            "removes old ukf function in memory"
            try:
                """for converting old files. removes deprecated function that
                plays havoc with pickle"""
                file.pop("ukf")
            except:
                pass
            instance =  class_dict_to_instance(file)
        else: 
            instance = file
            
        return instance
 
    
    
def results_converter(source, replace = False):
    
    
    """ old results stored as pickled class instances. want to move over to
    pickling class dictionaries instead.
    
    This is much more robust as it doesnt require functions staying the same
    under certain scenarios such as a refactor.
    
    Paramters
    ------
    source : str
        The `source` directory to load and save pickles to/from
        
    replace : bool
    
        If True, replace all files with dictionary versions with the same name.
        Else, return copies of all files with dict_ prefix.
        
    Returns
    -----
    in same folder. returns all files as dictionary pickles instead
    """

    files = glob.glob(source+"*")
    
    try:
        os.chdir("../experiments/ukf_experiments/ukf_old")
    except:
        pass
    
    for file in files:
        file = os.path.split(file)[1]
        u = pickle_main(file, source, True)
        if replace:
                os.remove(source+file)
        else:
            file = "dict_" + file
            
        "check for correct file type ending. old versions have no ending."
        if file[-4:] != ".pkl":
            file += ".pkl"
        u =  pickle_main(file, source, True,instance = u)
