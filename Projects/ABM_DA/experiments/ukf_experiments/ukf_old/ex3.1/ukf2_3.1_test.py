# -*- coding: utf-8 -*-

"""
The Unscented Kalman Filter (UKF) designed to be hyper efficient alternative to similar 
Monte Carlo techniques such as the Particle Filter. This file aims to unify the old 
intern project files by combining them into one single filter. It also aims to be geared 
towards real data with a modular data input approach for a hybrid of sensors.
This is based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
This file has no main. use ukf notebook/modules in experiments folder
!!dsitinguish between fx_kwargs and fx_kwargs_iter
"""

# general packages used for filtering

import os
import sys
import numpy as np  # numpy
import datetime  # for timing experiments
import pickle  # for saving class instances
import logging
from copy import deepcopy
import multiprocessing
from itertools import repeat

"""functions to modify starmap to allow additional kwargs
"""
def starmap_with_args_kwargs(pool, fn, args_iter, kwargs_iter):
    args_for_starmap = zip(repeat(fn), args_iter, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, args, kwargs):
    return fn(*args, **kwargs)    
  
def starmap_with_kwargs(pool, fn, sigmas, kwargs_iter):
    args_for_starmap = zip(repeat(fn), sigmas, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, sigmas, kwargs):
    return fn(sigmas, **kwargs)    
      
def model_step(model):
    with HiddenPrints():
        model.step()
    return model

class HiddenPrints:
    """stop repeat printing from stationsim 
    We get a lot of `iterations : X` prints as it jumps back 
    and forth over every 100th step. This stops that.
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
    """
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def covariance(data1, mean1, weight, data2=None, mean2=None, addition=None):
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

    covariance_matrix = np.linalg.multi_dot(
        [residuals, weighting, residuals2.T])

    """old versions"""
    "old quadratic form version. made faster with multi_dot."
    # covariance_matrix = np.matmul(np.matmul((data1.transpose()-mean1).T,np.diag(weight)),
    #                (data1.transpose()-mean1))+self.q

    "numpy quadratic form far faster than this for loop"
    #covariance_matrix =  self.wc[0]*np.outer((data1[:,0].T-mean1),(data2[:,0].T-mean2))+self.Q
    # for i in range(1,len(self.wc)):
    #    pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)

    "if some additive noise is involved (as with the Kalman filter) do it here"

    if addition is not None:
        covariance_matrix += addition

    return covariance_matrix


def MSSP(mean, p, g):
    """merwe's scaled sigma point calculations based on current mean x and covariance P
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
    sigmas : list
        list of MSSPs with each item representing one sigma point.
    """
    n = mean.shape[0]
    s = np.linalg.cholesky(p)
    sigmas = np.ones((n, (2*n)+1)).T*mean
    sigmas = sigmas.T
    sigmas[:, 1:n+1] += g*s  # 'upper' confidence sigmas
    sigmas[:, n+1:] -= g*s  # 'lower' confidence sigmas
    sigmas = sigmas.T.tolist()
    return sigmas


def unscented_Mean(sigmas, wm):
        """calculate unscented mean  estimate for some sample of agent positions

        Parameters
        ------
        sigmas, wm : array_like
            `sigmas` array of sigma points  and `wm` mean weights
    
        Returns 
        ------
        u_mean : array_like
            unscented mean of `u_mean` of sigmas.
        """
        sigmas = np.vstack(sigmas).T
        u_mean = np.dot(sigmas, wm)  # unscented mean for predicitons
        
        return u_mean

# %%
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

    def __init__(self, model_params, ukf_params, base_model, base_models = None,
                 truths = None):
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
        self.model_params = model_params  # stationsim parameters
        self.ukf_params = ukf_params  # ukf parameters
        self.base_model = base_model  # station sim
        #burn in to avoid wierd gcss 0s

        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])

        #list of particle models
        if base_models is None:
            self.base_models = []
            for i in range(int((4 * self.pop_total) + 1)):
                self.base_models.append(deepcopy(self.base_model))
        else:
            self.base_models = base_models
            
        #initial hx_kwargs iterable for starmap
        self.hx_kwargs_iter = [self.hx_kwargs] * (4 * (self.pop_total) + 1)



        #placeholder lists for various things
        self.obs = []  # actual sensor observations
        self.ukf_histories = []  # ukf predictions
        self.forecasts = []  # pure stationsim predictions
        self.truths = []  # noiseless observations
        self.ps = []

        self.obs_key = []  # which agents are observed (0 not, 1 agg, 2 gps)
        self.status_key = [] #status of agents over time (0 not started, 1 in progress, 2 finished)
        
        #initial ukf parameters/sigma weightings
        self.x = self.x0
        #initial covariance p in ukf_params. usually just identity matrix
        #same for q and r
        self.n = self.x.shape[0]  # state space dimension

        #MSSP sigma point scaling parameters
        self.lam = self.a**2*(self.n+self.k) - self.n
        self.g = np.sqrt(self.n+self.lam)  # gamma parameter
        self.sigmas = None

        #unscented mean and covariance weights based on a, b, and k
        main_weight = 1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-self.a**2+self.b)

                
        self.verbose = False
        
        if self.verbose:
            self.pxxs = []
            self.pxys = []
            self.pyys = []
            self.ks = []
            self.mus = []
        
        #timer
        self.time1 = datetime.datetime.now()  # timer
        self.time2 = None

    def ss_Predict(self, ukf_step):
        """ Forecast step of UKF for stationsim.
        - if at step 0 or after an update step generate new sigmas points
            and assign them to base_models
        - otherwise step the base_models with their given sigma points.
        
        Parameters
        ------
        step : int
            time point of main base_model
        """
        
        """if fx kwargs need updating do it here. Future ABMs may dynamically update parameters as they receive new
        state information. Stationsim does not do this but future ones may so its nice to think about now.
        """
        #if self.fx_kwargs_update_function is not None:
            # update transition function fx_kwargs if necessary. not necessary for stationsim but easy to add later.
            #self.fx_kwargs_update_function(*update_fx_args)
        
        if self.sigmas is None:
            self.sigmas = MSSP(self.x, self.p, self.g)
            for i, sigma in enumerate(self.sigmas):
                self.base_models[i] = self.model_set(self.base_models[i], sigma, 
                                                     **self.model_set_kwargs)

        print(self.base_models[0].step_id)        
        if self.pop_total <= 20 and self.station == None:
            with HiddenPrints():
                for model in self.base_models:
                            model.step()
                            
        else:
            pool = multiprocessing.Pool()
            self.base_models = pool.starmap(model_step, zip(self.base_models))
            pool.close()
            pool.join()
            
    def ss_Update(self, ukf_step, **hx_kwargs):
        """ Update step of UKF for stationsim.
        - if step is a multiple of sample_rate
        - step base_models forwards
        - get unscented mean and covariance estimate for desired state x pxx
        - measure state from base_model applying gaussian noise and converting it
            into observations using hx.
        - assimilate ukf with projected noisy osbervations.
        - calculate each agents observation type with obs_key_func.
        - append lists of ukf assimilations and model observations
        
        Parameters
        --------
        ukf_step : int
            `ukf_step` base model step id
        hx_kwargs : kwargs
            `hx_kwargs` any kwargs other than the true base model state needed to 
            run hx.
            
        Returns
        ------
        None.
        """
        #calculate desired state unscented mean and covariance
        self.fx_kwargs = [self.fx_kwargs_update(self)] * ((4 * self.n) + 1)
        
        pool = multiprocessing.Pool()
        self.sigmas = pool.map(self.fx, self.base_models)
        pool.close()
        pool.join()
        
        xhat = unscented_Mean(self.sigmas, self.wm)
        stacked_sigmas = np.vstack(self.sigmas).T
        pxx = covariance(stacked_sigmas, xhat, self.wc, addition=self.q)

        self.p = pxx  # update Sxx
        self.x = xhat  # update xhat

        #add noise to base_model true state
        state = self.base_model.get_state(sensor="location").astype(float)
        if self.noise != 0:
            noise_array = np.ones(self.pop_total*2)
            noise_array[np.repeat([agent.status != 1 for
                                   agent in self.base_model.agents], 2)] = 0
            noise_array *= np.random.normal(0, self.noise, self.pop_total*2)
            state += noise_array

        """if hx function arguements need updating it is done here. 
        for example if observed index changes e.g. if an agent
        changes from observed to unobserved due to leaving a camera's
        vision."""
        
        if self.hx_kwargs_update_function is not None:
            #update measurement function hx_kwargs if necessary
            hx_kwargs = self.hx_kwargs_update_function(state, *self.hx_update_args, **hx_kwargs)
            self.hx_kwargs = hx_kwargs
            self.ukf.hx_kwargs = hx_kwargs
            
            #hx_kwargs iterable for starmap_with_kwargs
            self.hx_kwargs = [hx_kwargs] * (4 * (self.pop_total) + 1)
            
        #convert full noisy true state to actual sensor observations
        new_state = self.hx(state, **self.hx_kwargs)
        
        #append obs key list if there is any. useful to have for nice plots later.
        if self.obs_key_func is not None:
            key = self.obs_key_func(state, **self.hx_kwargs)
            #force inactive agents to unobserved
            #removed for now.put back in if we do variable dimensions of desired state
            #key *= [agent.status % 2 for agent in self.base_model.agents]
            self.obs_key.append(key)
            
        
        #calculate unscenterd mean and covariance of observed state
        nl_sigmas = [self.hx(sigmas, **self.hx_kwargs) for sigmas in self.sigmas]
        #pool = multiprocessing.Pool()
        #nl_sigmas = starmap_with_kwargs(pool, 
        #                            self.hx, 
        #                            self.sigmas, 
        #                           self.hx_kwargs)
        #pool.close()
        #pool.join()
        #pool = None
        stacked_nl_sigmas = np.vstack(nl_sigmas).T
        yhat = unscented_Mean(nl_sigmas, self.wm)
        pyy = covariance(stacked_nl_sigmas, yhat, self.wc, addition=self.r)
        pxy = covariance(stacked_sigmas, self.x, self.wc, stacked_nl_sigmas, yhat)
        
        #assimilate predictions of x with observations
        self.k = np.matmul(pxy, np.linalg.inv(pyy))
        "i dont know why `self.x += ...` doesnt work here"
        mu = (new_state - yhat)
        self.x = self.x + np.matmul(self.k, mu)
        
        self.p = self.p - np.linalg.multi_dot([self.k, pyy, self.k.T])

        #append agent states, covariances, and observation types
        self.ukf_histories.append(self.x)  # append histories
        self.obs.append(new_state)
        self.ps.append(self.p)
        self.forecasts.append(xhat)
        self.sigmas = None
        
        #record various data for diagnostics. Warning this makes the pickles rather large
        if self.verbose:
            self.pxxs.append(pxx)
            self.pxys.append(pxy)
            self.pyys.append(pyy)
            self.ks.append(self.k)
            self.mus.append(mu)
            

    def step(self, ukf_step):
        """ukf step function
        - initiates ukf
        - while any agents are still active
        - predict with ukf
        - step true model
        - update ukf with new model positions
        - repeat until all agents finish or max iterations reached
        - if no agents then stop
        
        Parameters
        --------
        ukf_step : int
            `ukf_step` base model step id
        hx_kwargs : kwargs
            `hx_kwargs` any kwargs other than the true base model state needed to 
            run hx.
            
        Returns
        ------
        None.
        """
        
        self.ss_Predict(ukf_step)
        self.status_key.append([agent.status for agent in self.base_model.agents])
        self.base_model.step()
        self.truths.append(self.base_model.get_state(sensor="location"))    
        
        if ukf_step % self.sample_rate == 0 and ukf_step > 0:
            #assimilate new values
            self.ss_Update(ukf_step, **self.hx_kwargs)
            
    def main(self):
        
        """main function for applying ukf to gps style station StationSim
    
        - for each time step
        - if time 0 or step_limit + 1
            - generate new sigmas and step base_models forwards
        - if time is step_limit 
            - step base_models forwards and do update step and collapse sigmasd
        - else just step base_models forwards.
        
        Returns
        ------
        None.
        """

        #initialise UKF

        logging.info("ukf start")
        for ukf_step in range(self.step_limit):
            
            #if self.batch:
            #    if self.base_model.step_id == len(self.batch_truths):
            #        break
            #        print("ran out of truths. maybe batch model ended too early.")
                    
            #forecast next StationSim state and jump model forwards
            self.step(ukf_step)
            if ukf_step%100 == 0 :
                logging.info(f"Iterations: {ukf_step}")
            if self.base_model.pop_finished == self.pop_total:  # break condition
                logging.info("ukf broken early. all agents finished")
                break
            if ukf_step == self.step_limit:
                logging.info(f"ukf timed out. max iterations {self.step_limit} of stationsim reached.")

        self.time2 = datetime.datetime.now()  # timer
        time = self.time2-self.time1
        self.time = f"Time Elapsed: {time}"
        print(self.time)
        logging.info(self.time)
        
    """List of static methods for extracting numpy array data 
    """
    @staticmethod
    def truth_parser(self):
        """extract truths from ukf_ss class
        
        Returns
        -------
        truths : "array_like"
            `truths` array of agents true positions. Every 2 columns is an 
            agents xy positions over time bottom filled with np.nans. 
        """
        
        truths = np.vstack(self.truths)
        return truths
        
    @staticmethod
    def preds_parser(self, full_mode):
        """Parse ukf predictions of agents positions from some ukf class
        
        Parameters
        -------
        full_mode : 'bool'
        'full_mode' determines whether we print the ukf predictions as is
        or fill out a data frame for every time point. For example, if we have
        200 time points for a stationsim run and a sample rate of 5 we will
        get 200/5 = 40 ukf predictions. If we set full_mode to false we get an
        array with 40 rows. If we set it to True we get the full 200 rows with
        4 blanks rows for every 5th row of data. This is very useful for 
        plots later as it standardises the data times and makes animating much,
        much easier.
        
        truths : `array_like`
            `truths` numpy array to reference for full shape of preds array.
            I.E how many rows (time points) and columns (agents)
        
        Returns
        -------
        preds : `array_like`
            `preds` numpy array of ukf predictions.
        """
        
        raw_preds = np.vstack(self.ukf_histories)

        if full_mode:
            #print full preds with sample_rate blank gaps between rows.
            #we want the predictions to have the same number of rows
            #as the true data.
            #placeholder frame with same size as truths
            preds = np.zeros((len(self.truths), self.pop_total*2))*np.nan

            #fill in every sample_rate rows with ukf predictions
            preds[self.sample_rate::self.sample_rate ,:] = raw_preds
        else:
            #if not full_mode returns preds as is
            preds = raw_preds
        return preds
    
    @staticmethod
    def forecasts_parser(self, full_mode):
        """Parse ukf predictions of agents positions from some ukf class
        
        Parameters
        -------
        full_mode : 'bool'
        'full_mode' determines whether we print the ukf predictions as is
        or fill out a data frame for every time point. For example, if we have
        200 time points for a stationsim run and a sample rate of 5 we will
        get 200/5 = 40 ukf predictions. If we set full_mode to false we get an
        array with 40 rows. If we set it to True we get the full 200 rows with
        4 blanks rows for every 5th row of data. This is very useful for 
        plots later as it standardises the data times and makes animating much,
        much easier.
        
        truths : `array_like`
            `truths` numpy array to reference for full shape of preds array.
            I.E how many rows (time points) and columns (agents)
        
        Returns
        -------
        forecasts : `array_like`
            `preds` numpy array of ukf predictions.
        """
        
        raw_forecasts = np.vstack(self.forecasts)

        if full_mode:
            #print full preds with sample_rate blank gaps between rows.
            #we want the predictions to have the same number of rows
            #as the true data.
            #placeholder frame with same size as truths
            forecasts = np.zeros((len(self.truths), self.pop_total*2))*np.nan

            #fill in every sample_rate rows with ukf predictions
            forecasts[self.sample_rate::self.sample_rate ,:] = raw_forecasts
        else:
            #if not full_mode returns preds as is
            forecasts = raw_forecasts
        return forecasts
    
    @staticmethod
    def obs_parser(self, full_mode):
        """Parse ukf predictions of agents positions from some ukf class
        
        Parameters
        -------
        full_mode : 'bool'
        'full_mode' determines whether we print the ukf predictions as is
        or fill out a data frame for every time point. For example, if we have
        200 time points for a stationsim run and a sample rate of 5 we will
        get 200/5 = 40 ukf predictions. If we set full_mode to false we get an
        array with 40 rows. If we set it to True we get the full 200 rows with
        4 blanks rows for every 5th row of data. This is very useful for 
        plots later as it standardises the data times and makes animating much,
        much easier.
        
        truths : `array_like`
            `truths` numpy array to reference for full shape of preds array.
            I.E how many rows (time points) and columns (agents)
        
        Returns
        -------
        forecasts : `array_like`
            `preds` numpy array of ukf predictions.
        """
        
        raw_obs = np.vstack(self.obs)
        raw_obs_key = np.vstack(self.obs_key)
        
        if full_mode:
            #print full preds with sample_rate blank gaps between rows.
            #we want the predictions to have the same number of rows
            #as the true data.
            #placeholder frame with same size as truths
            obs = np.zeros((len(self.truths), self.pop_total*2))*np.nan
            obs_key = np.zeros((len(self.truths), self.pop_total))*np.nan
            
            obs_key[self.sample_rate::self.sample_rate, :] = raw_obs_key
            
            #fill in every sample_rate rows with ukf predictions
            where_observed = np.where(obs_key == 2)
            where = where_observed[1][:self.pop_total]
            where = np.repeat(2 * where, 2)
            where[1::2] += 1
            
            obs[self.sample_rate::self.sample_rate , where] = raw_obs
        else:
            #if not full_mode returns preds as is
            obs = obs
        return obs, obs_key
    
    @staticmethod

    def nan_array_parser(self, truths, base_model):
        """ Indicate when an agent leaves the model to ignore any outputs.
        
        Returns
        -------
        nan_array : "array_like"
        
        The `nan_array` indicates whether an agent is in the model
        or not for a given time step. This will be 1 if an agent is in and
        nan otherwise. We can times this array with our ukf predictions
        for nicer looking plots that cut the wierd tails off. In the long term,
        this could be replaced if the ukf removes states as they leave the model.
        """
        
        nan_array = np.ones(truths.shape)*np.nan
        status_array = np.vstack(self.status_key)
        
        #for i, agent in enumerate(base_model.agents):
            #find which rows are  NOT (None, None). Store in index.
        #    array = np.array(agent.history_locations[:truths.shape[0]])
        #    in_model = np.where(array!=None)
            #set anything in index to 1. I.E which agents are still in model to 1.
        #    nan_array[in_model, 2*i:(2*i)+2] = 1
        rows = np.repeat(np.where(status_array == 1)[0],2)
        columns = np.repeat(2 * np.where(status_array == 1)[1], 2)
        columns[::2]+=1
        
        nan_array[rows, columns] = 1
        
        return nan_array

def batch_save(model_params, n, seed):
    "save a stationsim model to use later in a batch"

    model_params["random_seed"] = seed
    model_params["pop_total"] = n
    base_model = Model(**model_params)
    "copy start of model for saving in batch later."
    "needed so we dont assign new random speeds"
    start_model = deepcopy(base_model)
    
    while base_model.status == 1:
        base_model.step()
        
    
    truths = base_model.history_state
    for i in range(len(truths)):
        truths[i] = np.ravel(truths[i])
    
    batch_pickle = [truths, start_model]
    n = model_params["pop_total"]
    
    pickler(batch_pickle, "pickles/", f"batch_test_{n}_{seed}.pkl")

def batch_load(file_name):
    "load a stationsim model to use as a batch for ukf"
    batch_pickle = depickler("pickles/", file_name)
    truths = batch_pickle[0]
    start_model = batch_pickle[1]
    return truths, start_model

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

    f = open(pickle_source + f_name, "wb")
    pickle.dump(instance, f)
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
    f = open(pickle_source+f_name, "rb")
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


def pickle_main(f_name, pickle_source, do_pickle, instance=None):
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
        print(f"Pickling file to {f_name}")
        pickler(instance.__dict__, pickle_source, f_name)
        return

    else:
        file = depickler(pickle_source, f_name)
        print(f"Loading pickle {f_name}")
        "try loading the specified file as a class dict. else an instance."
        if type(file) == dict:
            "removes old ukf function in memory"

            instance = class_dict_to_instance(file)
        else:
            instance = file

        return instance