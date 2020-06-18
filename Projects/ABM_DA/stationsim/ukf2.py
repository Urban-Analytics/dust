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

# general packages used for filtering

import os
import numpy as np  # numpy
import datetime  # for timing experiments
import pickle  # for saving class instances
from scipy.stats import chi2  # for adaptive ukf test
import logging
from copy import deepcopy
import multiprocessing


from itertools import repeat

def starmap_with_kwargs(pool, fn, sigmas, kwargs_iter):
    args_for_starmap = zip(repeat(fn), sigmas, kwargs_iter)
    return pool.starmap(apply_args_and_kwargs, args_for_starmap)

def apply_args_and_kwargs(fn, sigmas, kwargs):
    return fn(sigmas, **kwargs)

#from stationsim_model import Model

"""
shamelessly stolen functions for multiprocessing np.apply_along_axis
https://stackoverflow.com/questions/45526700/easy-parallelization-of-numpy-apply-along-axis
"""

def unpacking_apply_along_axis(all_args):
    (func1d, axis, arr, kwargs) = all_args    
    """wrap all args as multiprocessing.pool only takes one argument.
    
    This function is useful with multiprocessing.Pool().map(): (1)
    map() only handles functions that take a single argument, and (2)
    this function can generally be imported from a module, as required
    by map().
    
    Parameters
    ------
    all_args : tuple
        `all_args` all arguments for np.apply_along_axis.
    
        
    Returns
    single_arg : func
        `single_arg` wrapped to a single argument
    """
    single_arg = np.apply_along_axis(func1d, axis, arr, **kwargs)
    return single_arg

def parallel_apply_along_axis(func1d, axis, arr, **kwargs):
    """Like numpy.apply_along_axis(), but takes advantage of multiple cores.
    
    Parameters
    ------
    func1d : func
        function that takes a 1 dimensional argument
    axis : int
        which `axis` to slice and apply func1d on. E.g. if axis = 0 
        on a 2d array we take a full slice of the 0th axis (every row)
        and then a single column looping over the other axis (columns)
    """        
    # Effective axis where apply_along_axis() will be applied by each
    # worker (any non-zero axis number would work, so as to allow the use
    # of `np.array_split()`, which is only done on axis 0):
    effective_axis = 1 if axis == 0 else axis
    if effective_axis != axis:
        arr = arr.swapaxes(axis, effective_axis)

    # Chunks for the mapping (only a few chunks):
    chunks = [(func1d, effective_axis, sub_arr, kwargs)
              for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

    pool = multiprocessing.Pool()
    individual_results = pool.map(unpacking_apply_along_axis, chunks)
    # Freeing the workers:
    pool.close()
    pool.join()

    return np.concatenate(individual_results)




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
    sigmas : array_like
        matrix of MSSPs with each column representing one sigma point

    """
    n = mean.shape[0]
    s = np.linalg.cholesky(p)
    sigmas = np.ones((n, (2*n)+1)).T*mean
    sigmas = sigmas.T
    sigmas[:, 1:n+1] += g*s  # 'upper' confidence sigmas
    sigmas[:, n+1:] -= g*s  # 'lower' confidence sigmas
    return sigmas


# %%

class ukf:
    """main ukf class for assimilating data from some ABM

    Parameters
    ------
    model_params, ukf_params : dict
        dictionary of model `model_params` and ukf `ukf_params` parameters

    base_model : cls
        `base_model` initalised class instance of desired ABM.
    """

    def __init__(self, model_params, ukf_params, base_models, pool):
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

        # init initial state
        "full parameter dictionaries and ABM"
        self.model_params = model_params
        self.ukf_params = ukf_params
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])

        self.base_models = base_models
        self.base_model = deepcopy(base_models[0])

        "pull parameters from dictionary"
        # !!initialise some positions and covariances
        self.x = self.base_model.get_state(sensor="location")
        self.n = self.x.shape[0]  # state space dimension

        "MSSP sigma point scaling parameters"
        self.lam = self.a**2*(self.n+self.k) - self.n
        self.g = np.sqrt(self.n+self.lam)  # gamma parameter

        "unscented mean and covariance weights based on a, b, and k"
        main_weight = 1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-self.a**2+self.b)

        self.xs = []
        self.ps = []
        
        self.pool = pool
        
        self.verbose = True
        if self.verbose:
            self.pxxs = []
            self.pxys = []
            self.pyys = []
            self.ks = []
            self.mus = []
            
    def unscented_Mean(self, sigmas, wm, kf_function, function_kwargs):
        """calculate unscented tmean  estimate for some sample of agent positions
    
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
    
        #calculate either forecasted sigmas X- or measured sigmas Y with f/h
        #nl_sigmas = np.apply_along_axis(kf_function, 0, sigmas, **function_kwargs)
        #now with multiprocessing
        #nl_sigmas = parallel_apply_along_axis(kf_function, 0, sigmas, **function_kwargs).T
        #calculate unscented mean using non linear sigmas and MSSP mean weights
                
        n_sigmas = sigmas.shape[1]
        sigmas_iter = [sigmas[:, i] for i in range(n_sigmas)]
        
        nl_sigmas = starmap_with_kwargs(self.pool, 
                                        kf_function, 
                                        sigmas_iter, 
                                        function_kwargs)
        #nl_sigmas = self.pool.starmap(kf_function, list(zip( \
        #    range(n_sigmas),  # Particle numbers (in integer)
        #    [sigma for sigma in sigmas],
        #    function_kwargs  # Number of iterations to step each particle (an integer)
        #)))
        nl_sigmas = np.vstack(nl_sigmas).T
        xhat = np.dot(nl_sigmas, wm)  # unscented mean for predicitons
    
        return nl_sigmas, xhat
    
    def predict(self, fx_kwargs):
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
        forecasted_sigmas, xhat = self.unscented_Mean(sigmas, 
                                                      self.wm, 
                                                      self.fx,
                                                      fx_kwargs)
        self.sigmas = forecasted_sigmas

        pxx = covariance(forecasted_sigmas, xhat, self.wc, addition=self.q)

        self.p = pxx  # update Sxx
        self.x = xhat  # update xhat

        if self.verbose:
            self.pxxs.append(pxx)

    def update(self, z, hx_kwargs):
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

        nl_sigmas, yhat = self.unscented_Mean(self.sigmas, self.wm,
                                         self.hx, hx_kwargs)
        self.r = np.eye(yhat.shape[0])
        pyy = covariance(nl_sigmas, yhat, self.wc, addition=self.r)
        pxy = covariance(self.sigmas, self.x, self.wc, nl_sigmas, yhat)
        k = np.matmul(pxy, np.linalg.inv(pyy))

        "i dont know why `self.x += ...` doesnt work here"
        mu = (z-yhat)
        self.x = self.x + np.matmul(k, mu)
        
        p = self.p - np.linalg.multi_dot([k, pyy, k.T])

        

        """adaptive ukf augmentation. one for later."""
        adaptive = False
        if adaptive:
            if np.sum(np.abs(mu)) != 0:
                x, p = self.fault_test(
                    z, mu, pxy, pyy, self.x, self.p, k, yhat)

        """record various data for diagnostics. Warning this makes the pickles
        rather large"""
        if self.verbose:
            self.pxys.append(pxy)
            self.pyys.append(pyy)
            self.ks.append(k)
            self.mus.append(mu)
            
        "update mean and covariance states"
        self.ps.append(self.p)
        self.xs.append(self.x)

    def batch(self):
        """
        batch function hopefully coming soon
        """
        return

    def fault_test(self, z, mu, pxy, pyy, x, p, k, yhat):
        """ adaptive UKF augmentation

        -check chi squared test
        -if fails update q and r.
        -recalculate new x and p

        """
        sigma = 0.05

        psi = np.linalg.multi_dot([mu.T, np.linalg.inv(pyy), mu])
        # critical rejection point
        critical = chi2.ppf(1 - sigma, df=mu.shape[0])
        print(psi, critical)
        if psi <= critical:

            "accept null hypothesis. keep q,r"
            pass

        else:

            #update Q
            a = 0.95
            lambd_0 = 0.05
            lambd_1 = 1 - (a * critical)/psi

            lambd = np.max(lambd_0, lambd_1)
            self.q = (1-lambd)*self.q + lambd * \
                np.linalg.multi_dot([k, mu, mu.T, k.T])

            #update R

            eps = z - self.hx(x, **self.hx_kwargs)

            #build new pyy
            #generate new sigmas using current estimates of x and p 
            sigmas = MSSP(x, p, self.g)
            nl_sigmas, yhat = self.unscented_Mean(
                sigmas, self.wm, self.hx, self.hx_kwargs)
            syy = covariance(nl_sigmas, yhat, self.wc)
            b = 0.95
            delta_0 = 0.05
            delta_1 = 1 - (b * critical)/psi
            delta = np.max(delta_0, delta_1)
            self.r = (1-delta)*self.r + delta * \
                (np.linalg.multi_dot([eps, eps.T]) + syy)

            print("noises updated")
            #correct estimates of x and P using new noise"
            #generate covariances and kalman gain"
            pxx = covariance(sigmas, x, self.wc)
            pxy = covariance(self.sigmas, x, self.wc, nl_sigmas, yhat)
            pyy = syy + self.r
            k = np.matmul(pxy, np.linalg.inv(pyy))

            "update x and P using new noises"
            x = self.x + np.matmul(k, mu)
            p = pxx - np.linalg.multi_dot([k, np.linalg.inv(pxy), k.T])

        return x, p


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

    def __init__(self, model_params, ukf_params, base_model, truths = None):
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

        for key in model_params.keys():
            setattr(self, key, model_params[key])
        for key in ukf_params.keys():
            setattr(self, key, ukf_params[key])


        self.base_models = [deepcopy(self.base_model)] * int((2 * 2 * self.pop_total) + 1)
        self.fx_kwargs = []
        for i in range(len(self.base_models)):
            self.fx_kwargs.append({"base_model" : self.base_models[i]})

        self.pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())

        """lists for various data outputs
        observations
        ukf assimilations
        pure stationsim forecasts
        ground truths
        list of covariance matrices
        list of observation types for each agents at one time point
        
        """
        self.obs = []  # actual sensor observations
        self.ukf_histories = []  # ukf predictions
        self.forecasts = []  # pure stationsim predictions
        self.truths = []  # noiseless observations

        self.full_ps = []  # full covariances. again used for animations and not error metrics
        self.obs_key = []  # which agents are observed (0 not, 1 agg, 2 gps)
        self.status_key = []
        
        "timer"
        self.time1 = datetime.datetime.now()  # timer
        self.time2 = None
        
        "save the initial stationsim for batch purposes."

    
    '''
    @classmethod
    def set_random_seed(cls, seed=None):
        """Set a new numpy random seed
        :param seed: the optional seed value (if None then get one from os.urandom)
        
        """
        new_seed = int.from_bytes(os.urandom(
            4), byteorder='little') if seed == None else seed
        np.random.seed(new_seed)
    '''

    def init_ukf(self, ukf_params, seed=None):
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
        #if seed != None:
        #    self.set_random_seed(seed)

        self.ukf = ukf(self.model_params, ukf_params, self.base_models, self.pool)

    def ss_Predict(self):
        """ Forecast step of UKF for stationsim.

        - forecast state using UKF (unscented transform)
        - update forecasts list
        - jump base_model forwards to forecast time
        - update truths list with new positions
        
        """
        """if fx kwargs need updating do it here. Future ABMs may dynamically update parameters as they receive new
        state information. Stationsim does not do this but future ones may so its nice to think about now.
        """
        #if self.fx_kwargs_update_function is not None:
            # update transition function fx_kwargs if necessary. not necessary for stationsim but easy to add later.
            #self.fx_kwargs_update_function(*update_fx_args)


        self.ukf.predict(self.fx_kwargs)
        self.forecasts.append(self.ukf.x)
        #if self.batch:
        #        self.ukf.base_model.set_state(self.batch_truths[step],
        #        "if batch update stationsim with new truths after step"
        #                                      sensor = "location")
        
        self.base_model.step()
        self.truths.append(self.base_model.get_state(sensor="location"))

    def ss_Update(self, step, **hx_kwargs):
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

        if step % self.sample_rate == 0:

            state = self.base_model.get_state(sensor="location").astype(float)
            noise_array = np.ones(self.pop_total*2)
            noise_array[np.repeat([agent.status != 1 for
                                   agent in self.base_model.agents], 2)] = 0
            noise_array *= np.random.normal(0, self.noise, self.pop_total*2)
            state += noise_array

            """if hx function arguements need updating it is done here. 
            for example if observed index changes as 
            with experiment 4 we update it here such 
            0that the ukf takes the correct subset."""
            if self.hx_kwargs_update_function is not None:
                #update measurement function hx_kwargs if necessary
                hx_kwargs = self.hx_kwargs_update_function(state, *self.hx_update_args, **hx_kwargs)
                self.hx_kwargs = hx_kwargs
                self.ukf.hx_kwargs = hx_kwargs

            hx_kwargs_iter = [hx_kwargs] * (4 * (self.pop_total) + 1)
            #convert full noisy state to actual sensor observations
            new_state = self.ukf.hx(state, **self.hx_kwargs)

            self.ukf.update(new_state, hx_kwargs_iter)
            #self.base_model.set_state(self.ukf.x, sensor = "location")
                
            if self.obs_key_func is not None:
                key = self.obs_key_func(state, **self.hx_kwargs)
                #force inactive agents to unobserved
                #removed for now.put back in if we do variable dimensions of desired state
                #key *= [agent.status % 2 for agent in self.base_model.agents]
                self.obs_key.append(key)
                
            
            self.ukf_histories.append(self.ukf.x)  # append histories
            self.obs.append(new_state)

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

        #initialise UKF
        self.init_ukf(self.ukf_params)
        logging.info("ukf start")
        for step in range(self.step_limit-1):
            
            #if self.batch:
            #    if self.base_model.step_id == len(self.batch_truths):
            #        break
            #        print("ran out of truths. maybe batch model ended too early.")
                    
            #forecast next StationSim state and jump model forwards
            self.ss_Predict()
            self.status_key.append([agent.status for agent in self.base_model.agents])
            #assimilate new values
            self.ss_Update(step, **self.hx_kwargs)
            
            if step%100 == 0 :
                logging.info(f"Iterations: {step}")
            finished = self.base_model.pop_finished == self.pop_total
            if finished:  # break condition
                logging.info("ukf broken early. all agents finished")
                break

        self.time2 = datetime.datetime.now()  # timer
        if not finished:
            logging.info(f"ukf timed out. max iterations {self.step_limit} of stationsim reached")
        time = self.time2-self.time1
        time = f"Time Elapsed: {time}"
        print(time)
        logging.info(time)
        
        self.pool.close()
        self.pool.join()
        self.ukf.pool = None
        self.pool = None
        
    """List of static methods for extracting numpy array data 
    """
    @staticmethod
    def truth_parser(self):
        """extract truths from ukf_ss class
        

        Returns
        -------
        truths : "array_like"
            `truths` array of agents true positions. Every 2 columns is an 
            agents xy positions over time. 

        """
        
        truths = np.vstack(self.truths)
        return truths
        
    @staticmethod
    def preds_parser(self, full_mode, truths = None):
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
            if truths is None:
                print("Warning! full mode needs a truths frame for reference.")
            #placeholder frame with same size as truths
            preds = np.zeros((truths.shape[0], self.pop_total*2))*np.nan

            #fill in every sample_rate rows with ukf predictions
            for j in range(int(preds.shape[0]//self.sample_rate)):
                preds[j*self.sample_rate, :] = raw_preds[j, :]
        else:
            #if not full_mode returns preds as is
            preds = raw_preds
        return preds
    
    @staticmethod
    def obs_parser(self, full_mode, truths = None, obs_key = None):
        """Parse sensor observations I.E. observed state from ukf class
        
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
        obs_key : `array_like`
        Array of the same shape as truths indicating whether an agents positions
        are observered either directly, via some aggregate, or not at all.
            
        Returns
        -------
        obs : "array_like"
            numpy array of observations `obs` from observed state.
            
        !!todo issues aligning obs_key and observations dimensions
        e.g. trying to fit a 1x4 observation into 1x6 index
        """
        
        raw_obs = self.obs
    
        if full_mode:
            #print full preds with sample_rate blank gaps between rows.
            #we want the predictions to have the same number of rows
            #as the true data.
            if truths is None:
                print("Warning! full mode needs a truths frame for reference.")
            #placeholder frame with same size as truths
            obs = np.zeros((truths.shape[0], self.pop_total*2))*np.nan

            #fill in every sample_rate rows with ukf predictions
            for i in range(len(raw_obs)):
                index = np.where(obs_key[i*self.sample_rate, :]==2)
                index2 = 2*np.repeat(index,2)
                index2[1::2]+=1
                if index2.shape[0]>0:
                    obs[((i)*self.sample_rate), index2] = raw_obs[i]
                
        else:
            #if not full_mode returns preds as is
            obs = raw_obs
        return obs
        
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

    def obs_key_parser(self):
        """extract obs_key
        """
        obs_key2 = np.vstack(self.obs_key)
        shape = np.vstack(self.truths).shape[0]
        obs_key = np.zeros((shape, self.pop_total))*np.nan

        for j in range(int(shape//self.sample_rate)):
            obs_key[j*self.sample_rate, :] = obs_key2[j, :]

        return obs_key

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
