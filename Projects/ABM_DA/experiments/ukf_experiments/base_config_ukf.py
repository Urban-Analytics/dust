
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

#for filter
import numpy as np
from math import floor
import datetime
import multiprocessing
from copy import deepcopy
import os 
import sys 
import pickle

#due to import errors from other directories
sys.path.append("..")
from stationsim.stationsim_model import Model
from stationsim.ukf import plots
#for plots

"""
As of 3.6 only imageio (and ffmpeg dependency) and scipy.spatial are additional installs
pip install imageio
pip install ffmpeg
pip install scipy


suppress repeat printing in F_x from new stationsim
E.g. 
with HiddenPrints():
    everything done here prints nothing

everything here prints again
https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
"""

"for dark plots. purely an aesthetic choice. plt.style.available() for other styles"
#plt.style.use("dark_background")

#%%
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

"""general ukf class"""
class ukf:
    
    def __init__(self,ukf_params,init_x,fx,hx,P,Q,R):
        """
        x - state
        n - state size 
        P - state covariance
        fx - transition function
        hx - measurement function
        lam - lambda paramter
        g - gamma parameter
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

    def Sigmas(self,mean,S):
        """
        sigma point calculations based on current mean x and  covariance matrix square root S
        in:
            mean x (n by 1)
            covariance square root S (n by n)
            
        out:
            2*n+1 rows of n dimensional sigma points
        """
        
     
        sigmas = np.ones((self.n,(2*self.n)+1)).T*mean
        sigmas=sigmas.T
        sigmas[:,1:self.n+1] += self.g*S #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*S #'lower' confidence sigmas
        return sigmas 

    def predict(self,**fx_args):
        """
        - calculate sigmas using prior mean and UT element of covariance S
        - predict interim sigmas X for next timestep using transition function Fx
        - predict unscented mean for next timestep
        - calculate interim S using concatenation of all but first column of Xs
            and square root of process noise
        - cholesky update to nudge on unstable 0th row
        - calculate futher interim sigmas using interim S and unscented mean
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
        """
        Does numerous things in the following order
        - calculate interim sigmas using Sxx and unscented mean estimate
        - calculate measurement sigmas Y = h(X)
        - calculate unscented mean of Ys
        - calculate qr decomposition of concatenated columns of all but first Y scaled 
            by w1c and square root of sensor noise to calculate interim S
        - cholesky update to nudge interim S on potentially unstable 0th 
            column of Y
        - calculate sum of scaled cross covariances between Ys and Xs Pxy
        - calculate kalman gain
        - calculate x update
        - calculate S update
        """
        
        """
        posterior sigmas using above unscented interim estimates for x and P
        """
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.P)) #update using Sxx and unscented mean
        nl_sigmas = np.apply_along_axis(self.hx,0,sigmas)
        #p = multiprocessing.Pool()
        #nl_sigmas = np.vstack(p.map(self.hx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        #p.close()
        wnl_sigmas = nl_sigmas*self.wm

        """
        unscented estimate of posterior mean using said posterior sigmas
        """
        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        
        
        "similar weighted estimates as Pxx for cross covariance Pxy and observation covariance Pyy"
        "now with quadratic form"
        
        Pyy = np.matmul(np.matmul((nl_sigmas.transpose()-yhat).T,np.diag(self.wc)),(nl_sigmas.transpose()-yhat))+self.R
        
        "old for loop version. not sure on the speed but above makes sense in my opinion"
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
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        self.P = self.P -  np.matmul(K,np.matmul(Pyy,K.T))
        
        self.Ps.append(self.P)
        self.xs.append(self.x)
        
        
        
    def batch(self):
        """
        batch function maybe build later
        """
        return

class ukf_ss:
    """
    UKF for station sim using ukf filter class.
    """
    def __init__(self,model_params,filter_params,ukf_params,base_model):
        """
        in:
            *_params - loads in parameters for the model, station sim filter and general UKF parameters
            base_model - initiate stationsim 
        
        out:
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
        self.base_model = base_model #station sim
        
        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        #number of batch iterations
        self.number_of_iterations = model_params['step_limit']
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
        
        
        self.ukf_truth = []#true
        self.ukf_actual= []#actual / noisy true
        self.ukf_preds=[] #predicted
        self.ukf_histories = [] #assimilated 
        
        

    
        self.time1 =  datetime.datetime.now()#timer
        self.time2 = None
    def fx(self,x,**fx_args):
        """
        Transition function for the state space giving where it is predicted to be
        at the next time step.

        In this case it is a placeholder which receives a vector a base_model class
        instance with specified agent locations and speed and predicts where
        they will be at the next time step
        
        in:
            base_model class with current agent attributes
        out:
            base_model positions predicted for next time step
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
        """
        Measurement function for aggregates.
        This converts our state space with latent variables output by fx 
        into one with the same state space as what we can observe.
        For example, if we may use position and speed in our
        transition function fx to predict the state at the next time interval.
        If we can only measure position then this function may 
        just simply omit the speed or use it to further estimate the position.
        
        In this case this function simply omits agents which cannot be observed.
        
        in:
            full latent state space output by fx
        out: 
            vector of aggregates from measuring how many agents in each polygon in poly_list 
        """
        state = state[self.index2]
        
        return state
    
    def init_ukf(self,ukf_params):
        """
        initialise ukf with initial state and covariance structures.
        in:
            base model
            number of agents
            some list of aggregate polygons
            some transition f and measurement h functions
            
        out:
            initalised ukf class object
        """
        
        x = self.base_model.get_state(sensor="location")#initial state
        Q = np.eye(self.pop_total*2)#process noise
        R = np.eye(len(self.index2))#sensor noise
        P = np.eye(self.pop_total*2)#inital guess at state covariance
        self.ukf = ukf(ukf_params,x,self.fx,self.hx,P,Q,R)
        self.ukf_truth.append(self.base_model.get_state(sensor="location"))#truth init
        self.ukf_actual.append(self.base_model.get_state(sensor="location"))#actual init

    
    def main(self):
        """
        main function for ukf station sim
        -initiates ukf
        while any agents are still active
            -predict with ukf
            -step true model
            -update ukf with new model positions
            -repeat until all agents finish or max iterations reached
            
        in: 
            __init__ with various parameters including base model, parameters for
            filter,ukf, and model, which agents (if any) are unobserved and
            storage for data
        
        out:
            -agents trajectories and UKF predictions of said trajectories
        """
        
        #seeding if  wanted else hash it

        self.init_ukf(self.ukf_params) 
        for _ in range(self.number_of_iterations-1):

            #f_name = f"temp_pickle_model_ukf_{self.time1}"
            #f = open(f_name,"wb")
            #pickle.dump(self.base_model,f)
            #f.close()
            
            
            self.ukf.predict() #predict where agents will jump
            self.ukf_preds.append(self.ukf.x) #preds
                        
            "apply noise to active agents"
          
            self.base_model.step() #jump stationsim agents forwards
            
            self.ukf_truth.append(self.base_model.get_state(sensor="location"))#truth
            if self.filter_params["bring_noise"]:
                noise_array=np.ones(self.pop_total*2)
                noise_array[np.repeat([agent.status!=1 for agent in self.base_model.agents],2)]=0
                noise_array*=np.random.normal(0,self.filter_params["noise"],self.pop_total*2)
                state = self.base_model.get_state(sensor="location") #observed agents states
                state+=noise_array
                self.base_model.set_state(state=state,sensor="location")
                self.ukf_actual.append(state)#actual
                

            "DA update step and data logging"
            "data logged for full preds and only assimilated preds (just predict step or predict and update)"
            if _%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                self.ukf.update(z=state) #update UKF
                
                self.ukf_histories.append(self.ukf.x) #append histories

                x = self.ukf.x
                if np.sum(np.isnan(x))==x.shape[0]:
                    print("math error. try larger values of alpha else check fx and hx.")
                    break

                ""
            if self.base_model.pop_finished == self.pop_total: #break condition
                break
        
        self.time2 = datetime.datetime.now()#timer
        print(self.time2-self.time1)
        
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
        
        truth = np.vstack(self.ukf_truth)
        actual = np.vstack(self.ukf_actual)
        preds = np.vstack(self.ukf_preds)
        if self.filter_params["sample_rate"]==1:
            histories = np.vstack(self.ukf_histories)
        else:
            histories = np.ones(truth.shape)*np.nan
            histories2 = np.vstack(self.ukf_histories)
            histories[::self.sample_rate,:] = histories2
        return truth,actual,preds,histories


#%%
if __name__ == "__main__":
    #np.random.seed(seed = 8)
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
    do_restrict - restrict to a proportion prop of the agents being observed
    do_animate - bools for doing animations of agent/wiggle aggregates
    do_wiggle_animate
    do_density_animate
    do_pair_animate
    prop - proportion of agents observed. this is a floor function that rounds the proportion 
        DOWN to the nearest intiger number of agents. 1 is all <1/pop_total is none
    
    heatmap_rate - after how many updates to record a frame
    bin_size - square sizes for aggregate plots,
    bring_noise: add noise to true ukf paths
    noise -  variance of said noise producing IID p dimensional normal noise e~N_p(0,diag(noise))
    do_batch - do batch processing on some pre-recorded truth data.
       np.save(f"ukf_results/agents_{n}_rate_{r}_noise_{var}_base_config_errors_{run_id}",means) """
    
    filter_params = {      
            "Sensor_Noise":  1, 
            "Process_Noise": 1, 
            'sample_rate': 50,
            "do_restrict": True, 
            "do_animate": False,
            "prop": 1,
            "heatmap_rate": 1,
            "run_id":0,
            "bin_size":10,
            "bring_noise":True,
            "noise":0,
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
            
            "a":.1,
            "b":2,
            "k":0,

            }
    
    """run and extract data"""
    base_model = Model(**model_params)
    u = ukf_ss(model_params,filter_params,ukf_params,base_model)
    u.main()
    true,actual,preds,histories= u.data_parser(True)
    plts=plots(u)
    errors = {}
    errors["actual"] = plts.AEDs(true,actual)
    errors["preds"] = plts.AEDs(true[1:,:],preds)
    errors["ukf"] = plts.AEDs(true[1:,:],histories)
    
    means = []
    for key in errors.keys():
        means.append(np.nanmean(errors[key][0]))
    
    means = np.array(means)
    n = model_params["pop_total"]
    r = filter_params["sample_rate"]
    var = filter_params["noise"]
    run_id = filter_params["run_id"]
    
    f_name = "ukf_results/agents_{}_rate_{}_noise_{}_base_config_errors_{}".format(str(n),str(r),str(var), str(run_id))
    f = open(f_name,"wb")
    pickle.dump(means,f)
    f.close()