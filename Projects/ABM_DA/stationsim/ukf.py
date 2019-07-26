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
from math import floor,log10,ceil,cos,sin
import matplotlib.pyplot as plt
import datetime
import multiprocessing
from copy import deepcopy
import os 
import sys 

#due to import errors from other directories
sys.path.append("..")
from stationsim.stationsim_model import Model

#for plots

import matplotlib.gridspec as gridspec
import imageio
from scipy.stats import norm
from shutil import rmtree
from filterpy.stats import covariance_ellipse #needed solely for pairwise_frames_stack_ellipse for covariance ellipse plotting
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
plt.style.use("dark_background")

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
        #nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
        p = multiprocessing.Pool()
        nl_sigmas = np.vstack(p.map(self.fx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        p.close()
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
        
        
        "similar weighted estimates as Pxx for cross covariance and posterior covariance"
        "now with quadratic form"
        
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
        self.x += np.matmul(K,(z-yhat))
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        Pxx = self.P
        Pxx -= np.matmul(K,np.matmul(Pyy,K.T))
        
        self.P = Pxx
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
        
        self.ukf_histories = []
   
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
            #if _%100 ==0: #progress bar
            #    print(f"iterations: {_}")
                

            #f_name = f"temp_pickle_model_ukf_{self.time1}"
            #f = open(f_name,"wb")
            #pickle.dump(self.base_model,f)
            #f.close()
            
            
            self.ukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            

            if self.base_model.step_id%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                
                state = self.base_model.get_state(sensor="location") #observed agents states
                self.ukf.update(z=state[self.index2]) #update UKF
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
        sample_rate = self.sample_rate


        a2 = {}
        for k,agent in  enumerate(self.base_model.agents):
            a2[k] =  agent.history_locations
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

class plots:
    """
    class for all plots using in UKF
    """
    def __init__(self,filter_class):
        "define which class to plot from"
        self.filter_class=filter_class
        
    def plot_data_parser(self,a,b,observed):
        """
        takes data from ukf_ss data parser and preps it for plotting
        in: 
            ukf_ss class
        out: 
            split data depending on plotting observed or unobserved agents
            
        """

        filter_class = self.filter_class
        if observed:
                a = a[:,filter_class.index2]
                if len(filter_class.index2)<b.shape[1]:
                    b = b[:,filter_class.index2]
                plot_range =filter_class.model_params["pop_total"]*(filter_class.filter_params["prop"])

        else:      
                mask = np.ones(a.shape[1])
                mask[filter_class.index2]=False
                a = a[:,np.where(mask!=0)][:,0,:]
                b = b[:,np.where(mask!=0)][:,0,:]
                plot_range = filter_class.model_params["pop_total"]*(1-filter_class.filter_params["prop"])
        return a,b,plot_range

        
    def MAEs(self,a,b):
        """
        MAE (mean absolute error) metric. 
        finds mean average euclidean error at each time step and per each agent
        provides whole array of distances per agent and time
        and MAEs per agent and time. 
        """
        c = np.ones((a.shape[0],int(a.shape[1]/2)))*np.nan
        
        "!!theres probably a faster way of doing this with apply over axis"
        #loop over each agent
        a = a[:,::self.filter_class.sample_rate]
        b = b[:,::self.filter_class.sample_rate]


        #loop over each time per agent
        for i in range(a.shape[0]):
                a2 = np.array([a[i,0::2],a[i,1::2]]).T
                b2 = np.array([b[i,0::2],b[i,1::2]]).T
                res = a2-b2
                c[i,:]=np.apply_along_axis(np.linalg.norm,1,res)
                    
        agent_means = np.nanmean(c,axis=0)
        time_means = np.nanmean(c,axis=1)
        return c,agent_means,time_means
        
    def diagnostic_plots(self,a,b,observed,save):
        """
        self - UKf class for various information
        
        a-observed agents
        
        b-UKF predictions of a
        
        observed- bool for plotting observed or unobserved agents
        if True observed else unobserved
        
        save- bool for saving plots in current directory. saves if true
        
        
        """
        a,b,plot_range = self.plot_data_parser(a,b,observed)
        
        f=plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(a[:,(2*j)],a[:,(2*j)+1])    
            plt.title("True Positions")

        g = plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(b[::self.filter_class.sample_rate,2*j],b[::self.filter_class.sample_rate,(2*j)+1])    
            plt.title("KF predictions")
            
        """
        MAE metric. 
        finds mean average euclidean error at each time step and per each agent
        """
        c,agent_means,time_means = self.MAEs(a,b)
        
        h = plt.figure(figsize=(12,8))
        plt.plot(time_means[::self.filter_class.sample_rate])
        plt.axhline(y=0,color="r")
        plt.title("MAE over time")
            
        """find agent with highest MAE and plot it.
        mainly done to check something odd isnt happening"""
        if len(agent_means)>1:
            index = np.where(agent_means == np.nanmax(agent_means))[0][0]
            print(index)
            a1 = a[:,(2*index):(2*index)+2]
            b1 = b[:,(2*index):(2*index)+2]
            
            i = plt.figure(figsize=(12,8))
            plt.plot(a1[:,0],a1[:,1],label= "True Path")
            plt.plot(b1[::self.filter_class.sample_rate,0],b1[::self.filter_class.sample_rate,1],label = "KF Prediction")
            plt.legend()
            plt.title("Worst agent")
            
        j = plt.figure(figsize=(12,8))
        plt.hist(agent_means)
        plt.title("Mean Error per agent histogram")
                  
        if save:
            if observed:
                s = "observed"
            else:
                s = "unobserved"
            f.savefig(f"{s}_actual")
            g.savefig(f"{s}_kf")
            h.savefig(f"{s}_mae")
            if len(agent_means)>1:
                i.savefig(f"{s}_worst")
            j.savefig(f"{s}_agent_hist")
        return c,time_means
    
        
    def pair_frames(self,a,b):
        "paired side by side preds/truth"
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)
        
        os.mkdir("output_pairs")
        for i in range(a.shape[0]):
            a_s = [a_o[i,:],a_u[i,:]]
            b_s = [b_o[i,:], b_u[i,:]]
            f = plt.figure(figsize=(12,8))
            ax = plt.subplot(111)
            plt.xlim([0,width])
            plt.ylim([0,height])
            
            "plot true agents and dummies for legend"
            ax.scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o")
            ax.scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o")
            ax.scatter(-1,-1,color="orangered",label = "KF_Observed",marker="o")
            ax.scatter(-1,-1,color="yellow",label = "KF_Unobserved",marker="^")

            markers = ["o","^"]
            colours = ["orangered","yellow"]
            for j in range(len(a_s)):

                a = a_s[j]
                b = b_s[j]
                if np.abs(np.nansum(a-b))>1e-4: #check for perfect conditions (initial)
                    for k in range(int(a.shape[0]/2)):
                        a2 = a[(2*k):(2*k)+2]
                        b2 = b[(2*k):(2*k)+2]          
                        if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                            x = [a2[0],b2[0]]
                            y = [a2[1],b2[1]]
                            ax.plot(x,y,color="white")
                            ax.scatter(b2[0],b2[1],color=colours[j],marker = markers[j])
            
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            plt.xlabel("corridor width")
            plt.ylabel("corridor height")
            plt.title("True Positions vs UKF Predictions")
            number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
            file = f"output_pairs/pairs{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_pairs",f"pairwise_gif_{filter_class.pop_total}")

    def pair_frames_stack(self,a,b):
        "paired side by side preds/truth"
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)#uobs
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)#obs
        c,agent_means,time_means = self.MAEs(a_o,b_o) #maes
        c2,agent_means2,time_means2 = self.MAEs(a_u,b_u) #maes

        os.mkdir("output_pairs")
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
            axes[0].scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o")
            axes[0].scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o")
            axes[0].scatter(-1,-1,color="orangered",label = "KF_Observed",marker="o")
            axes[0].scatter(-1,-1,color="yellow",label = "KF_Unobserved",marker="^")

            
            markers = ["o","^"]
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
                            axes[0].plot(x,y,color="white")
                            axes[0].scatter(b2[0],b2[1],color=colours[j],marker = markers[j])
            
            #box = axes[1].get_position()
            #axes[1].set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            
            axes[1].plot(time_means[:i],label="observed")
            axes[2].plot(time_means2[:i],label="unobserved")

            axes[0].legend(bbox_to_anchor=(1.012, 1.2),
                      ncol=3,prop={'size':7})
            axes[0].set_xlabel("corridor width")
            axes[0].set_ylabel("corridor height")
            axes[1].set_ylabel("Observed MAE")
            axes[1].set_xlabel("Time (steps)")
            axes[2].set_ylabel("Unobserved MAE")
            axes[2].set_xlabel("Time (steps)")
            #axes[0].title("True Positions vs UKF Predictions")
            number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
            axes[0].text(0,1.05*height,"Frame Number: "+str(i))
            
            file = f"output_pairs/pairs{number}"
            f.tight_layout()
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_pairs",f"pairwise_gif_{filter_class.pop_total}")
        
    def pair_frames_stack_ellipse(self,a,b):

        "paired side by side preds/truth"
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        a_u,b_u,plot_range = self.plot_data_parser(a,b,False)#uobs
        a_o,b_o,plot_range = self.plot_data_parser(a,b,True)#obs
        c,agent_means,time_means = self.MAEs(a_o,b_o) #maes
        c2,agent_means2,time_means2 = self.MAEs(a_u,b_u) #maes

        os.mkdir("output_pairs")
        for i in range(a.shape[0]):
            a_s = [a_o[i,:],a_u[i,:]]
            b_s = [b_o[i,:], b_u[i,:]]
            f=plt.figure(figsize=(12,12))
            gs = gridspec.GridSpec(4,4)
            axes = [plt.subplot(gs[:2,:]),plt.subplot(gs[2:,:2]),plt.subplot(gs[2:,2:])]
            
             
            
            "plot true agents and dummies for legend"
            
            P = self.filter_class.ukf.Ps[i]
            agent_covs = []
            for j in range(int(a.shape[1]/2)):
                agent_covs.append(P[(2*j):(2*j)+2,(2*j):(2*j)+2])
            #axes[0].scatter(a_s[0][0::2],a_s[0][1::2],color="skyblue",label = "Truth",marker = "o")
            #axes[0].scatter(a_s[1][0::2],a_s[1][1::2],color="skyblue",marker = "o")
            
            "placeholders for a consistent legend. make sure theyre outside the domain of plotting"
            axes[0].scatter(-1,-1,color="skyblue",label = "Truth",marker = "o")
            axes[0].scatter(-1,-1,color="orangered",label = "KF_Observed",marker="o")
            axes[0].scatter(-1,-1,color="yellow",label = "KF_Unobserved",marker="^")

            
            markers = ["o","^"]
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
                            axes[0].plot(x,y,color="white")
                            axes[0].scatter(b2[0],b2[1],color=colours[j],marker = markers[j])
                            plot_covariance((x[0],y[0]),agent_covs[k],ax=axes[0],edgecolor="skyblue",alpha=0.6)
            #box = axes[1].get_position()
            #axes[1].set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            
            axes[2].set_xlim([0,a.shape[0]])
            axes[2].plot(time_means2[:i],label="unobserved")
            axes[2].set_xlim([0,a.shape[0]])
            axes[2].set_ylim([0,np.nanmax(time_means2)*1.05])  
            axes[2].set_ylabel("Unobserved MAE")
            axes[2].set_xlabel("Time (steps)")


            axes[1].set_xlim([0,a_u.shape[0]])
            axes[1].set_ylim([0,np.nanmax(time_means)*1.05])
            axes[1].set_ylabel("Observed MAE")
            axes[1].set_xlabel("Time (steps)")
            axes[1].plot(time_means[:i],label="observed")
            

            axes[0].legend(bbox_to_anchor=(1.012, 1.2),ncol=3,prop={'size':14})
            axes[0].set_xlim([0,width])
            axes[0].set_ylim([0,height])
            axes[0].set_xlabel("corridor width")
            axes[0].set_ylabel("corridor height")
            

            #axes[0].title("True Positions vs UKF Predictions")
            
            number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
            axes[0].text(0,1.05*height,"Frame Number: "+str(i))

            file = f"output_pairs/pairs{number}"
            
            f.tight_layout()
            f.savefig(file,bbox_inches="tight")
            plt.close()
        
        animations.animate(self,"output_pairs",f"pairwise_gif_{filter_class.pop_total}")
                

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
    "filterpy.stats covariance ellipse plot function with added parameter for custom axis"
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
    def animate(self,file,name):
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}GIF.mp4', images,fps=24)
        rmtree(file)
        #animations.clear_output_folder(self,file)
        
#%%
if __name__ == "__main__":
    np.random.seed(seed = 8)
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
			'pop_total': 10,

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
    do_batch - do batch processing on some pre-recorded truth data.
    """
    
    filter_params = {      
           
            "Sensor_Noise":  1, 
            "Process_Noise": 1, 
            'sample_rate': 1,
            "do_restrict": True, 
            "do_animate": False,
            "do_wiggle_animate": False,
            "do_density_animate":True,
            "do_pair_animate":False,
            "prop": 0.2,
            "heatmap_rate": 1,
            "bin_size":10,
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
            "d_rate" : 10, #data assimilotion rate every "d_rate model steps recalibrate UKF positions with truth

            }
    
    """run and extract data"""
    base_model = Model(**model_params)
    u = ukf_ss(model_params,filter_params,ukf_params,base_model)
    u.main()
    actual,preds= u.data_parser(True)
            
    """plots"""
    plts = plots(u)


    plot_save=False
    if filter_params["prop"]<1:
        distances,t_mean = plts.diagnostic_plots(actual,preds,False,plot_save)
    distances2,t_mean2 = plts.diagnostic_plots(actual,preds,True,plot_save)
    
    #plts.trajectories(actual)
    #plts.pair_frames(actual,preds)
    #plts.heatmap(actual)
