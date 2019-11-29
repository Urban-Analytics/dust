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
import numpy as np
"floor/ceil used for a lot of things. log10 keeps file names nice "
from math import floor, log10, ceil  
"general timer"
import datetime 
"used to calculate sigma points in parallel."
import multiprocessing  
"used in fx to restore stepped model"
from copy import deepcopy  
"used for a lot of things"
import os 
import sys 
"used to save clss instances when run finished."
import pickle 

"import stationsim model"
sys.path.append("..")
from stationsim.stationsim_model import Model

"for plots"
#from seaborn import kdeplot  # will be back shortly when diagnostic plots are better
"general plotting"
import matplotlib.pyplot as plt 
import matplotlib.lines as lines
"for nested plots in matplotlib pair_frames_stack"
import matplotlib.gridspec as gridspec 
"for heatmap plots"
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
"for rendering animations"
import imageio 
from shutil import rmtree

"for polygon (square or otherwise) use in aggregates"
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep

"used for plotting covariance ellipses for each agent. not really used anymore"
# from filterpy.stats import covariance_ellipse  
# from scipy.stats import norm #easy l2 norming  
# from math import cos, sin

plt.rcParams.update({'font.size':20})  # make plot font bigger

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
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
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


def L2s(truth,preds):
        
    
    """L2 distance errors between measurements and ukf predictions
    
    finds mean L2 (euclidean) distance at each time step and per each agent
    provides whole array of distances per agent and time
    and L2s per agent and time. 
    
    Parameters
    ------
    truth, preds: array_like
        `truth` true positions and `preds` ukf arrays to compare
        
    Returns
    ------
    
    distances : array_like
        `distances` matrix of  L2 distances between a and b over time and agents.
    """
    "placeholder"
    distances = np.ones((truth.shape[0],int(truth.shape[1]/2)))*np.nan

    "loop over each agent"
    "!!theres probably a better way to do this with apply_along_axis etc."
    for i in range(int(truth.shape[1]/2)):
            "pull one agents xy coords"
            truth2 = truth[:,(2*i):((2*i)+2)]
            preds2 = preds[:,(2*i):((2*i)+2)]
            res = truth2-preds2
            "loop over xy coords to get L2 value for ith agent at jth time"
            for j in range(res.shape[0]):
                distances[j,i]=np.linalg.norm(res[j,:]) 
                
    return distances

class ukf_plots:
    
    
    """class for all plots used in aggregate UKF
    
    
    !! big list of plots
    Parameters
    ------
    filter_class : class
        `filter_class` some finished ABM with UKF fitted to it 
    
    save_dir: string
        directory to save plots to. If using current directory "".
        e.g into ukf_experiments directory from stationsim "../experiments/ukf_experiments"
    """
    
    def __init__(self,filter_class,save_dir):
        "define which class to plot from"
        self.filter_class=filter_class
        self.width = filter_class.model_params["width"]
        self.height = filter_class.model_params["height"]
        "where to save any plots"
        
        self.obs_key = np.vstack(self.filter_class.obs_key)
        "circle, filled plus, filled triangle, and filled square"
        self.markers = ["o", "P", "^", "s"]
        "nice little colour scheme that works for all colour blindness"
        self.colours = ["black", "orangered", "yellow", "skyblue"]
        
        self.save_dir = save_dir
                        
    def trajectories(self,truth):
        
        
        """GPS style animation
        
        Parameters
        ------ 
        truth : array_like
            `truth` true positions 

        """
        os.mkdir(self.save_dir+"output_positions")
        for i in range(truth.shape[0]):
            locs = truth[i,:]
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            if np.abs(np.nansum(locs))>0:
                ax.scatter(locs[0::2],locs[1::2],color="k",label="True Positions",edgecolor="k",s=100)
                ax.set_ylim(0,self.height)
                ax.set_xlim(0,self.width)
            else:
                fake_locs = np.array([-10,-10])
                ax.scatter(fake_locs[0],fake_locs[1],color="k",label="True Positions",edgecolor="k",s=100)
            
            "set boundaries"
            ax.set_ylim(0,self.height)
            ax.set_xlim(0,self.width)   
            
            "set up cbar. colouration proportional to number of agents"
            #ticks = np.array([0.001,0.01,0.025,0.05,0.075,0.1,0.5,1.0])
           
               
            "set legend to bottom centre outside of plot"
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            "labels"
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            plt.title("Agent Positions")
            """
            frame number and saving. padded zeroes to keep frames in order.
            padded to nearest upper order of 10 of number of iterations.
            """
            number = str(i).zfill(ceil(log10(truth.shape[0])))
            file = self.save_dir+ f"output_positions/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir+"output_positions",
                           self.save_dir+f"positions_{self.filter_class.pop_total}_",12)
    
    def pair_frames_main(self, truth, preds, obs_key,plot_range,save_dir):
        
        
        """main pair wise frame plot
        """
        for i in plot_range:
            "extract rows of tables"
            truth2 = truth[i,:]
            preds2 = preds[i,:]
            obs_key2 = self.obs_key[i//self.filter_class.ukf_params["sample_rate"],:]
            
            f = plt.figure(figsize=(12,8))
            ax = plt.subplot(111)
            plt.xlim([0,self.width])
            plt.ylim([0,self.height])
            
            "plot true agents and dummies for legend"
            
            ax.scatter(truth2[0::2], truth2[1::2], color=self.colours[0], marker = self.markers[0])
            for j in range(self.filter_class.pop_total):
                    obs_key3 = int(obs_key2[j]+1)
                    colour = self.colours[obs_key3]
                    marker = self.markers[obs_key3]
                    ax.scatter(preds2[(2*j)],preds2[(2*j)+1],color=colour,marker = marker,edgecolors="k")
                    x = np.array([truth2[(2*j)],preds2[(2*j)]])
                    y = np.array([truth2[(2*j)+1],preds2[(2*j)+1]])
                    plt.plot(x,y,linewidth=3,color="k",linestyle="-")
                    plt.plot(x,y,linewidth=1,color="w",linestyle="-")
    
                    
            "dummy markers for consistent legend" 
            ax.scatter(-1,-1,color=self.colours[0],label = "Truth",marker=self.markers[0],edgecolors="k")
            ax.scatter(-1,-1,color=self.colours[1],label = "Unobserved",marker=self.markers[1],edgecolors="k")
            ax.scatter(-1,-1,color=self.colours[2],label = "Aggregate",marker=self.markers[2],edgecolors="k")
            ax.scatter(-1,-1,color=self.colours[3],label = "GPS",marker=self.markers[3],edgecolors="k")
            
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
            number =  str(i).zfill(ceil(log10(truth.shape[0]))) #zfill names files such that sort() does its job properly later
            file = save_dir +f"/ukf_pairs{number}"
            f.savefig(file)
            plt.close()
    
    def pair_frames_animation(self, truth, preds, plot_range):
        
        
        """ pairwise animation of ukf predictions and true measurements over ABM run
        
        Parameters
        ------
        truth,preds : array_like
            `a` measurements and `b` ukf estimates
        
        plot_range : list
            `plot_range` range of frames to plot
            
        """
        os.mkdir(self.save_dir +"output_pairs")
        
        save_dir = self.save_dir+ "output_pairs"
        self.pair_frames_main(truth,preds,obs_key,plot_range,save_dir)
        animations.animate(self,self.save_dir +"output_pairs",
                            self.save_dir +f"pairwise_gif_{self.filter_class.pop_total}",24)
        
    
    def pair_frame(self, truth, preds, frame_number):
        
        
        """single frame version of above
        
        Parameters
        ------
        truth,preds : array_like
            `a` measurements and `b` ukf estimates
        
        frame_number : int
            `frame_number` frame to plot
            
        save_dir : str 
            `save_dir` where to plot to
        """
        self.pair_frames_main(truth,preds,obs_key,[frame_number],self.save_dir)

        
        
    def path_plots(self, truth, preds, save):
        
        
        """plot paths taken by agents and their ukf predictions
        """
        f=plt.figure(figsize=(12,8))
        for i in range(self.filter_class.pop_total):
            plt.plot(truth[::self.filter_class.sample_rate,(2*i)],
                           truth[::self.filter_class.sample_rate,(2*i)+1],lw=3)  
            plt.xlim([0,self.filter_class.model_params["width"]])
            plt.ylim([0,self.filter_class.model_params["height"]])
            plt.xlabel("Corridor Width")
            plt.ylabel("Corridor Height")
            plt.title("True Positions")
            
        g = plt.figure(figsize=(12,8))
        for j in range(self.filter_class.pop_total):
            plt.plot(preds[::self.filter_class.sample_rate,2*j],
                     preds[::self.filter_class.sample_rate,(2*j)+1],lw=3) 
            plt.xlim([0,self.width])
            plt.ylim([0,self.height])
            plt.xlabel("Corridor Width")
            plt.ylabel("Corridor Height")
            plt.title("KF Predictions")
        
        if save:
            f.savefig("True_Paths.pdf")
            g.savefig("UKF_Paths.pdf")
            
        
    def error_hist(self, save):
        
        
        """Plot distribution of median agent errors
        """
        
        distances = L2s(truth,preds)
        agent_means = np.nanmedian(distances,axis=0)
        j = plt.figure(figsize=(12,8))
        plt.hist(agent_means,density=False,
                 bins = self.filter_class.model_params["pop_total"],edgecolor="k")
        plt.xlabel("Agent L2")
        plt.ylabel("Agent Counts")
        # kdeplot(agent_means,color="red",cut=0,lw=4)

        if save:
            j.savefig(self.save_dir+f"Aggregate_agent_hist.pdf")
    
    def heatmap_main(self, truth, ukf_params, plot_range, save_dir):
        """main heatmap plot
        
        """        
        "cmap set up. defining bottom value (0) to be black"
        cmap = cm.cividis
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0.0,0.0,0.0,1.0)
        cmap = col.LinearSegmentedColormap("custom_cmap",cmaplist,N=cmap.N)
        cmap = cmap.from_list("custom",cmaplist)
        "bottom heavy norm for better vis variable on size"
        n = self.filter_class.model_params["pop_total"]
        """this function is basically make it linear for low pops and squeeze 
        colouration lower for higher. Google it youll see what I mean"""
        n_prop = n*(1-np.tanh(n/15))
        norm =CompressionNorm(1e-5,n_prop,0.1,0.9,1e-8,n)

        sm = cm.ScalarMappable(norm = norm,cmap=cmap)
        sm.set_array([])  
        
        for i in plot_range:
            locs = truth[i,:]
            counts = poly_count(ukf_params["poly_list"],locs)
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",size="5%",pad=0.05)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            #ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
            ax.set_ylim(0,self.height)
            ax.set_xlim(0,self.width)
            
            
            
            #column = frame["counts"].astype(float)
            #im = frame.plot(column=column,
            #                ax=ax,cmap=cmap,norm=norm,vmin=0,vmax = n)
       
            patches = []
            for item in ukf_params["poly_list"]:
               patches.append(mpatches.Polygon(np.array(item.exterior),closed=True))
             
            collection = PatchCollection(patches,cmap=cmap, norm=norm, alpha=1.0)
            ax.add_collection(collection)
            "if no agents in model for some reason just give a black frame"
            if np.nansum(counts)!=0:
                collection.set_array(np.array(counts))
            else:
                collection.set_array(np.zeros(np.array(counts).shape))
    
            for k,count in enumerate(counts):
                plt.plot
                ax.annotate(s=count, xy=ukf_params["poly_list"][k].centroid.coords[0], 
                            ha='center',va="center",color="w")
                
            "set up cbar. colouration proportional to number of agents"
            ax.text(0,101,s="Total Agents: " + str(np.sum(counts)),color="k")
            
            
            cbar = plt.colorbar(sm,cax=cax,spacing="proportional")
            cbar.set_label("Agent Counts")
            cbar.set_alpha(1)
            #cbar.draw_all()
            
            "set legend to bottom centre outside of plot"
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            "labels"
            ax.set_xlabel("Corridor width")
            ax.set_ylabel("Corridor height")
            #ax.set_title("Agent Densities vs True Positions")
            cbar.set_label(f"Agent Counts (out of {n})")
            """
            frame number and saving. padded zeroes to keep frames in order.
            padded to nearest upper order of 10 of number of iterations.
            """
            number = str(i).zfill(ceil(log10(truth.shape[0])))
            file = save_dir+f"heatmap_{number}"
            f.savefig(file)
            plt.close()
    
    def heatmap(self,truth, ukf_params, plot_range):
        """ Aggregate grid square agent density map animation
        
        Parameters
        ------ 
        a : array_like
            `a` noisy measurements 
        poly_list : list
            `poly_list` list of polygons to plot
        
        """
        os.mkdir(self.save_dir+"output_heatmap")
        self.heatmap_main(truth, ukf_params, range(plot_range), self.save_dir+"output_heatmap/")
        animations.animate(self,self.save_dir+"output_heatmap",
                           self.save_dir+f"heatmap_{self.filter_class.pop_total}_",12)
    
    def heatmap_frame(self, truth, ukf_params, frame_number):
        
        
        """single frame version of above
        
        Parameters
        ------
        truth : array_like
            `truth` true agent positions
        
        frame_number : int
            `frame_number` frame to plot

        """
        self.heatmap_main(truth, ukf_params, [frame_number], self.save_dir)
    
class CompressionNorm(col.Normalize):
    def __init__(self, vleft,vright,vlc,vrc, vmin=None, vmax=None):
        """RCs customised matplotlib diverging norm
        
        The original matplotlib version (DivergingNorm) allowed the user to split their 
        data about some middle point (e.g. 0) and a symmetric colourbar for symmetric plots. 
        This is a slight generalisation of that.
        
        It allows you change how the colour bar concentrates itself for skewed data. 
        Say your data is very bottom heavy and you want more precise colouring in the bottom 
        of your data range. For example, if your data was between 5 and 10 and 
        90% of it was <6. If we used parameters:
            
        vleft=5,vright=6,vlc=0,vrc=0.9,vmin=5,vmax=10
        
        Then the first 90% of the colour bar colours would put themselves between 
        5 and 6 and the remaining 10% would do 6-10. 
        This gives a bottom heavy colourbar that matches the data.
        
        This works for generally heavily skewed data and could probably 
        be generalised further but starts to get very very messy
        
        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
      
        vleft: float
            left limit to tight band
        vright : flaot
            right limit to tight band
            
        vlc/vrc: float between 0 and 1 
        
            value left/right colouration.
            Two floats that indicate how many colours of the  256 colormap colouration
            are within the vleft/vright band as a percentage.
            If these numbers are 0 and 1 all 256 colours are in the band
            If these numbers are 0.1 and 0,2 then the 
            25th to the 51st colours of the colormap represent the band.
            
        
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.
        """

        self.vleft = vleft
        self.vright = vright
        self.vmin = vmin
        self.vmax = vmax
        self.vlc=vlc
        self.vrc=vrc
        if vleft>vright:
            raise ValueError("vleft and vright must be in ascending order"
                             )
        if vright is not None and vmax is not None and vright >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vleft is not None and vmin is not None and vleft <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vleft:
            self.vmin = self.vleft
        if self.vmax < self.vright:
            self.vmax = self.vright


    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vleft and self.vright <= self.vmax:
            raise ValueError("vmin, vleft,vright, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vleft,self.vright, self.vmax],
                      [0, self.vlc,self.vrc, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result    
    
class animations():
    
    
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
    
"fucntions to bring general filter above int more bespoke use for experiments"
        
def fx(x):
    
    
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
    model = deepcopy(base_model)
    model.set_state(state = x,sensor="location")    
    with HiddenPrints():
        model.step() #step model with print suppression
    state = model.get_state(sensor="location")
    
    return state
#%%
def omission_index(n,sample_size):
    
    
    """randomly pick agents to omit 
    used in experiment 1 hx function
    
    Parameters 
    ------
    n,p : int
         population `n` and proportion `p` observed. need p in [0,1]
         
    Returns
    ------
    index,index2: array_like:
        `index` of which agents are observed and `index2` their correpsoding
        index for xy coordinates from the desired state vector.
    """
    index = np.sort(np.random.choice(n,sample_size,replace=False))
    index2 = np.repeat(2*index,2)
    index2[1::2] += 1
    return index, index2


def hx1(state, model_params, ukf_params):
    
    
    """Convert each sigma point from noisy gps positions into actual measurements
    
    -   omits pre-definied unobserved agents given by index/index2
    
    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted

    Returns
    ------
    obs_state : array_like
        `obs_state` actual observed state
    """
    obs_state = state[ukf_params["index2"]]
    
    return obs_state   

def omission_params(model_params, ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 1
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    n = model_params["pop_total"]
    ukf_params["prop"] = 0.5
    ukf_params["sample_size"]= floor(n * ukf_params["prop"])

    
    ukf_params["index"], ukf_params["index2"] = omission_index(n, ukf_params["sample_size"])
    
    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * ukf_params["sample_size"])#sensor noise
    
   
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx1
    
    def obs_key_func(state, model_params, ukf_params):
        """which agents are observed"""
        
        key = np.zeros(model_params["pop_total"])
        key[ukf_params["index"]] +=2
        return key
    
    ukf_params["obs_key_func"] = obs_key_func
        
    return ukf_params

#%%
def poly_count(poly_list,points):
    
    
    """ counts how many agents in each closed polygon of poly_list
    
    -    use shapely polygon functions to count agents in each polygon
    
    Parameters
    ------
    poly_list : list
        list of closed polygons over StationSim corridor `poly_list`
    points : array_like
        list of agent GPS positions to count
    
    Returns
    ------
    counts : array_like
        `counts` how many agents in each polygon
    """
    counts = []
    points = np.array([points[::2],points[1::2]]).T
    points =MultiPoint(points)
    for poly in poly_list:
        poly = prep(poly)
        counts.append(int(len(list(filter(poly.contains,points)))))
    return counts
    
def grid_poly(width,length,bin_size):
    
    
    """generates complete grid of tesselating square polygons covering corridor in station sim.
   
    Parameters
    -----
    width,length : float
        `width` and `length` of StationSim corridor. 
    
    bin_size : float
     size of grid squares. larger implies fewer squares `bin_size`
     
    Returns
    ------
    polys : list
        list of closed square polygons `polys`
    """
    polys = []
    for i in range(int(width/bin_size)):
        for j in range(int(length/bin_size)):
            bl = [x*bin_size for x in (i,j)]
            br = [x*bin_size for x in (i+1,j)]
            tl = [x*bin_size for x in (i,j+1)]
            tr = [x*bin_size for x in (i+1,j+1)]
            
            polys.append(Polygon((bl,br,tr,tl)))
    "hashed lines for plots to verify desired grid"
    #for poly in polys:
    #    plt.plot(*poly.exterior.xy)
    return polys

def aggregate_params(ukf_params):
    
    
    """update ukf_params with fx/hx and their parameters for experiment 2
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    
    n = model_params["pop_total"]
    
    ukf_params["bin_size"] = 50
    ukf_params["poly_list"] = grid_poly(model_params["width"],
              model_params["height"],ukf_params["bin_size"]) 
        
    ukf_params["p"] = np.eye(2*n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2*n)
    ukf_params["r"] = np.eye(len(ukf_params["poly_list"]))#sensor noise 
    
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx2
    
    def obs_key_func(state,model_params,ukf_params):
        """which agents are observed"""
        
        key = np.ones(model_params["pop_total"])
        
        return key
    
    ukf_params["obs_key_func"] = obs_key_func
        
    return ukf_params
    
def hx2(state,model_params,ukf_params):
        """Convert each sigma point from noisy gps positions into actual measurements
        
        -   uses function poly_count to count how many agents in each closed 
            polygon of poly_list
        -   converts perfect data from ABM into forecasted 
            observation data to be compared and assimilated 
            using actual observation data
        
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
        counts = poly_count(ukf_params["poly_list"],state)
        
        return counts
#%%
        
def hx3(state, model_params, ukf_params):


    """Convert each sigma point from noisy gps positions into actual measurements
    
    -   omits pre-definied unobserved agents given by index/index2
    
    Parameters
    ------
    state : array_like
        desired `state` n-dimensional sigmapoint to be converted
    
    Returns
    ------
    obs_state : array_like
        `obs_state` actual observed state
    """
    ukf_params["index"], ukf_params["index2"] = omission_index(model_params["pop_total"], ukf_params["sample_size"])

    obs_state = state[ukf_params["index2"]]
    
    return obs_state   
    
def lateral_params(model_params, ukf_params):
    
    
    """update ukf_params with fx/hx for lateral partial omission.
    I.E every agent has some observations over time
    (potential experiment 3)
    
    Parameters
    ------
    ukf_params : dict
        
    Returns
    ------
    ukf_params : dict
    """
    n = model_params["pop_total"]
    ukf_params["prop"] = 0.5
    ukf_params["sample_size"]= floor(n * ukf_params["prop"])

    ukf_params["p"] = np.eye(2 * n) #inital guess at state covariance
    ukf_params["q"] = np.eye(2 * n)
    ukf_params["r"] = np.eye(2 * ukf_params["sample_size"])#sensor noise
    
   
    ukf_params["fx"] = fx
    ukf_params["hx"] = hx3
    
    def obs_key_func(state,ukf_params):
        """which agents are observed"""
        
        key = np.zeros(model_params["pop_total"])
        key[ukf_params["index"]] +=2
        return key
    
    ukf_params["obs_key_func"] = obs_key_func
        
    return ukf_params
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
        #ukf_params = omission_params(model_params, ukf_params)
        "aggregate version"
        ukf_params = aggregate_params(ukf_params)
        "lateral omission"
        #ukf_params = lateral_params(model_params, ukf_params)
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
            
        plts.heatmap(truth,ukf_params,truth.shape[0])
        
        plts.pair_frame(truth,preds,50)
        plts.heatmap_frame(truth,ukf_params,50)
        plts.error_hist(False)
        plts.path_plots(truth,preds,False)
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



