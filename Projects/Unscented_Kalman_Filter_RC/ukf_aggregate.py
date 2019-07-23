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

#import pip packages
import numpy as np
from math import floor,log10,ceil
import matplotlib.pyplot as plt
import datetime
import multiprocessing
import matplotlib.cm as cm
import matplotlib.colors as col
import imageio
from scipy.spatial import distance as dist
from copy import deepcopy
import os #for animations folder handling
from shutil import rmtree#as above
import sys #for print suppression
from stationsim_model import Model
#import local packages


#for dark plots. purely an aesthetic choice.
plt.style.use("dark_background")

"""
suppress repeat printing in F_x from new stationsim
E.g. 
with HiddenPrints():
    everything done here prints nothing

everything here prints again
https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python
"""

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
        """sigma point calculations based on current mean x and  UT (upper triangular) 
        decomposition S of covariance P"""
        
     
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
        Pxx =  self.wc[0]*np.outer((nl_sigmas[:,0].T-xhat),(nl_sigmas[:,0].T-xhat))+self.Q
        for i in range(1,len(self.wc)): 
            Pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)
            
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
        "need to do this with quadratic form at some point"
        Pyy =  self.wc[0]*np.outer((nl_sigmas[:,0].transpose()-yhat),(nl_sigmas[:,0].transpose()-yhat))+self.R
        for i in range(1,len(self.wc)):
            Pyy += self.wc[i]*np.outer((nl_sigmas[:,i].transpose()-yhat),(nl_sigmas[:,i].transpose()-yhat))
        

        Pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),(nl_sigmas[:,0].transpose()-yhat))
        for i in range(1,len(self.wc)):
            Pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),(nl_sigmas[:,i].transpose()-yhat))
            
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

    def fx(self,x,**fx_args):
        """
        Transition function for each agent. where it is predicted to be.
        For station sim this is essentially a placeholder step with
        varying initial locations depending on each sigma point.
        

        """
        """
        call placeholder base_model created in main via pickling
        this is done to essentially rollback the basemodel as stepping 
        it is non-invertible (afaik).
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
        Measurement function for agent. I.E. where the agents are recorded to be
        returns observed agents.
        in:
            full state space
        out: 
            observed subset of full state
        """
        state = state[self.index2]
        
        return state
    
    def init_ukf(self,ukf_params):
        "initialise ukf with initial state and covariance structures."
        x = self.base_model.get_state(sensor="location")#initial state
        Q = np.eye(self.pop_total*2)#process noise
        R = np.eye(len(self.index2))#sensor noise
        P = np.eye(self.pop_total*2)#inital guess at state covariance
        self.ukf = ukf(ukf_params,x,self.fx,self.hx,P,Q,R)
        self.ukf_histories.append(self.ukf.x) #
    
    
    def main(self):
        """
        main function for ukf station sim
        -initiates ukf
        -predict with ukf
        -step true model
        -update ukf with new model positions
        -repeat until model ends or max iterations reached
        
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
        
        time2 = datetime.datetime.now()#timer
        print(time2-self.time1)
        
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

class DivergingNorm(col.Normalize):
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.Rebuilt to be left heavy with the colouration
        given a skewed data set.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.

        
        
        """

        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter


    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin, self.vcenter, self.vmax],
                      [0, 0.9, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

class DoubleDivergingNorm(col.Normalize):
    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.Rebuilt to be left heavy with the colouration
        given a skewed data set.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the the max value of the dataset.

        
        
        """

        self.vcenter = vcenter
        self.vmin = vmin
        self.vmax = vmax
        if vcenter is not None and vmax is not None and vcenter >= vmax:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')
        if vcenter is not None and vmin is not None and vcenter <= vmin:
            raise ValueError('vmin, vcenter, and vmax must be in '
                             'ascending order')

    def autoscale_None(self, A):
        """
        Get vmin and vmax, and then clip at vcenter
        """
        super().autoscale_None(A)
        if self.vmin > self.vcenter:
            self.vmin = self.vcenter
        if self.vmax < self.vcenter:
            self.vmax = self.vcenter


    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The clip argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)  # sets self.vmin, self.vmax if None

        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError("vmin, vcenter, vmax must increase monotonically")
        result = np.ma.masked_array(
            np.interp(result, [self.vmin,-self.vcenter,0,self.vcenter, self.vmax],
                      [0, 0.15,0.5,0.85, 1.]), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result


class plots:
    """
    class for all plots using in UKF
    """
    def __init__(self,filter_class):
        self.filter_class=filter_class
        self.frame_number=0
        
    "filters data into observed/unobserved if necessary"
    def plot_data_parser(self,a,b,observed):
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

    def trajectories(self,a):
        "provide density of agents positions as a 2.5d histogram"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir("output_positions")
        for i in range(a.shape[0]):
            locs = a[i,:]
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            if np.abs(np.nansum(locs))>0:
                ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
                ax.set_ylim(0,height)
                ax.set_xlim(0,width)
            else:

                fake_locs = np.array([-1,-1])
                ax.scatter(fake_locs[0::2],fake_locs[1::2],color="cyan",label="True Positions")
               
            
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
            number = str(i).zfill(ceil(log10(a.shape[0])))
            file = f"output_positions/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_positions",f"positions_{filter_class.pop_total}_")
            
        
    def heatmap(self,a):
        "provide density of agents positions as a 2.5d histogram"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        filter_class = self.filter_class
        bin_size = filter_class.filter_params["bin_size"]
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir("output_heatmap")

        for i in range(a.shape[0]):
            locs = a[i,:]
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            if np.abs(np.nansum(locs))>0:
                ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
                ax.set_ylim(0,height)
                ax.set_xlim(0,width)        
                hist,xb,yb = np.histogram2d(locs[0::2],locs[1::2],
                                            range = [[0,width],[0,height]],
                                            bins = [int(width/bin_size),int(height/bin_size)],density=True)
                hist *= bin_size**2
                hist= hist.T
                hist = np.flip(hist,axis=0)
        
                extent = [0,width,0,height]
                plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                           ,cmap = cm.Spectral ,extent=extent
                           ,norm=DivergingNorm(vmin=1/filter_class.pop_total,
                                               vcenter=5/filter_class.pop_total,vmax=1))
            else:
                """
                dummy frame if no locations present e.g. at the start. 
                prevents divide by zero error in hist2d
                """
                fake_locs = np.array([-1,-1])
                ax.scatter(fake_locs[0::2],fake_locs[1::2],color="cyan",label="True Positions")
                hist,xb,yb = np.histogram2d(fake_locs[0::2],fake_locs[1::2],
                                            range = [[0,width],[0,height]],
                                            bins = [int(width/bin_size),int(height/bin_size)],density=True)
                hist *= bin_size**2
                hist= hist.T
                hist = np.flip(hist,axis=0)
        
                extent = [0,width,0,height]
                plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                           ,cmap = cm.Spectral ,extent=extent
                           ,norm=DivergingNorm(vmin=1/filter_class.pop_total,
                                               vcenter=5/filter_class.pop_total,vmax=1))
            
            "set up cbar. colouration proportional to number of agents"
            #ticks = np.array([0.001,0.01,0.025,0.05,0.075,0.1,0.5,1.0])
            cbar = plt.colorbar(fraction=0.046,pad=0.04,shrink=0.71,
                                spacing="proportional")
            cbar.set_label("Agent Density (x100%)")
            plt.clim(0,1)
            cbar.set_alpha(1)
            cbar.draw_all()
               
            "set legend to bottom centre outside of plot"
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            "labels"
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            plt.title("Agent Densities vs True Positions")
            cbar.set_label("Agent Density (x100%)")
            """
            frame number and saving. padded zeroes to keep frames in order.
            padded to nearest upper order of 10 of number of iterations.
            """
            number = str(i).zfill(ceil(log10(a.shape[0])))
            file = f"output_heatmap/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_heatmap",f"heatmap_{filter_class.pop_total}_")

    """
    old dont use
    !! probably not worth getting this to work post hoc with new stationsim
    """   
    def wiggle_heatmap(self,a,b):
          
        bins = self.filter_params["bin_size"]
        width = self.model_params["width"]
        height = self.model_params["height"]
        
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
                            ticks = ticks,spacing="proportional")
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
        
    
    def diagnostic_plots(self,a,b,observed,save):
        """
        self - UKf class for various information
        
        a-observed agents
        
        b-UKF predictions of a
        
        observed- bool for plotting observed or unobserved agents
        if True observed else unobserved
        
        save- bool for saving plots in current directory. saves if true
        
        
        """
        fil=self.filter_class
        a,b,plot_range = self.plot_data_parser(a,b,observed)
        
        f=plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(a[:,(2*j)],a[:,(2*j)+1])    
            plt.title("True Positions")

        g = plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(b[::fil.sample_rate,2*j],b[::fil.sample_rate,(2*j)+1])    
            plt.title("KF predictions")

            
        
            
        """MAE metric. 
        finds mean average euclidean error at each time step and per each agent"""
        c = np.ones((a.shape[0],int(a.shape[1]/2)))*np.nan
        
       
        
        for i in range(int(a.shape[1]/2)):
            a_2 =   a[:,(2*i):(2*i)+2] 
            b_2 =   b[:,(2*i):(2*i)+2] 
    

            for k in range(floor(np.min([a.shape[0],b.shape[0]]))):
                if not(np.any(np.isnan(a_2[k,:])) or np.any(np.isnan(b_2[k,:]))):                
                    c[k,i]=dist.euclidean(a_2[k,:],b_2[k,:])
                        
        agent_means = np.nanmean(c,axis=0)
        time_means = np.nanmean(c,axis=1)
        h = plt.figure(figsize=(12,8))
        plt.plot(time_means[::fil.sample_rate])
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
            plt.plot(b1[::fil.sample_rate,0],b1[::fil.sample_rate,1],label = "KF Prediction")
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
    
            
    def difference_frames(self,a,b):
        "snapshots of densities"
        filter_class = self.filter_class
        bin_size = filter_class.filter_params["bin_size"]
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir("output_diff")
        #generate full from observed
        densities=[]
        kf_densities =[]
        diffs = []
        for _ in range(1,a.shape[0]):
            hista,xb,yb = np.histogram2d(a[_,::2],a[_,1::2],
                                    range = [[0,width],[0,height]],
                                    bins = [int(width/bin_size),int(height/bin_size)],density=True)
            hista *= bin_size**2
            hista= hista.T
            hista = np.flip(hista,axis=0)
            densities.append(hista)
    
        for _ in range(1,b.shape[0]):
            histb,xb,yb = np.histogram2d(b[_,0::2],b[_,1::2],
                                    range = [[0,width],[0,height]],
                                    bins = [int(width/bin_size),int(height/bin_size)],density=True)
            histb *= bin_size**2
            histb= histb.T
            histb = np.flip(histb,axis=0)
            kf_densities.append(histb)
    
        for _ in range(len(densities)):
           diffs.append(np.abs(densities[_]-kf_densities[_]))
           
        for i,hist in enumerate(diffs):
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            
            if np.abs(np.nansum(hist))>0:
                  ax.set_ylim(0,height)
                  ax.set_xlim(0,width)        
                  
                  extent = [0,width,0,height]
                  plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                             ,cmap = cm.Spectral ,extent=extent
                             ,norm=DivergingNorm(vmin=1/filter_class.pop_total,
                                                 vcenter=5/filter_class.pop_total,vmax=1))
            else:
                """
                dummy frame if no locations present e.g. at the start. 
                e.g. perfect aggregates
                """
                fake_locs = np.array([-1,-1])
                ax.scatter(fake_locs[0::2],fake_locs[1::2],color="cyan",label="True Positions")
            
                extent = [0,width,0,height]
                plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                           ,cmap = cm.Spectral ,extent=extent
                           ,norm=DivergingNorm(vmin=1/filter_class.pop_total,
                                               vcenter=5/filter_class.pop_total,vmax=1))   
            

            #colourbar and various plot fluff
            ticks = np.arange(0,1.1,0.1)
            cbar = plt.colorbar(fraction=0.046,pad=0.04,shrink=0.71,
                                ticks = ticks,spacing="uniform")
            cbar.set_label("Agent Density (x100%)")
            plt.clim(0,1)
            cbar.set_alpha(1)
            cbar.draw_all()
               
            #"set legend to bottom centre outside of plot"
            #box = ax.get_position()
            #ax.set_position([box.x0, box.y0 + box.height * 0.1,
            #                 box.width, box.height * 0.9])
            # 
            # ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            #           ncol=2)
            "labels"
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            plt.title("Predicted vs Actual Densities")
            cbar.set_label("Agent Density (x100%)")
            """
            frame number and saving. padded zeroes to keep frames in order.
            padded to nearest upper order of 10 of number of iterations.
            """
            number = str(i).zfill(ceil(log10(a.shape[0])))
            file = f"output_diff/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_diff",f"diff_gif_{filter_class.pop_total}")
        
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
            number =  str(i).zfill(5) #zfill names files such that sort() does its job properly later
            file = f"output_pairs/pairs{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,"output_pairs",f"pairwise_gif_{filter_class.pop_total}")

class animations:
    def animate(self,file,name):
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}GIF.mp4', images,fps=24)
        rmtree(file)
        
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

    if filter_params["prop"]<1:
        distances,t_mean = plts.diagnostic_plots(actual,preds,False,False)
    distances2,t_mean2 = plts.diagnostic_plots(actual,preds,True,False)
    
    #plts.trajectories(actual)
    #plts.pair_frames(actual,preds)
    #plts.heatmap(actual)