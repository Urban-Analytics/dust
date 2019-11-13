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

import sys #for print suppression#
#sys.path.append('../../stationsim')
sys.path.append('..')
from stationsim.stationsim_model import Model
import numpy as np
from math import floor,ceil,log10
import datetime
import multiprocessing
from copy import deepcopy
import os #for animations folder handling
import pickle

"plotting"
import matplotlib.pyplot as plt
from seaborn import kdeplot
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

"for polygon (square or otherwise) use"
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep

"import other useful things from other ukf file"
from ukf import animations


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
class agg_ukf:
    
    def __init__(self,ukf_params,init_x,poly_list,fx,hx,P,Q,R):
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

        self.poly_list = poly_list
        
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
        "need to do this with quadratic form at some point"
        
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

class agg_ukf_ss:
    """
    UKF for station sim using ukf filter class.
    """
    def __init__(self,model_params,filter_params,ukf_params,poly_list,base_model):
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
        self.poly_list = poly_list
        
        """
        calculate how many agents are observed and take a random sample of that
        many agents to be observed throughout the model
        """
        self.pop_total = self.model_params["pop_total"] #number of agents
        #number of batch iterations
        self.number_of_iterations = model_params['step_limit']
        self.sample_rate = self.filter_params["sample_rate"]
        #how many agents being observed
        
        self.sample_size= floor(self.pop_total*self.filter_params["prop"])
        
            
        #random sample of agents to be observed
        self.index = np.sort(np.random.choice(self.model_params["pop_total"],
                                                     self.sample_size,replace=False))
        self.index2 = np.empty((2*self.index.shape[0]),dtype=int)
        self.index2[0::2] = 2*self.index
        self.index2[1::2] = (2*self.index)+1
        
        self.ukf_histories = []
        self.agg_ukf_preds=[]
        self.full_Ps = []
        self.truths = []
        self.obs = []
        
        self.time1 =  datetime.datetime.now()#timer
        self.time2 = 0
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
        
        In this case this is used to convert our vector of latent agent positions
        into a vector of counts of agent in each polygon in poly list
        in:
            full latent state space output by fx
        out: 
            vector of aggregates from measuring how many agents in each polygon in poly_list 
        """
        counts = self.poly_count(self.poly_list,state)
        
        return counts
    
    def poly_count(self,poly_list,points):
        """
        counts how many agents in each polygon
        
        in: 
            1D vector of points from agents2state()/get_state(sensor=location),
        out: 
            counts of agents in each polygon in poly_list
        """
        counts = []
        points = np.array([points[::2],points[1::2]]).T
        points =MultiPoint(points)
        for poly in self.poly_list:
            poly = prep(poly)
            counts.append(int(len(list(filter(poly.contains,points)))))
        return counts
    
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
        R = np.eye(len(self.poly_list))#sensor noise
        P = np.eye(self.pop_total*2)#inital guess at state covariance
        self.ukf = agg_ukf(ukf_params,x,self.poly_list,self.fx,self.hx,P,Q,R)
    
    
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
        for _ in range(self.number_of_iterations):
            #if _%100 ==0: #progress bar
            #    print(f"iterations: {_}")
                

            #f_name = f"temp_pickle_model_ukf_{self.time1}"
            #f = open(f_name,"wb")
            #pickle.dump(self.base_model,f)
            #f.close()
            
            
            self.ukf.predict() #predict where agents will jump
            self.base_model.step() #jump stationsim agents forwards
            self.truths.append(self.base_model.get_state(sensor="location"))
         

            if _%self.sample_rate == 0: #update kalman filter assimilate predictions/measurements
                if self.filter_params["bring_noise"]:
                    noise_array=np.ones(self.pop_total*2)
                    noise_array[np.repeat([agent.status!=1 for agent in self.base_model.agents],2)]=0 #no noise for finished agents
                    noise_array*=np.random.normal(0,self.filter_params["noise"],self.pop_total*2)
                    
               
                state = self.poly_count(self.poly_list,
                                        self.base_model.get_state(sensor="location")+noise_array) #observed agents states

                self.ukf.update(z=state) #update UKF
                self.ukf_histories.append(self.ukf.x) #append histories
                self.agg_ukf_preds.append(self.ukf.x)
                self.full_Ps.append(self.ukf.P)       
                status = np.repeat(np.array([float(agent.status) for agent in self.base_model.agents]),2)
                status[status!=1] = np.nan 
                self.obs.append((self.base_model.get_state(sensor="location")+noise_array)*status)
                x = self.ukf.x
                if np.sum(np.isnan(x))==x.shape[0]:
                    print("math error. try larger values of alpha else check fx and hx. Could also be random rounding errors.")
                    break
                ""
            else:
                "update full preds that arent assimilated"
                self.agg_ukf_preds.append(self.ukf.x)
                self.full_Ps.append(self.ukf.P)
                
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
            a - noisy observations of agents positions
            b - ukf predictions of said agent positions
            c - if sampling rate >1 fills inbetween predictions with pure stationsim prediciton
                this is solely for smoother animations later
            d- true agent positions
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
            c= np.vstack(self.agg_ukf_preds)

        
            
            "all agent observations"
        
            return a,b,c,d,nan_array
        else:
            return a,b,d,nan_array
        
def grid_poly(width,length,bin_size):
    """
    generates grid of aggregate square polygons for corridor in station sim.
    UKF should work with any list of connected simple polys whose union 
    lies within space of interest.
    This is just an example poly that is nice but in theory works for any.
    !!potentially add randomly generated polygons or camera circles/cones.
    
    in:
        corridor parameters and size of squares
    out: 
        grid of squares for aggregates
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
       



class DivergingNorm(col.Normalize):
    def __init__(self, vleft,vright,vlc,vrc, vmin=None, vmax=None):
        """
        Normalize data with a set center.Rebuilt to be left heavy with the colouration
        given a skewed data set.

        Useful when mapping data where most points are tightly clustered and the rest 
        very sparsely spread. The large spread of sparse points dilutes the 
        colouration in the tight area resulting in heavy information loss.
        This Norm 
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
            Two floats tat indicate how many colours of the colormap colouration
            are within the tight band as a percentage.
            If these numbers are 0 and 1 all 256 colours are in the tight band
            If these numbers are 0.1 and 0,2 then the 
            25th to the 51st colours of the colormap represent the tight band.
            
        
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


class agg_plots:
    """
    class for all plots using in UKF
    """
    def __init__(self,filter_class,save_dir):
        "define which class to plot from"
        self.filter_class=filter_class
        self.save_dir = save_dir

    def trajectories(self,a,poly_list):
        "provide density of agents positions as a 2.5d histogram"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir(self.save_dir+"output_positions")
        for i in range(a.shape[0]):
            locs = a[i,:]
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            if np.abs(np.nansum(locs))>0:
                ax.scatter(locs[0::2],locs[1::2],color="k",label="True Positions",edgecolor="k")
                ax.set_ylim(0,height)
                ax.set_xlim(0,width)
            else:

                fake_locs = np.array([-1,-1])
                ax.scatter(fake_locs[0::2],fake_locs[1::2],color="k",label="True Positions",edgecolor="k")
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)   
            
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
            file = self.save_dir+ f"output_positions/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir+"output_positions",self.save_dir+f"positions_{filter_class.pop_total}_",12)
            
        
    def heatmap(self,a,poly_list):
        "provide density of agents positions as a heatmap"
        "!! add poly list not working yet"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        filter_class = self.filter_class
        bin_size = filter_class.filter_params["bin_size"]
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir(self.save_dir+"output_heatmap")
        
        
        "cmap set up. defining bottom value (0) to be black"
        cmap = cm.cividis
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0.0,0.0,0.0,1.0)
        cmap = col.LinearSegmentedColormap("custom_cmap",cmaplist,N=cmap.N)
        cmap = cmap.from_list("custom",cmaplist)
        "split norm for better vis"
        n = self.filter_class.model_params["pop_total"]
        norm =DivergingNorm(1,n/3,0.1,0.9,1e-8,n)

        sm = cm.ScalarMappable(norm = norm,cmap=cmap)
        sm.set_array([])  
        for i in range(a.shape[0]):
            locs = a[i,:]
            
             
            counts = self.filter_class.poly_count(poly_list,locs)
            #if np.nansum(counts)!=0:
            #    densities = np.array(counts)/np.nansum(counts) #density
            #else:
            #    densities = np.array(counts)
            #counts[np.where(counts==0)]=np.nan
            #norm =col.DivergingNorm(0.2)
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",size="5%",pad=0.05)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            #ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)
            
            
            
            #column = frame["counts"].astype(float)
            #im = frame.plot(column=column,
            #                ax=ax,cmap=cmap,norm=norm,vmin=0,vmax = n)
       
            patches = []
            for item in poly_list:
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
                ax.annotate(s=count, xy=poly_list[k].centroid.coords[0], 
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
            number = str(i).zfill(ceil(log10(a.shape[0])))
            file = self.save_dir+f"output_heatmap/{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir+"output_heatmap",self.save_dir+f"heatmap_{filter_class.pop_total}_",12)
        
    def L2s(self,a,b):
        """
        L2 distance error metric. 
        finds mean L2 (euclidean) distance at each time step and per each agent
        provides whole array of distances per agent and time
        and L2s per agent and time. 
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
    
    def agg_diagnostic_plots(self,a,b,save):
            """
            self - UKf class for various information
            
            a-observed agents
            
            b-UKF predictions of a
            
            observed- bool for plotting observed or unobserved agents
            if True observed else unobserved
            
            save- bool for saving plots in current directory. saves if true
            
            
            """
            
                
            #sample_rate =self.filter_class.filter_params["sample_rate"]
            
            #f=plt.figure(figsize=(12,8))
            #for j in range(int(a.shape[1]/2)):
            #    plt.plot(a[:,(2*j)],a[:,(2*j)+1],lw=3)  
            #    plt.xlim([0,self.filter_class.model_params["width"]])
            #    plt.ylim([0,self.filter_class.model_params["height"]])
            #    plt.xlabel("Corridor Width")
            #    plt.ylabel("Corridor Height")
            #    plt.title(f"Agent True Positions")
    
            #g = plt.figure(figsize=(12,8))
            #for j in range(int(a.shape[1]/2)):
            #    plt.plot(b[::sample_rate,2*j],b[::sample_rate,(2*j)+1],lw=3) 
            #    plt.xlim([0,self.filter_class.model_params["width"]])
            #    plt.ylim([0,self.filter_class.model_params["height"]])
            #    plt.xlabel("Corridor Width")
            #    plt.ylabel("Corridor Height")
            #    plt.title(f"Aggregate KF Predictions")
            
          
            c,c_index,agent_means,time_means = self.L2s(a,b)
            
           # h = plt.figure(figsize=(12,8))
           #time_means[np.isnan(time_means)]
           # plt.plot(c_index,time_means,lw=5,color="k",label="Mean Agent L2")
           # for i in range(c.shape[1]):
           #     plt.plot(c_index,c[:,i],linestyle="-.",lw=3)
           #     
           # plt.axhline(y=0,color="k",ls="--",alpha=0.5)
           # plt.xlabel("Time (steps)")
           # plt.ylabel("L2 Error")
           # plt.legend()
           # """find agent with highest L2 and plot it.
           # mainly done to check something odd isnt happening"""
           # 
           # index = np.where(agent_means == np.nanmax(agent_means))[0][0]
           # print(index)
           # a1 = a[:,(2*index):(2*index)+2]
           # b1 = b[:,(2*index):(2*index)+2]
           # 
           # i = plt.figure(figsize=(12,8))
           # plt.plot(a1[:,0],a1[:,1],label= "True Path",lw=3)
           # plt.plot(b1[::self.filter_class.sample_rate,0],
           #          b1[::self.filter_class.sample_rate,1],label = "KF Prediction",lw=3)
           # plt.legend()
           # plt.xlim([0,self.filter_class.model_params["width"]])
           # plt.ylim([0,self.filter_class.model_params["height"]])
           # plt.xlabel("Corridor Width")
           # plt.ylabel("Corridor Height")
           #plt.title("worst agent")
    
            j = plt.figure(figsize=(12,8))
            plt.hist(agent_means,density=False,
                     bins = self.filter_class.model_params["pop_total"],edgecolor="k")
            plt.xlabel("Agent L2")
            plt.ylabel(f" {self.filter_class.sample_size} Aggregated Agent Counts")
           # kdeplot(agent_means,color="red",cut=0,lw=4)
  
            if save:
                #f.savefig(self.save_dir+f"Aggregate_obs.pdf")
                #g.savefig(self.save_dir+f"Aggregate_kf.pdf")
                #h.savefig(self.save_dir+f"Aggregate_l2.pdf")
                #i.savefig(self.save_dir+f"Aggregate_worst.pdf")
                j.savefig(self.save_dir+f"Aggregate_agent_hist.pdf")
                
            return c,time_means
    
    def pair_frames(self,a,b):
        "pairwise animation"
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        
        os.mkdir(self.save_dir+"output_pairs")
        for i in range(a.shape[0]):
            
            f = plt.figure(figsize=(12,8))
            ax = plt.subplot(111)
            plt.xlim([0,width])
            plt.ylim([0,height])
            
            "plot true agents and dummies for legend"
            ax.scatter(a[i,0::2],a[i,1::2],color="skyblue",
                       label = "Truth",marker = "o",edgecolors="k")
            ax.scatter(a[i,0::2],a[i,1::2],color="skyblue",
                       marker = "o",edgecolors="k")
            ax.scatter(-1,-1,color="orangered",label = "Aggregate UKF Predictions",
                       marker="P",edgecolors="k")


            for k in range(int(a.shape[1]/2)):
                a2 = a[i,(2*k):(2*k)+2]
                b2 = b[i,(2*k):(2*k)+2]          
                if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                    x = [a2[0],b2[0]]
                    y = [a2[1],b2[1]]
                    ax.plot(x,y,color="k")
                    ax.scatter(b2[0],b2[1],color="orangered",marker = "P",edgecolors="k")
    
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            plt.xlabel("corridor width")
            plt.ylabel("corridor height")
            number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
            file = self.save_dir+f"output_pairs/pairs{number}"
            f.savefig(file)
            plt.close()
        
        animations.animate(self,self.save_dir+"output_pairs",self.save_dir+f"aggregate_pairwise_{self.filter_class.pop_total}",24)

    def pair_frames_single(self,a,b,frame_number):
        filter_class = self.filter_class
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        
        i=frame_number
        
        f = plt.figure(figsize=(12,8))
        ax = plt.subplot(111)
        plt.xlim([0,width])
        plt.ylim([0,height])
        
        "plot true agents and dummies for legend"
        ax.scatter(a[i,0::2],a[i,1::2],color="skyblue",
                   label = "Truth",marker = "o",edgecolors="k")
        ax.scatter(a[i,0::2],a[i,1::2],color="skyblue",
                   marker = "o",edgecolors="k")
        ax.scatter(-1,-1,color="orangered",label = "Aggregate UKF Predictions",
                   marker="P",edgecolors="k")


        for k in range(int(a.shape[1]/2)):
            a2 = a[i,(2*k):(2*k)+2]
            b2 = b[i,(2*k):(2*k)+2]          
            if not np.isnan(np.sum(a2+b2)): #check for finished agents that appear NaN
                x = [a2[0],b2[0]]
                y = [a2[1],b2[1]]
                ax.plot(x,y,color="k")
                ax.scatter(b2[0],b2[1],color="orangered",marker = "P",edgecolors="k")

        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])
        
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                  ncol=2)
        plt.xlabel("corridor width")
        plt.ylabel("corridor height")
        number =  str(i).zfill(ceil(log10(a.shape[0]))) #zfill names files such that sort() does its job properly later
        file = self.save_dir+f"agg_pairs{number}"
        f.savefig(file)
            
    def heatmap_single(self,a,poly_list,frame_number):
        
        

        "provide density of agents positions as a heatmap"
        "!! add poly list not working yet"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        filter_class = self.filter_class
        bin_size = filter_class.filter_params["bin_size"]
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        n = filter_class.model_params["pop_total"]
        i =frame_number
        
        
        "cmap set up. defining bottom value (0) to be black"
        cmap = cm.cividis #colour scheme
        cmaplist = [cmap(_) for _ in range(cmap.N)] #creating colourmap using cividis with black bad values (black background)
        cmaplist[0] = (0.0,0.0,0.0,1.0)
        cmap = col.LinearSegmentedColormap("custom_cmap",cmaplist,N=cmap.N)
        cmap = cmap.from_list("custom",cmaplist)
        """
        How the colourbar is spread we have 
        """
        norm =DivergingNorm(1,n/5,0.1,0.9,1e-8,n)
        "10% to 90% quantiles of the colour bar are between 1 and n/5 (e.g. 6 for n =30). colouration is between 1e-8 (not quite 0) and n"

        locs = a[i,:]
        
         
        counts = self.filter_class.poly_count(poly_list,locs)
        #if np.nansum(counts)!=0:
        #    densities = np.array(counts)/np.nansum(counts) #density
        #else:
        #    densities = np.array(counts)
                
        f = plt.figure(figsize=(12,8))
        ax = f.add_subplot(111)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.05)
        "plot density histogram and locations scatter plot assuming at least one agent available"
        #ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
        ax.set_ylim(0,height)
        ax.set_xlim(0,width)
        
        
        patches = []
        for item in poly_list:
           patches.append(mpatches.Polygon(np.array(item.exterior),closed=True))
         
        collection = PatchCollection(patches,cmap=cmap, norm=norm, alpha=1.0)
        ax.add_collection(collection)
        "if no agents in model for some reason just give a black frame"
        if np.nansum(counts)!=0:
            collection.set_array(np.array(counts))
        for k,count in enumerate(counts):
            plt.plot
            ax.annotate(s=count, xy=poly_list[k].centroid.coords[0], 
                        ha='center',va="center",color="w")
    
    
       
        
        "set up cbar. colouration proportional to number of agents"
        ax.text(0,101,s="Total Agents: " + str(np.sum(counts)),color="k")
        
        sm = cm.ScalarMappable(norm = norm,cmap=cmap)
        sm.set_array([])  
        cbar = f.colorbar(mappable=sm,cax=cax,spacing="proportional")

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
        number = str(i).zfill(ceil(log10(a.shape[0])))
        file = self.save_dir+f"heatmap_{number}"
        f.savefig(file)


#%%
if __name__ == "__main__":
    np.random.seed(seed = 8) #seeded example hash if want random
    # this seed (8) is a good example of an agent getting stuck for 10 agents
    
    recall = True #recall previous run
    do_pickle = True #pickle new run
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
    			'pop_total': 30,
    
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
        """
        
        filter_params = {      
               
                "Sensor_Noise":  1, 
                "Process_Noise": 1, 
                'sample_rate': 5,
                "do_restrict": True, 
                "prop": 1,
                "bin_size":25,
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
                "a":30,
                "b":2,
                "k":0,
                }
        
        
        
        """run and extract data"""
        base_model = Model(**model_params)
        poly_list = grid_poly(model_params["width"],model_params["height"],filter_params["bin_size"]) #generic square grid over corridor
        u = agg_ukf_ss(model_params,filter_params,ukf_params,poly_list,base_model)
        u.main()
        
        n = model_params["pop_total"]
        bin_size = filter_params["bin_size"]
        noise = filter_params["noise"]
        if do_pickle:
            f_name = f"test_agg_ukf_pickle_{n}_{bin_size}_{noise}"
            f = open("../experiments/ukf_experiments/"+f_name,"wb")
            pickle.dump(u,f)
            f.close()
        
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

    plot_save = True
    
    obs,preds,full_preds,truth,nan_array= u.data_parser(True)
    truth[~nan_array]=np.nan
    preds[~nan_array]=np.nan
    full_preds[~nan_array]=np.nan

    "additional step for aggregate"
    preds[np.isnan(obs)]=np.nan 
    full_preds[np.isnan(obs)]=np.nan 

    """plots"""
    agg_plts = agg_plots(u,"")
    
    distances,t_mean = agg_plts.agg_diagnostic_plots(truth,preds,plot_save)
    
    frame_number = 100 #which frame to take snapshot of
    if filter_params["sample_rate"]>1:
        agg_plts.pair_frames_single(truth,full_preds,frame_number)
    else:
        agg_plts.pair_frames_single(truth,preds,frame_number)
        
    agg_plts.heatmap_single(obs,poly_list,100)
    animate = False
    if animate:
        if filter_params["sample_rate"]>1:
            agg_plts.pair_frames(truth,full_preds)
        else:
            agg_plts.pair_frames(truth,preds)

        agg_plts.heatmap(obs[::u.sample_rate,:],poly_list)
        agg_plts.trajectories(obs[::u.sample_rate,:],poly_list)
