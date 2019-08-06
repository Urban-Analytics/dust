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
from ukf import plots,animations
import numpy as np
from math import floor
import matplotlib.pyplot as plt
import datetime
import multiprocessing
from copy import deepcopy
import os #for animations folder handling
from math import ceil,log10

import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon,MultiPoint
from shapely.prepared import prep
import geopandas as gpd

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
        counts = poly_count(poly_list,state)
        
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
                
                state =poly_count(poly_list,self.base_model.agents2state()) #observed agents states
                self.ukf.update(z=state) #update UKF
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
       
def poly_count(poly_list,points):
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
    for poly in poly_list:
        poly = prep(poly)
        counts.append(int(len(list(filter(poly.contains,points)))))
    return counts


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


    def trajectories(self,a,poly_list):
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
            
        
    def heatmap(self,a,poly_list):
        "provide density of agents positions as a 2.5d histogram"
        "!! add poly list"
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        filter_class = self.filter_class
        bin_size = filter_class.filter_params["bin_size"]
        width = filter_class.model_params["width"]
        height = filter_class.model_params["height"]
        os.mkdir("output_heatmap")
        
        cmap = cm.Spectral
        cmap.set_bad(color="black")

        for i in range(a.shape[0]):
            locs = a[i,:]
            
             
            counts = poly_count(poly_list,locs)
            counts = np.array(counts)/np.nansum(counts)
            #counts[np.where(counts==0)]=np.nan
            frame =gpd.GeoDataFrame([counts,poly_list]).T
            frame.columns= ["counts","geometry"]
            
            norm =col.DivergingNorm(0.2)
 
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",size="5%",pad=0.05)
            "plot density histogram and locations scatter plot assuming at least one agent available"
            if np.abs(np.nansum(locs))>0:
                #ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
                ax.set_ylim(0,height)
                ax.set_xlim(0,width)
                frame.plot(column="counts",ax=ax,cax=cax,cmap=cmap,
                                     norm=norm)
                
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
            cbar = plt.colorbar(sm,cax=cax)
            cbar.set_label("Agent Density (x100%)")
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
        a = a[::self.filter_class.sample_rate,:]
        b = b[::self.filter_class.sample_rate,:]


        #loop over each time per agent
        for i in range(a.shape[0]):
                a2 = np.array([a[i,0::2],a[i,1::2]]).T
                b2 = np.array([b[i,0::2],b[i,1::2]]).T
                res = a2-b2
                c[i,:]=np.apply_along_axis(np.linalg.norm,1,res)
                    
        agent_means = np.nanmean(c,axis=0)
        time_means = np.nanmean(c,axis=1)
        return c,agent_means,time_means


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
			'pop_total': 5,

			'width': 400,
			'height': 200,

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
            'sample_rate': 10,
            "do_restrict": True, 
            "do_animate": False,
            "do_wiggle_animate": False,
            "do_density_animate":True,
            "do_pair_animate":False,
            "prop": 1,
            "heatmap_rate": 1,
            "bin_size":25,
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
    poly_list = grid_poly(model_params["width"],model_params["height"],filter_params["bin_size"]) #generic square grid over corridor
    u = agg_ukf_ss(model_params,filter_params,ukf_params,poly_list,base_model)
    u.main()
    actual,preds= u.data_parser(True)
    "additional step for aggregate"
    preds[np.isnan(actual)]=np.nan 
    """plots"""
    plts = plots(u)

    if filter_params["prop"]<1:
        distances,t_mean = plts.diagnostic_plots(actual,preds,False,False)
    distances2,t_mean2 = plts.diagnostic_plots(actual,preds,True,False)
    
    #plts.trajectories(actual)
    #plts.pair_frames(actual,preds)
    #plts.heatmap(actual)
    
