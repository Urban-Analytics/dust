#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 11:27:06 2019

@author: rob
"""
import os
import sys

import numpy as np
from math import ceil, log10

try:
    sys.path.append("..")
    from ukf_experiments.poly_functions import poly_count
except:
    sys.path.append("../experiments/ukf_experiments")
    from poly_functions import poly_count

    
"for plots"
#from seaborn import kdeplot  # will be back shortly when diagnostic plots are better
"general plotting"
import matplotlib.pyplot as plt 

"for heatmap plots"
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
"for rendering animations"
import imageio 
from shutil import rmtree


plt.rcParams.update({'font.size':20})  # make plot font bigger


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
    
    def __init__(self,filter_class,save_dir, prefix):
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
        self.prefix = prefix
                        
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
            file = save_dir + self.prefix + f"pairs{number}"
            f.savefig(file)
            plt.close()
    
    def pair_frames_animation(self, truth, preds, obs_key, plot_range):
        
        
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
        
    
    def pair_frame(self, truth, preds, obs_key, frame_number):
        
        
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
            
        
    def error_hist(self, truth, preds, save):
        
        
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
        """n_prop function basically makes linear colouration for low pops and 
        large bins but squeeze colouration into bottom percentage for higher pops/low bins.
        This is done to get better visuals for higher pops/bin size
        Google sech^2(x) or 1-tanh^2(x) youll see what I mean.
        Starts near 1 and slowly goes to 0.
        Used tanh identity as theres no sech function in numpy.
        There's probably a nice kernel I dont know of."""
        n_prop = n*(1-np.tanh(n/ukf_params["bin_size"])**2)
        norm =CompressionNorm(1e-5,0.9*n_prop,0.1,0.9,1e-8,n)

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
            collection = PatchCollection(patches,cmap=cmap, norm=norm, alpha=1.0, edgecolor="w")
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
            file = save_dir + self.prefix + f"heatmap_{number}"
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
        
        
    def plot_polygons(ukf_params):
        """little function to plot polygons of poly_list"""
        
        poly_list = ukf_params["poly_list"]
        f,ax = plt.subplots()
        
        for poly in poly_list:
            a = poly.boundary.coords.xy
            plt.plot(a[0],a[1],color='w')
    
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
        
        


    
