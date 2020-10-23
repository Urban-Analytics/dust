#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File in which all single run ukf plots are generated.
"""

# from external modules
import os
import sys
import imageio 
from shutil import rmtree
import shutil    
import numpy as np
from math import ceil, log10
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
import matplotlib.colors as col
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

# from modules
sys.path.append("../..")
from modules.default_ukf_configs import marker_attributes as marker_attributes
from modules.poly_functions import poly_count


plt.rcParams.update({'font.size':20})  # make plot font bigger


def L2s(truth, preds):
    """L2 distance errors between measurements and ukf predictions
    
    Finds L2 (euclidean) distance between each agents truth and preds.
    Assembled into a matrix such that each column follows an agent 
    over time and each row looks at all agents for a given time 
    point.
        
    Parameters
    ------
    truth, preds: array_like
        `truth` true positions and `preds` ukf arrays to compare
    Returns
    ------
    distances : array_like
        matrix of L2 `distances` between truth and preds over time and agents.
    """
    
    # placeholder note half as many columns from collapsing 2d vectors into scalars
    distances = np.ones((truth.shape[0],int(truth.shape[1]/2)))*np.nan

    # loop over each agent
    # loop over each time point for each agent
    #!!theres probably a better way to do this with apply_along_axis etc.
    for i in range(int(truth.shape[1]/2)):
            #pull one agents xy coords
            truth2 = truth[:,(2*i):((2*i)+2)]
            preds2 = preds[:,(2*i):((2*i)+2)]
            #residual difference
            res = truth2-preds2
            #loop over each row to get distance at each time point for given agent
            for j in range(res.shape[0]):
                distances[j,i]=np.linalg.norm(res[j,:]) 
                
    return distances

class ukf_plots:
    
    def __init__(self, filter_class, destination, prefix, save, animate,
                 marker_attributes = marker_attributes):
        """class for all plots used in UKF experiments
        
        Parameters
        ------
        filter_class : class
            `filter_class` some finished ABM with UKF fitted to it 
        
        destination , prefix : str
            `destination` to save plots in and some `prefix` for the file name
            e.g. "../plots" and "ex1_"
            
        save, animate : bool
            `save` plots or `animate` who ABM run?
        
        marker_attributes : dict
            This dictionary `marker_attributes` determines the colour
            and the shape of a marker given its observation types. 
            Typically the truth are black circles, unobserved are orange
            crosses, aggregates are yellow triangles, and GPS observations
            are blue squares.
        """
        
        self.filter_class=filter_class
        self.width = filter_class.model_params["width"]
        self.height = filter_class.model_params["height"]
        
        # markers and colours for pairwise plots
        # determined in the experiment module
        self.markers = marker_attributes["markers"]
        self.colours = marker_attributes["colours"]
        self.labels = marker_attributes["labels"]
        
        self.destination = destination
        self.prefix = prefix
        self.save = save
    
    """animations"""
    
    def trajectories(self,truths, destination):
        """GPS style animation of how agent move
        
        - For each time point
            - plot each agents true xy positions over stationsim corridor
        
        Parameters
        ------ 
        truths : array_like
            `truth` true positions 
        
        """
        # move to folder to store animation frames
        os.mkdir(destination+"output_positions")
        # loop over time
        for i in range(truths.shape[0]):
            #take positions at current time i
            locs = truths[i,:]
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            # plot density histogram and locations scatter plot assuming 
            # at least one agent available
            if np.abs(np.nansum(locs))>0:
                # if there are agents in the model plot them
                ax.scatter(locs[0::2],locs[1::2],color="k",
                           label="True Positions",edgecolor="k",s=100)
                ax.set_ylim(0,self.height)
                ax.set_xlim(0,self.width)
            else:
                # if there are no agents in the model use a dummy plot off the 
                # boundary
                # dont set alpha = 0 here it messes with the legend
                # may need to move this fake point if the boundary changes
                fake_locs = np.array([-10,-10])
                ax.scatter(fake_locs[0],fake_locs[1],color="k",
                           label="True Positions",edgecolor="k",s=100)
            
            # set boundaries of stationsim
            ax.set_ylim(0,self.height)
            ax.set_xlim(0,self.width)   
               
            # set legend to bottom centre outside of plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            # labels
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            plt.title("Agent Positions")
            # frame number and saving. padded zeroes to keep frames in order.
            # number basically ensures the padding up to the nearest order of 10
            # if there are 100 frames there are 2 padded 0s 001.
            # if there are 1000 frames there are 3 0001.            
            number = str(i).zfill(ceil(log10(truths.shape[0])))
            # saver file and close figure
            file = destination + "output_positions/" + self.prefix + f"_{number}"
            f.savefig(file)
            plt.close()
        
        # stitch the frames together and save it as an mp4.
        animations.animate(self, destination + "output_positions/",
                           destination + self.prefix + f"positions_{self.filter_class.pop_total}",12)
    
    def pair_frames_main(self, truths, preds, obs_key, plot_range, destination):
        """Main pair wise frame plot
        
        - given some true values and predictions
        - for every time point i in plot_range
            - extract ith row of truth/preds
            - plot xy positions of each
            - tether respective xy positions together
            - save plot or show it if its just 1 frame
        
        Parameters
        ------ 
        truths, preds, obs_key  : array_like
            `truth` true positions, `preds` ukf predicted positions, 
            and `obs_key` types of observation for each agent 
        
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """
        # some parameters that are used a lot. simpified here for neatness.
        n = self.filter_class.model_params["pop_total"]
        sample_rate = self.filter_class.ukf_params["sample_rate"]
        
        # loop over time
        for i in plot_range:
            # if hold repeat every `sample_rate`th frame sample_rate times.
            # since the truths are at every time step and the preds aren't
            # holding can make the animations smoother.
            # this is deprecated just use the ::sample_rate index in numpy
            hold = False
            if hold:
                truths2 = truths[i - i%sample_rate, :]
                preds2 = preds[i - i%sample_rate, :]
                obs_key2 = obs_key[i - i%sample_rate, :]
            else:
                truths2 = truths[i, :]
                preds2 = preds[i, :]
                obs_key2 = obs_key[i, :]
            ms = 10 #marker_size
            alpha = 1 # plot opacity. not to be confused with ukf alpha
            
            f = plt.figure(figsize = (12,8))
            ax = plt.subplot(111)
            plt.xlim([0,self.width])
            plt.ylim([0,self.height])
            
            "plot true agents and dummies for legend"
            "one scatter for translucent fill. one for opaque edges"
            ax.scatter(truths2[0::2], truths2[1::2], color=self.colours[-1],
                       s= ms**2, marker = self.markers[-1],alpha=alpha)
            ax.scatter(truths2[0::2], truths2[1::2],
                       c="none", s= ms**2, marker = self.markers[-1],ec="k",
                       linewidths=1.5)

            #plot truth, prediction and tether for each agent
            for j in range(n):
                    #choose tether width and get observation type from obs_key2
                    tether_width = ms/5
                    key = obs_key2[j]       
                    #stop error being thrown when agent not observed
                    if np.isnan(obs_key2[j]):
                        continue
                    
                    # choose colours and shape from observation type
                    # and marker attribute dictionary
                    
                    colour = self.colours[key]
                    marker = self.markers[key]
                    
                    # two scatter plots for each set of markers here.
                    # first scatter plot makes translucent centre.
                    # second plot makes black outline.
                    
                    #scatters for predictions
                    
                    ax.scatter(preds2[(2*j)], preds2[(2*j)+1], c="none", 
                               marker = marker, s= ms**2, edgecolors="k",
                               linewidths=1.5)
                    ax.scatter(preds2[(2*j)],preds2[(2*j)+1], color=colour,
                               marker = marker, s= ms**2, alpha=alpha,
                               edgecolors="k")
                    
                    # pairing tethers between truth and prediction
                    # stack x and ys for truths and preds
        
                    x = np.array([truths2[(2*j)], preds2[(2*j)]])
                    y = np.array([truths2[(2*j)+1], preds2[(2*j)+1]])
                    
                    #plot lines between each truths and preds pair
                    plt.plot(x,y,linewidth=tether_width+2, color="k",
                             linestyle="-")
                    plt.plot(x,y,linewidth=tether_width, color="w",
                             linestyle="-")

                    
            # plotting list of polygons in which agents are observed for ex4
            # ignore anyone ever comes back to it.
            ukf_keys = self.filter_class.ukf_params.keys()
            if 'cameras' in ukf_keys:
                cameras = self.filter_class.ukf_params["cameras"]
                polygons = [camera.polygon for camera in cameras]
                self.plot_polygons(ax, polygons)
                
            
            # dummy markers for a consistent legend
            # make sure these are outside the stationsim boundary
            for key in self.colours.keys():
                ax.scatter(-1,-1,color=self.colours[key],label = self.labels[key], s= ms**2,
                           marker=self.markers[key],edgecolors="k",linewidths=1.5)
            
            #put legend outside of plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                      ncol=2)
            #labelling
            plt.xlabel("corridor width")
            plt.ylabel("corridor height")
            #plt.title("True Positions vs UKF Predictions")
            
            # save frame and close plot else struggle for RAM
            # see trajectories for more info on this
            number =  str(i).zfill(ceil(log10(truths.shape[0]))) #zfill names files such that sort() does its job properly later
            file = destination + self.prefix + f"pairs{number}"
            
            if len(plot_range) ==1:
                # show the plot if theres just one frame.
                plt.show()
                plt.close()
                        
            if self.save:
                # save the plot.
                plt.tight_layout()
                f.savefig(file)
                plt.close()
    
    def pair_frames(self, truths, forecasts, obs_key, plot_range, destination):
        """ pairwise animation of ukf predictions and true measurements over ABM run
        
        - using pair frames_main above plot an animation for the entire ABM run
        - create directory for frames
        - for every time point in truths plot a pairwise plot using
            pair_frame_main
        - animate frames from directory together and delete the frames.
        
        Parameters
        ------ 
        truths, preds, obs_key  : array_like
            `truth` true positions, `preds` ukf predicted positions, 
            and `obs_key` types of observation for each agent 
        
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """
        # make directory for frames to go in.
        save_dir = destination +"output_pairs/"
        try:
            # make the directory if it doesnt exist
            os.mkdir(save_dir)
        except:
            # else delete the current directory with this name and make another.
            shutil.rmtree(save_dir)
            os.mkdir(save_dir)
        
        # generate the frames to be animated and animate them.
        self.pair_frames_main(truths, forecasts, obs_key, range(plot_range), save_dir)
        animations.animate(self,save_dir, destination + self.prefix +
                           f"pairwise_gif_{self.filter_class.pop_total}", 6)
        
    
    def pair_frame(self, truths, forecasts, obs_key, frame_number, destination):
        """single frame version of above
        
        - plot truths for a single time point. save as a png in plots.
        - doesnt make an mp4.
       
        Parameters
        ------ 
        truths, preds, obs_key  : array_like
            `truth` true positions, `preds` ukf predicted positions, 
            and `obs_key` types of observation for each agent 
        
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """
        # just get a frame. dont animate it.
        self.pair_frames_main(truths ,forecasts, obs_key, [frame_number], 
                              destination)

        
        
    """aggregate heatmap for experiment 2"""
    
    def heatmap_main(self, truths, plot_range, destination):
        """main heatmap plot for aggregates.
           
        Parameters
        ------ 
        truths : array_like
            `truth` true positions
            
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """   
        # shorter name for params for neatness
        ukf_params = self.filter_class.ukf_params

        """Setting up custom colour map. defining bottom value (0) to be black
        and everything else is just cividis
        """
        # build colourmap and set bad value to black.
        cmap = cm.cividis
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmaplist[0] = (0.0,0.0,0.0,1.0)
        cmap = col.LinearSegmentedColormap("custom_cmap",cmaplist,N=cmap.N)
        cmap = cmap.from_list("custom",cmaplist)
        
        """
        TLDR: compression norm allows for more interesting colouration of heatmap
        
        For a large number of grid squares most of the squares only have 
        a few agents in them ~5. If we have 30 agents this means roughly 2/3rds 
        of the colourmap is not used and results in very boring looking graphs.
        
        In these cases I suggest we `move` the colourbar so more of it covers the bottom
        1/3rd and we get a more colour `efficient` graph.
        
        n_prop function makes colouration linear for low pops and large 
        square sizes but squeezes colouration into the bottom of the data range for
        higher pops/low bins. 
        
        This squeezing is proportional to the sech^2(x) = (1-tanh^2(x)) function
        for x>=0.
        (Used tanh identity as theres no sech function in numpy.)
        http://mathworld.wolfram.com/HyperbolicSecant.html
        
        It starts near 1 so 90% of the colouration initially covers 90% of 
        the data range (I.E. linear colouration). 
        As x gets larger sech^2(x) decays quickly to 0 so 90% of the colouration covers
        a smaller percentage of the data range.

        E.g if x = 1, n = 30, 30*0.9*sech^2(1) ~= 10 so 90% of the colouration would be 
        used for the bottom 10 agents and much more of the colour bar would be used.

        There's probably a nice kernel alternative to sech
        """
        # finish building cmap according to above
        n= self.filter_class.model_params["pop_total"]
        n_prop = n*(1-np.tanh(n/ukf_params["bin_size"])**2)
        norm =CompressionNorm(1e-15,0.9*n_prop,0.1,0.9,1e-16,n)

        sm = cm.ScalarMappable(norm = norm,cmap=cmap)
        sm.set_array([])  
        
        #  loop over time
        for i in plot_range:
            # get positions
            locs = truths[i,:]
            #count how many agents in each square
            counts = poly_count(ukf_params["poly_list"],locs)
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right",size="5%",pad=0.05)
            # plot density histogram and locations scatter plot 
            # assuming at least one agent available"
            # ax.scatter(locs[0::2],locs[1::2],color="cyan",label="True Positions")
            ax.set_ylim(0,self.height)
            ax.set_xlim(0,self.width)
       
            #plot individual squares with their count and colour
            patches = []
            for item in ukf_params["poly_list"]:
               patches.append(mpatches.Polygon(np.array(item.exterior),closed=True))
            collection = PatchCollection(patches,cmap=cmap, norm=norm, alpha=1.0, edgecolor="w")
            ax.add_collection(collection)

            # if no agents in model for some reason just give a black frame
            if np.nansum(counts)!=0:
                collection.set_array(np.array(counts))
            else:
                collection.set_array(np.zeros(np.array(counts).shape))
    
            for k,count in enumerate(counts):
                plt.plot
                ax.annotate(s=count, xy=ukf_params["poly_list"][k].centroid.coords[0], 
                            ha='center',va="center",color="w",
                                         size = ukf_params["bin_size"])
            
            # set up colourbar. colouration proportional to number of agents
            ax.text(0,101,s="Total Agents: " + str(np.sum(counts)),color="k")
            cbar = plt.colorbar(sm,cax=cax,spacing="proportional")
            cbar.set_label("Agent Counts")
            cbar.set_alpha(1)
            #cbar.draw_all()
            
            # legend
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            # labelling
            ax.set_xlabel("Corridor width")
            ax.set_ylabel("Corridor height")
            #ax.set_title("Agent Densities vs True Positions")
            cbar.set_label(f"Agent Counts (out of {n})")
            
            # frame number and saving frame
            # see trajectories for more details
            number = str(i).zfill(ceil(log10(truths.shape[0])))
            file = destination + self.prefix + f"heatmap_{number}"
            
            # if just one frame plot it
            if len(plot_range) ==1:
                plt.show()
             
            # else save it
            if self.save:
                f.savefig(file)
                plt.close()
    
    def heatmap(self,truths, plot_range, destination):
        """ Aggregate grid square agent density map animation
        
        Parameters
        ------ 
        truths : array_like
            `truth` true positions 
        
        plot_range : list
            `plot_range` what range of time points (indices) from truths to
            plot. 
        """
        
        # make directory
        save_dir = destination + "output_heatmap/"
        os.mkdir(save_dir)
        # generate frames and animate them
        self.heatmap_main(truths, range(plot_range), save_dir)
        animations.animate(self, save_dir, destination +
                           self.prefix  + f"heatmap_{self.filter_class.pop_total}_",12)
    
    def heatmap_frame(self, truths, frame_number, destination):
        """single frame version of above
        
        Parameters
        ------
        truth : array_like
            `truth` true agent positions
        
        frame_number : int
            `frame_number` frame to plot

        """
        # make single frame
        self.heatmap_main(truths, [frame_number], destination)

    def path_plots(self, data, title, polygons = None):
        """spaghetti style plot of some set of trajectories.
        
        data : array_like
            `data` some array of agent positions to plot
        
        title : str
            `title` of plot 
            e.g. `True` gives title `True Positions`
            
            
        polygons : list
            list of `polygons` to plot. good for looking at boundaries 
            where agents jump
        """
        f=plt.figure(figsize=(12,8))
        
        if polygons is not None:
            for poly in polygons:
                a = poly.boundary.coords.xy
                plt.plot(a[0],a[1],color='k', alpha = 0.5)
        
        # loop over agents.
        # get each agents xy and plot them
        for i in range(data.shape[1]//2):
            plt.plot(data[:,(2*i)],data[:,(2*i)+1],lw=3)  
            plt.xlim([0,self.filter_class.model_params["width"]])
            plt.ylim([0,self.filter_class.model_params["height"]])
            plt.xlabel("Corridor Width")
            plt.ylabel("Corridor Height")
            plt.title(f"{title} Positions")
            
        #save final figure
        if self.save:
            f.savefig(self.destination + f"{title}_Paths.pdf")
            
    def dual_path_plots(self, data, data2, title, polygons = None):
        """spaghetti style plot of 2 set of trajectories.
        
        data, data2 : array_like
            `data` and `data2` some array of agent positions to plot
        
        title : str
            `title` of plot 
            e.g. `True` gives title `True Positions`
            
            
        polygons : list
            list of `polygons` to plot. good for looking at boundaries 
            where agents jump
        """
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        f=plt.figure(figsize=(12,8))
        
        # plot boundary if not specified
        if polygons is not None:
            for poly in polygons:
                a = poly.boundary.coords.xy
                plt.plot(a[0],a[1],color='k', alpha = 0.5)
        
        # loop over agent
        # get xy for each data set
        # plot them both
        # data 2 will have a thicker, bolder, dashed line.
        for i in range(data.shape[1]//2):
            plt.plot(data[:,(2*i)],data[:,(2*i)+1],lw=8, alpha= 0.5, 
                     color = colours[i%len(colours)])  
            plt.plot(data2[:,(2*i)],data2[:,(2*i)+1],lw=4, linestyle = "--", 
                     alpha= 1, color = colours[i%len(colours)])  
            # limits and labels
            plt.xlim([0,self.filter_class.model_params["width"]])
            plt.ylim([0,self.filter_class.model_params["height"]])
            plt.xlabel("Corridor Width")
            plt.ylabel("Corridor Height")
            #plt.title(f"{title} Positions")
            
        #save
        if self.save:
            f.savefig(self.destination + f"{title}_Paths.pdf")
            
    def error_hist(self, truths, preds, title):
        """Plot distribution of median agent errors as a histogram
        
        
        Parameters
        ------ 
        truths, preds,  : array_like
            `truth` true positions and `preds` ukf predicted positions
        
        title : str
            `title` of plot 
            e.g. `True` gives title `True Positions`
        """
        
        # l2 distance between truths and predictions
        distances = L2s(truths,preds)
        # median l2 error for each agent
        agent_medians = np.nanmedian(distances,axis=0)
        j = plt.figure(figsize=(12,8))
        #histogram of median agent errors
        plt.hist(agent_medians, 
                 density=False,
                 bins = self.filter_class.model_params["pop_total"],
                 edgecolor="k")
        #labels and save
        plt.xlabel("Agent Median L2 Error")
        plt.ylabel("Agent Counts")
        plt.title(title) 
        # kdeplot(agent_means,color="red",cut=0,lw=4)

        if self.save:
            j.savefig(self.destination + f"{title}_Agent_Hist.pdf")        
            
    def plot_polygons(self, ax, poly_list):
        """little function to plot shapely polygons of poly_list
        
        
        Parameters
        ------
        ax : ax
            matplotlib axis `ax` to plot the polygons on
        
        poly_list : list
            `poly_list` list of polygons to plot on the axis.
        """
        
        for poly in poly_list:
            ax.fill(*poly.exterior.xy, color = "blue", alpha=0.5)
            #plt.plot(*poly.exterior.xy)


    def gate_choices(self, gates, sample_rate):
        f, ax = plt.subplots()
    
        for i in range(gates.shape[1]):
            plt.plot(np.arange(gates.shape[0])*sample_rate,  (0.02 * i) + gates[:, i], alpha = 0.7)
        plt.xlabel("Time")
        plt.ylabel("Gate Choice")
        ax.set_yticks(np.unique(gates)*1.02)
        #ax.set_yticks(np.arange(gates.shape[1] + 1))
        ax.set_yticklabels(np.unique(gates))
        
        plt.tight_layout()
        f.savefig("../../plots/gate_choices.pdf")
            
    def split_gate_choices(gates, sample_rate):
        
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        f, ax = plt.subplots(1, 3, sharey = True, gridspec_kw = {'wspace' : 0})
    
        for i in range(gates.shape[1]):
            ax[i].plot(np.arange(gates.shape[0])*sample_rate, gates[:, i],
                       color = colours[i%len(colours)], alpha = 0.7)
            ax[i].set_xlabel("Time")
        ax[0].set_ylabel("Gate Choice")
        ax[0].set_yticks(np.unique(gates))
        #ax.set_yticks(np.arange(gates.shape[1] + 1))
        ax[0].set_yticklabels(np.unique(gates))
        
        plt.tight_layout()
        f.savefig("../../plots/split_gate_choices.pdf")    
        
class CompressionNorm(col.Normalize):
    
    def __init__(self, vleft, vright, vlc, vrc, vmin=None, vmax=None):
        """Customised matplotlib diverging norm fir bottom heavy data.
        
        The original matplotlib version (DivergingNorm) allowed the user to split their 
        data about some middle point (e.g. 0) and a symmetric colourbar for symmetric plots. 
        This is a slight generalisation of that.
        
        It allows you change how the colour bar concentrates itself for skewed data. 
        Say your data is very bottom heavy and you want more precise colouring in the bottom 
        of your data range. For example, if your data was between 5 and 10 and 
        90% of it was <6. If we used parameters:
            
        vleft=5,vright=6,vlc=0,vrc=0.9,vmin=5,vmax=10
        
        Then the first 90% of the colour bar colours would put themselves between 
        5 and 6 and the remaining 10% between 6-10. 
        This gives a bottom heavy colourbar that matches the data.
        
        This works for heavily skewed data and could probably 
        be generalised further but starts to get messy
        
        Parameters
        ----------
        vleft: float
            left limit to tight band
        vright : flaot
            right limit to tight band
            
        vlc/vrc: float between 0 and 1 
        
            value left/right colouration.
            Two floats that indicate how many colours of the 256 colormap colours
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
    """class to animate frames recorded by several functions into actual mp4s"""

    
    def animate(self,file,name,fps):
        """ animate some list of frames into an MP4.
            
        Parameters
        ------
        file, name : str
            Which `file` of frames to animate.
            What `name` to give the mp4.
            
        fps : int
            frames per second `fps` of mp4. higher fps means faster smoother video

        Returns
        -------
        None.

        """
        
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}.mp4', images,fps=fps)
        rmtree(file)
        #animations.clear_output_folder(self,file)
        
        


    
