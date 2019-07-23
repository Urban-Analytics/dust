import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as col
import imageio
from scipy.spatial import distance as dist
from math import floor
import os

"""
various plot scripts and their dependencies
"""
#!!need to sync to lower sampling rates for diff map and similar


"""
modified matplotlib diverging norms that are very bottom heavy look at like 77 
to see 90% of the colourbar is to the left of the user defined centre.
in the density plots this center is typically 0.1 
so 90% of the colouration is in the bottom 10%
!!maybe add a second arguement to vary the percentage
"""
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


"""
similar to above but double sided about a centrsl point.
See line 149 to see that the the central vcenter either side of 0
contains 70% of the colouration

!! maybe add 2 variables for the true center and custom percents
"""
    
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
        
