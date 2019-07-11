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
    
    def init(self,ukf):
        self.ukf = ukf #!!not sure this is necessary but it works so it's not stupid
    

    def heatmap(self,a,b):
        #sample_agents = [self.base_model.agents[j] for j in self.index]
        #swap if restricting observed agents
        
        bins = self.filter_params["bin_size"]
        width = self.model_params["width"]
        height = self.model_params["height"]
        for j in range(a.shape[0]):
            locs = a[j,:]
            
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
    
            plt.scatter(locs[:,0],locs[:,1],color="cyan")
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)        
            hist,xb,yb = np.histogram2d(locs[:,0],locs[:,1],
                                        range = [[0,width],[0,height]],
                                        bins = [2*bins,bins],density=True)
            hist *= bins**2
            hist= hist.T
            hist = np.flip(hist,axis=0)
    
            extent = [0,width,0,height]
            plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                       ,cmap = cm.Spectral ,extent=extent
                       ,norm=DivergingNorm(vmin=1e-10,vcenter=0.1,vmax=1))
            
            ticks = np.array([0.001,0.1,0.2,0.5,1.0])
            cbar = plt.colorbar(fraction=0.046,pad=0.04,shrink=0.71,
                                ticks = ticks,spacing="proportional")
            plt.clim(0,1)
            cbar.set_alpha(1)
            cbar.draw_all()
                
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            cbar.set_label("Agent Density (x100%)") 
            number = str(self.frame_number).zfill(5)
            file = f"output/heatmap{number}"
            f.savefig(file)
            plt.close()
            self.frame_number+=1
        
    
    def wiggle_heatmap(self,a,b):
        """
        old dont use
        """     
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
        if observed:
                a = a[:,self.index2]
                if len(self.index2)<b.shape[1]:
                    b = b[:,self.index2]
                plot_range = self.model_params["pop_total"]*(self.filter_params["prop"])

        else:      
                mask = np.ones(a.shape[1])
                mask[self.index2]=False
                a = a[:,np.where(mask!=0)][:,0,:]
                b = b[:,np.where(mask!=0)][:,0,:]
                plot_range = self.model_params["pop_total"]*(1-self.filter_params["prop"])

            
        f=plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(a[:,(2*j)],a[:,(2*j)+1])    
            plt.title("True Positions")

        g = plt.figure(figsize=(12,8))
        for j in range(int(plot_range)):
            plt.plot(b[::self.sample_rate,2*j],b[::self.sample_rate,(2*j)+1])    
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
        plt.plot(time_means[::self.sample_rate])
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
            plt.plot(b1[::self.sample_rate,0],b1[::self.sample_rate,1],label = "KF Prediction")
            plt.legend()
            plt.title("Worst agent")
            
        j = plt.figure(figsize=(12,8))
        plt.hist(agent_means)
        plt.legend()
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
    
            
    def density_frames(self,a,b):
        "snapshots of densities"
        bins = self.filter_params["bin_size"]
        width = self.model_params["width"]
        height = self.model_params["height"]
        #generate full from observed
        first_time =True
        if first_time:
            for _ in range(a.shape[0]):
                hista,xb,yb = np.histogram2d(a[_,::2],a[_,1::2],
                                        range = [[0,width],[0,height]],
                                        bins = [2*bins,bins],density=True)
                hista *= bins**2
                hista= hista.T
                hista = np.flip(hista,axis=0)
                self.densities.append(hista)
        
        for _ in range(b.shape[0]):
            histb,xb,yb = np.histogram2d(b[_,0::2],b[_,1::2],
                                    range = [[0,width],[0,height]],
                                    bins = [2*bins,bins],density=True)
            histb *= bins**2
            histb= histb.T
            histb = np.flip(histb,axis=0)
            self.kf_densities.append(histb)
    
        for _ in range(len(self.densities)):
           self.diff_densities.append(np.abs(self.densities[_]-self.kf_densities[_]))
           
        c = np.dstack(self.diff_densities)
        
        for _ in range(1,c.shape[2]):
            f = plt.figure(figsize=(12,8))
            ax = f.add_subplot(111)
            bins = self.filter_params["bin_size"]
            width = self.model_params["width"]
            height = self.model_params["height"]
            #plot non-wigglers and set plot size
            ax.set_ylim(0,height)
            ax.set_xlim(0,width)

            cmap = cm.Spectral
            cmap.set_bad(color="black")
            #check for any wigglers and plot the 2dhist 
            if np.sum(c[:,:,_])!=0:
                hista = c[:,:,_]
                extent = [0,width,0,height]
                im=plt.imshow(np.ma.masked_where(hista==0,hista)
                           ,cmap = cmap,extent=extent,norm=DoubleDivergingNorm(vcenter=0.05))
                
            #if no wiggles plot a "ghost histogram" to maintain frame structure  
            else:
                #ghost histogram with one entry and (1,1)
                hist,xb,yb = np.histogram2d(np.array([-1]),np.array([-1]),
                                            range = [[0,width],[0,height]],
                                            bins = [bins,bins],density=True)   
               
                extent = [0,width,0,height]
                #plot ghost hist with no opacity (alpha=0) to make it invisible
                im=plt.imshow(np.ma.masked_where(hist==0,hist),interpolation="none"
                              ,cmap = cmap ,extent=extent,alpha=1
                              ,norm=DoubleDivergingNorm(vmin=-0.5,vcenter=0.05,vmax=0.5))
            #colourbar and various plot fluff
            ticks = np.array([-0.5,-0.2,-0.1,-0.05,0,0.05,0.1,0.2,0.5])
            #!! numbers adjusted by trial and error for 200x100 field. 
            #should probably generalise this and the bin structure at some point
            cbar = plt.colorbar(im,fraction=0.046,pad=0.04,shrink=0.71,
                                ticks = ticks,spacing="proportional")
            plt.clim(-0.5,0.5)
            cbar.set_alpha(1)
            cbar.draw_all()
            
            plt.xlabel("Corridor width")
            plt.ylabel("Corridor height")
            cbar.set_label("Wiggle Density (x100%)")
            
            number =  str(_).zfill(5)
            file = f"output_diff/wiggle{number}"
            f.savefig(file)
            plt.close()
       
        
    def pair_frames(self,a,b):
        "paired side by side preds/truth"

        a = a[::self.filter_params["sample_rate"],self.index2]

        for i in range(b.shape[0]):
            a2 = a[i,:]
            b2 = b[i,:]
            
            f = plt.figure(figsize=(12,8))
            ax = plt.subplot(111)
            plt.xlim([0,200])
            plt.ylim([0,100])
            
            
            ax.scatter(a2[0::2],a2[1::2],color="skyblue",label = "Truth")
            ax.scatter(-1,-1,color="orangered",label = "KF_Predictions")

            
            if np.nansum(a2-b2)>1e-4: #check for perfect conditions (initial)
            
                for j in range(int(b.shape[1]/2)):
                    a3 = a2[(2*j):(2*j)+2]
                    b3 = b2[(2*j):(2*j)+2]          
                    if not np.isnan(np.sum(a3+b3)): #check for finished agents that appear NaN
                        x = [a3[0],b3[0]]
                        y = [a3[1],b3[1]]
                        ax.plot(x,y,color="white")
                        ax.scatter(b3[0],b3[1],color="orangered")
           
            box = ax.get_position()
            ax.set_position([box.x0, box.y0 + box.height * 0.1,
                             box.width, box.height * 0.9])
            
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
                      ncol=2)
            plt.xlabel("corridor width")
            plt.ylabel("corridor height")
            number =  str(i).zfill(5) #zfill names files such that sort() does its job properly later
            file = f"output_pairs/pairs{number}"
            f.savefig(file)
            plt.close()
    
class animations:
    def animate(self,file,name):
        files = sorted(os.listdir(file))
        print('{} frames generated.'.format(len(files)))
        images = []
        for filename in files:
            images.append(imageio.imread(f'{file}/{filename}'))
        imageio.mimsave(f'{name}GIF.mp4', images,fps=10)
        
        animations.clear_output_folder(self,file)
        

    """clears animated frames after being animated"""
    def clear_output_folder(self,file_name):
       folder = file_name
       for the_file in os.listdir(folder):
           file_path = os.path.join(folder, the_file)
           try:
               if os.path.isfile(file_path):
                   os.unlink(file_path)
           except Exception as e:
               print(e)
    
    
    
    
