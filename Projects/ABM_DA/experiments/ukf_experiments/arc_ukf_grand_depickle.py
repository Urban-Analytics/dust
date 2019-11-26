"""
 produces a more generalised diagnostic over multiple runs using multiple
 numbers of agents for arc_ukf.py only. 
 This produces a chloropleth style map showing the grand mean error
 over both time and agents for various fixed numbers of agents 
 and proportions observed.


import data from arc with following in bash terminal
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
change to relevant directories
"""
import pickle
import sys
sys.path.append("../../stationsim")
sys.path.append("../..")

from stationsim.ukf import plots


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.patheffects as pe

import glob
import seaborn as sns
import pandas as pd

#plt.rcParams.update({'font.size':20})

#%%
#plt.style.use("dark_background")
  

def grand_depickle_ukf_agg_data_parser(instance):
    """PUll data from aggregate ukf class into nice arrays
    Parameters
    ------
    instance : class
    
    Returns
    ------
    b,c,d,nan_array : array_like
        `b` UKF predictions every sample rate time steps. All other elements nan
        `c` Full UKF predictions. Say we assimilate every 5 time points then 4 
            are just ABM forecasts and the 5th are assimilated values. Useful
            for animations
        `d` Full true agent positions for comparison
        `nan_array` which elements are nan good for accurate plots/error metrics`
    """
    
    sample_rate = instance.sample_rate
    
    nan_array = np.ones(shape=(max([len(agent.history_locations) for 
                                    agent in instance.base_model.agents]),
                                    2*instance.pop_total))*np.nan
    for i in range(instance.pop_total):
        agent = instance.base_model.agents[i]
        array = np.array(agent.history_locations)
        array[array==None] ==np.nan
        nan_array[:len(agent.history_locations),2*i:(2*i)+2] = array
    
    nan_array = ~np.isnan(nan_array)
    
    b2 = np.vstack(instance.ukf_histories)
    d = np.vstack(instance.truths)

    
    b= np.zeros((d.shape[0],b2.shape[1]))*np.nan
  
    for j in range(int(b.shape[0]//sample_rate)):
        b[j*sample_rate,:] = b2[j,:]
     
    if sample_rate>1:
        c= np.vstack(instance.agg_ukf_preds)

        return b,c,d,nan_array
    else:
        return b,d,nan_array    
    
def l2_parser(instance):
    """gets real and UKF predicted data. Measures L2 distances between them
    
    Parameters
    ------
    instance : class
    
    Returns
    ------
    distance_obs : array_like
        `distance_obs` numpy array of distance between agents true positions 
        and their respective UKF predictions

    """    
      
    if instance.filter_params["sample_rate"]==1:
            preds,truth,nan_array = grand_depickle_ukf_agg_data_parser(instance)
    else:
        preds,full_preds,truth,nan_array = grand_depickle_ukf_agg_data_parser(instance)
        full_preds[~nan_array]=np.nan #make empty values to prevent mean skewing in diagnostic plots

    truth[~nan_array]=np.nan #make empty values to prevent mean skewing in diagnostic plots
    preds[~nan_array]=np.nan #make empty values to prevent mean skewing in diagnostic plots
    plts = plots(instance,"ukf_results/")
    distances_obs,oindex,agent_means,t_mean_obs = plts.L2s(truth,preds)

    
    return distances_obs


def grand_L2_matrix(n,bin_size,source): 
    """produces grand median matrix for all 30 ABM runs for choropleth plot
    
    Parameters
    ------
    n,bin_size : float
        population `n` and square size `bin_size`
    
    source : string
        `source` file for data
        
    Returns
    ------
    L2 : array_like
        `L2` matrix of grand medians. each row is a pop each column is a bin size
    

    """    
    "empty frames"
    L2 = np.ones((len(n),len(bin_size)))*np.nan
    
    "cycle over pairs of number of agents and proportions. taking grand (mean of medians) L2 mean for each pair"
    for i,num in enumerate(n):
        
        files={}
        for j in bin_size: 
            files[j] = glob.glob(source + f"/agg_ukf_agents_{num}_bin_{j}-*")

        for k,_ in enumerate(files.keys()):
            L2_2=[]
            for file in files[_]:
                f = open(file,"rb")
                uagg = pickle.load(f)
                f.close()
                distances = l2_parser(uagg)#
                "grand agent means"
                "grand agent medians"
                L2_2.append(np.nanmedian(distances,axis=0))

            L2[i,k]=np.nanmean(np.hstack(L2_2))
            
    return L2
    
def grand_L2_plot(data,n,bin_size,save):
    """produces grand median matrix for all 30 ABM runs for choropleth plot
    
    Parameters
    ------
    data : array_like
        L2 `data` matrix from grand_L2_matrix
    n,bin_size : float
        population `n` and square size `bin_size`
    save : bool
        `save` plot?

    """    
    "rotate frame 90 degrees so population on x axis"
    data = np.rot90(data,k=1) 

    "initiate plot"
    f,ax=plt.subplots(figsize=(8,8))
    "colourmap"
    cmap = cm.viridis
    "set nans for 0 agents unobserved to white (not black because black text)"
    cmap.set_bad("white") 
    
    " mask needed to get bad white squares in imshow"
    data2 = np.ma.masked_where(np.isnan(data),data)
    "rotate again so imshow right way up (origin bottom left i.e. lower)"
    data2=np.flip(data2,axis=0) 
    im=ax.imshow(data2,interpolation="nearest",cmap=cmap,origin="lower")
    
    "labelling"
    ax.set_xticks(np.arange(len(n)))
    ax.set_yticks(np.arange(len(bin_size)))
    ax.set_xticklabels(n)
    ax.set_yticklabels(bin_size)
    ax.set_xticks(np.arange(-.5,len(n),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(bin_size),1),minor=True)
    ax.grid(which="minor",color="k",linestyle="-",linewidth=2)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Aggregate Grid Squre Size")
    #plt.title("Grand L2s Over Varying Agents and Percentage Observed")


    "text on top of squares for clarity"
    data = np.flip(data,axis=0)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j,i,str(data[i,j].round(2)),ha="center",va="center",color="w",
                     path_effects=[pe.Stroke(linewidth = 0.7,foreground='k')])
            
    "colourbar alignment and labelling"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",pad=0.05)
    cbar=plt.colorbar(im,cax,cax)
    cbar.set_label("Grand Mean L2 Error")
    
    "further labelling and saving"
    cbar.set_label("Aggregate Median L2s")
    ax.set_ylabel("Aggregate Grid Squre Width")
    if save:
        plt.savefig("Aggregate_Grand_L2s.pdf")

            
def boxplot_parser(n,bin_size):
    """similar to grand_L2_matrix but creats a pandas frame for sns.catplot to read
    
    .. deprecated:: 
        use median boxplots this is dumb
    
    Parameters
    ------
    n,bin_size : float
        population `n` and square size `bin_size`
    
    source : str
        `source` file for data
    Returns
    ------
    L2 : array_like
        `L2` matrix of grand medians. Produces data frame with columns for pop,
        bin_size, and median. This version gives a median for every AGENT rather
        than a grand median for each run (I.E 30x12xpopulations rows vs 30x12 rows)
    """    
    L2 = {}
    for i in n:
        files={}
        for j in bin_size:
            files={}
            for j in bin_size: 
                files[j] = glob.glob(source +f"/agg_ukf_agents_{i}_bin_{j}-*")
        "sub dictionary for each bin size"       
        L2[i] = {} 
        for _ in files.keys():
            L2_2=[]
            for file in files[_]:
                f = open(file,"rb")
                u = pickle.load(f)
                f.close()
                distances = l2_parser(u)#
                
                L2_2.append(np.nanmedian(np.nanmean(distances,axis=0)))

            L2[i][_] = np.hstack(L2_2)
          
    "stack dictionaries into dataframe with corresponding n and bin_size next to each agent error"
    sub_frames = []

    for i in n:
        for j in bin_size:
            L2s = L2[i][j]
            sub_frames.append(pd.DataFrame([[i]*len(L2s),[j]*len(L2s),L2s]).T)

    "stack into grand frames and label columns"
    frame = pd.concat(sub_frames)
    frame.columns = ["n","square width","L2 agent errors"]

    return frame

def boxplot_plots(n,bin_size,frame,separate,save):  
    """produces grand median boxplot for all 30 ABM runs for choropleth plot
    
    ..deprecated:: 
        use medians below
    Parameters
    ------
    frame : array_like
        L2 `data` matrix from grand_L2_matrix
    n,bin_size : float
        population `n` and square size `bin_size`
    save,seperate : bool
        `save` plot?
        `seperate` box plots by population or have one big catplot?

    """    
    "seperate the plots by pop"
    if separate:
        for i in n:
            f_name = f"Aggregate_boxplot_{i}.pdf"
            y_name = "L2 agent errors"
            n_subframe = frame.loc[frame["n"]==str(i)]

            f = plt.figure()
            sns.boxplot(x="square width",y=y_name,data=n_subframe)
            if save:
                f.savefig(f_name)
    else:
        "or one big catplot"
        f_name = f"Aggregate_boxplot.pdf"
        y_name = "L2 agent errors"

        f = plt.figure()
        sns.catplot(x="square width",y=y_name,col="n",kind="box", data=frame)
        plt.tight_layout()
        if save:
            plt.savefig(f_name)
    
def boxplot_medians(n,bin_size,source):
    """similar to grand_L2_matrix but creats a pandas frame for sns.catplot to read
    
   
    Parameters
    ------
    n,bin_size : float
        population `n` and square size `bin_size`
    
    Returns
    ------
    L2 : array_like
        `L2` matrix of grand medians. Produces data frame with columns for pop,
        bin_size, and median. This version gives a  grand (mean of) median 
        each run (I.E 30x12 rows total.). This is independent of population size
        and statistically makes a lot more sense vs the other boxplot.
    

    """    
    L2 = {}
    
    "cycle over pairs of number of agents and proportions. taking grand (mean of medians) L2 mean for each pair"
    for i,num in enumerate(n):
        
        files={}
        for j in bin_size: 
            files[j] = glob.glob(source+f"/agg_ukf_agents_{num}_bin_{j}-*")

        for k,_ in enumerate(files.keys()):
            L2_2=[]
            for file in files[_]:
                f = open(file,"rb")
                uagg = pickle.load(f)
                f.close()
                distances = l2_parser(uagg)#
                #"grand agent means"
                #L2_2.append(np.nanmean(distances,axis=0))
                "grand agent medians"
                L2_2.append(np.nanmedian(np.nanmean(distances,axis=0)))
        
            L2[num,_]=L2_2

    L2_frame = pd.DataFrame(columns =["n","square width","grand_median"])        
    
    for i in n:
        for j in bin_size:
            data = L2[i,j]
            ns = [i]*len(data)
            bins = [j]*len(data)
            L2_2_frame = pd.DataFrame(np.array([ns,bins,data]).T,columns =["n","square width","grand_median"])
            L2_frame = pd.concat([L2_frame,L2_2_frame],axis=0)
            
    return L2_frame

def median_boxplot(L2_frame):
    """produces grand median boxplot for all 30 ABM runs for choropleth plot
    
   
    Parameters
    ------
    frame : array_like
        L2 `data` matrix from grand_L2_matrix
    n,bin_size : float
        population `n` and square size `bin_size`
    save,seperate : bool
        `save` plot?
        `seperate` box plots by population or have one big catplot?

    """       
    f_name = f"Aggregate_grand_median_boxplot.pdf"
    y_name = "grand_median"
    f = plt.figure()
    sns.catplot(x="square width",y=y_name,col="n",kind="box", data=L2_frame)
    plt.tight_layout()
    if save:
        plt.savefig(f_name)
    

            
 #%%
if __name__ == "__main__":
    
    """
    plot1    : choropleth
    plot2    : all agents boxplot (deprecated)
    plot3    : grand median boxplot
    n        : list of populations
    bin_size :  list of grid square sizes
    save     : save plots?
    """
    plot1 = False
    plot2 = False
    plot3 = True
    n=[10,20,30]
    bin_size = [5,10,25,50]
    source = "media/rob/ROB1/ukf_results"
    
    save=True
    if plot1:
        L2 = grand_L2_matrix(n,bin_size, source)
        grand_L2_plot(L2,n,bin_size,save)
    if plot2:
        frame = boxplot_parser(n, bin_size, source)
        boxplot_plots(n,bin_size,frame,False,save)        

    if plot3:
        L2_frame = boxplot_medians(n, bin_size, source)
        median_boxplot(L2_frame)
