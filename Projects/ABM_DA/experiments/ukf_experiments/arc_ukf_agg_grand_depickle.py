"""
 produces a more generalised diagnostic over multiple runs using multiple
 numbers of agents for arc_ukf.py only. 
 This produces a chloropleth style map showing the grand mean error
 over both time and agents for various fixed numbers of agents 
 and proportions observed.

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
    "extract arrays of real paths, predicted paths, L2s between them from main class using parser above."
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


def grand_L2_matrix(n,bin_size): 
    "empty frames"
    L2 = np.ones((len(n),len(bin_size)))*np.nan
    
    "cycle over pairs of number of agents and proportions. taking grand (mean of medians) L2 mean for each pair"
    for i,num in enumerate(n):
        
        files={}
        for j in bin_size: 
            files[j] = glob.glob(f"ukf_results/agg_ukf_agents_{num}_bin_{j}-*")

        for k,_ in enumerate(files.keys()):
            L2_2=[]
            for file in files[_]:
                f = open(file,"rb")
                uagg = pickle.load(f)
                f.close()
                distances = l2_parser(uagg)#
                "grand agent means"
                L2_2.append(np.nanmean(distances,axis=0))
                "grand agent medians"
                L2_2.append(np.nanmedian(distances,axis=0))

            L2[i,k]=np.nanmean(np.hstack(L2_2))
            
    return L2
    
def grand_L2_plot(data,n,bin_size,observed,save):

    "rotate frame 90 degrees so origin in bottom left"
    data = np.rot90(data,k=1)
    
    "initiate plot"
    f,ax=plt.subplots(figsize=(8,8))
    "colourmap"
    cmap = cm.viridis
    cmap.set_bad("white") #set nans for unobserved full prop to white
    
    data2 = np.ma.masked_where(np.isnan(data),data) #needed to get bad white squares in imshow
    data2=np.flip(data2,axis=0) #rotate so imshow right way up (origin bottom left)
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


    "labelling squares"
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
    
    "further labelling and saving depending on observed/unobserved plot"
    cbar.set_label("Aggregate Median L2s")
    ax.set_ylabel("Aggregate Grid Squre Width")
    if save:
        plt.savefig("Aggregate_Grand_L2s.pdf")

            
def boxplot_parser(n,bin_size):
    L2 = {}
    for i in n:
        files={}
        for j in bin_size:
            files={}
            for j in bin_size: 
                files[j] = glob.glob(f"ukf_results/agg_ukf_agents_{i}_bin_{j}-*")
                
        L2[i] = {} #sub dictionary for bin_size
        for _ in files.keys():
            L2_2=[]
            for file in files[_]:
                f = open(file,"rb")
                u = pickle.load(f)
                f.close()
                distances = l2_parser(u)#
                
                L2_2.append(np.nanmean(distances,0))

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

            grand_L2_plot(L2,n,bin_size,True,save)

    return frame

def boxplot_plots(n,bin_size,frame,separate,save):  
    if separate:
        for i in n:
            if observed:
                f_name = f"Aggregate_boxplot_{i}.pdf"
                y_name = "L2 agent errors"
                n_subframe = frame.loc[frame["n"]==str(i)]
    
            f = plt.figure()
            sns.boxplot(x="square width",y=y_name,data=n_subframe)
            if save:
                f.savefig(f_name)
    
    else:
        f_name = f"Aggregate_boxplot.pdf"
        y_name = "L2 agent errors"

        f = plt.figure()
        sns.catplot(x="square width",y=y_name,col="n",kind="box", data=frame)
        plt.tight_layout()
        if save:
            plt.savefig(f_name)
    
def boxplot_medians(n,binsize):
    "empty frames"
    L2 = {}
    
    "cycle over pairs of number of agents and proportions. taking grand (mean of medians) L2 mean for each pair"
    for i,num in enumerate(n):
        
        files={}
        for j in bin_size: 
            files[j] = glob.glob(f"ukf_results/agg_ukf_agents_{num}_bin_{j}-*")

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
                L2_2.append(np.mean(np.nanmedian(distances,axis=0)))
        
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
    "boxplot for mean of medians for each run (30 in for each pop and bin size. mainly to see if we can get clearer results"
    
    f_name = f"Aggregate_grand_median_boxplot.pdf"
    y_name = "grand_median"
    f = plt.figure()
    sns.catplot(x="square width",y=y_name,col="n",kind="box", data=L2_frame)
    plt.tight_layout()
    if save:
        plt.savefig(f_name)
    

            
 #%%
if __name__ == "__main__":
    
    
    plot1 = True
    plot2 = True
    
    n=[10,20,30]
    bin_size = [5,10,25,50]
    
    save=True
    if plot1:
        L2 = grand_L2_matrix(n,bin_size)
        grand_L2_plot(L2,n,bin_size,True,save)
    if plot2:
        frame = boxplot_parser(n,bin_size)
        boxplot_plots(n,bin_size,frame,False,save)        
