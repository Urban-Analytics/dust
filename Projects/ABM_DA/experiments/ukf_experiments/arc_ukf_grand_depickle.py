"""
 produces a more generalised diagnostic over multiple runs using multiple
 numbers of agents for arc_ukf.py only. 
 This produces a chloropleth style map showing the grand mean error
 over both time and agents for various fixed numbers of agents 
 and proportions observed.

"""
import pickle
import sys
import os
sys.path.append("../../stationsim")
sys.path.append("../..")

from stationsim.ukf import ukf,ukf_ss,plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
import matplotlib.colors as col
import matplotlib.patheffects as pe

import glob
import seaborn as sns
import pandas as pd

plt.rcParams.update({'font.size':20})


#plt.style.use("dark_background")
        
def l2_parser(instance,prop):
    "extract arrays of true paths, predicted paths and l2 distances between them."
    
    actual,preds,full_preds,truth = instance.data_parser(False)
    truth[np.isnan(actual)]=np.nan #make empty values to prevent mean skewing in diagnostic plots
    
    plts = plots(instance)
    true_u,b_u,plot_range = plts.plot_data_parser(truth,preds,False)
    true_o,b_o,plot_range = plts.plot_data_parser(truth,preds,True)    

    distances_obs,oindex,agent_means,t_mean_obs = plts.L2s(true_o,b_o)
    if prop<1:
        distances_uobs,uindex,agent_means,t_mean_uobs = plts.L2s(true_u,b_u)
    else:
        distances_uobs = []
    matplotlib.use("module://ipykernel.pylab.backend_inline")    
    preds[np.isnan(actual)]=np.nan
    
    return actual,preds,distances_obs,distances_uobs



def grand_L2_matrix(n,prop,n_step): 
    "empty frames"
    o_L2 = np.ones((len(n),len(prop)))*np.nan
    u_L2 = np.ones((len(n),len(prop)))*np.nan
    
    "cycle over number of agents and proportions. taking grand (mean of means) L2 mean for each pair"
    for i in n:
        i_index = (i-n.min())//n_step  
        files={}
        for j in prop:
            if j ==1:
                files[j] = glob.glob(f"ukf_results/ukf_agents_{i}_prop_{1}*") 
                #wierd special case with 1 and 1.0 discrepancy
            else:
                files[j.round(2)] = glob.glob(f"ukf_results/ukf_agents_{i}_prop_{j.round(2)}*")

        for _ in files.keys():
            o_L2_2=[]
            u_L2_2 = []
            for file in files[_]:
                f = open(file,"rb")
                u = pickle.load(f)
                f.close()
                actual,pred,do,du = l2_parser(u,float(_))#
                o_L2_2.append(np.nanmean(do))
                u_L2_2.append(np.nanmean(du))
        
            o_L2[i_index,int(_*len(prop))-1]=np.nanmean(o_L2_2)
            u_L2[i_index,int(_*len(prop))-1]=np.nanmean(u_L2_2)
            
    return o_L2,u_L2
    
def grand_L2_plot(data,n,prop,n_step,p_step,observed,save):

    
    data = np.rot90(data,k=1) #rotate frame 90 degrees so right way up for plots
    
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
    ax.set_yticks(np.arange(len(prop)))
    ax.set_xticklabels(n)
    ax.set_yticklabels(prop.round(2))
    ax.set_xticks(np.arange(-.5,len(n),1),minor=True)
    ax.set_yticks(np.arange(-.5,len(prop),1),minor=True)
    ax.grid(which="minor",color="k",linestyle="-",linewidth=2)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Proportion of Agents Observed")
    plt.title("Grand L2s Over Varying Agents and Percentage Observed")


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
    if observed:
        cbar.set_label("Observed L2s")
        ax.set_title("Observed Agent L2s")
        ax.set_ylabel("Proportion of Agents Observed (x100%)")
        if save:
            plt.savefig("Observed_Grand_L2s.pdf")

    else:
        cbar.set_label("Unobserved L2")
        ax.set_title("Unobserved Agent L2s")
        ax.set_ylabel("Proportion of Agents Observed (x100%)")
        if save:
            plt.savefig("Unobserved_Grand_L2s.pdf")
            
def boxplot_parser(n,prop):
    observed = {}
    unobserved ={}
    for i in n:
        i_index = (i-n.min())//n_step  
        files={}
        for j in prop:
            if j ==1:
                files[j] = glob.glob(f"ukf_results/ukf_agents_{i}_prop_{1}*") 
                #wierd special case with 1 and 1.0 discrepancy
            else:
                files[j.round(2)] = glob.glob(f"ukf_results/ukf_agents_{i}_prop_{j.round(2)}*")
        observed[i] = {}
        unobserved[i] = {}
        for _ in files.keys():
            o_L2_2=[]
            u_L2_2 = []
            for file in files[_]:
                f = open(file,"rb")
                u = pickle.load(f)
                f.close()
                actual,pred,do,du = l2_parser(u,float(_))#
                
                o_L2_2.append(np.apply_along_axis(np.nanmean,0,do))
                u_L2_2.append(np.apply_along_axis(np.nanmean,0,du))
            observed[i][_] = np.hstack(o_L2_2)
            unobserved[i][_] = np.hstack(u_L2_2)
          
    "stack dictionaries into dataframe with corresponding n and prop next to each agent error"
    obs_sub_frames = []
    uobs_sub_frames = []
    
    for i in n:
        for j in prop:
            obs = observed[i][j]
            uobs = unobserved[i][j]
            obs_sub_frames.append(pd.DataFrame([[str(i)]*len(obs),[str(j)]*len(obs),obs]).T)
            uobs_sub_frames.append(pd.DataFrame([[str(i)]*len(uobs),[str(j)]*len(uobs),uobs]).T)

    "stack into grand frames and label columns"
    obs_frame = pd.concat(obs_sub_frames)
    obs_frame.columns = ["n","proportion observed (x100%)","observed L2 agent errors"]
    uobs_frame = pd.concat(uobs_sub_frames)
    uobs_frame.columns = ["n","proportion observed (x100%)","unobserved L2 agent errors"]

    
    return obs_frame ,uobs_frame

def boxplot_plots(n,prop,frame,separate,observed,save):  
    if separate:
        for i in n:
            if observed:
                f_name = f"Observed_boxplot_{i}.pdf"
                y_name = "observed L2 agent errors"
                n_subframe = obs_frame.loc[obs_frame["n"]==str(i)]
    
            else:
                y_name = "unobserved L2 agent errors"
                f_name = f"Unobserved_boxplot_{i}.pdf"
                n_subframe = uobs_frame.loc[uobs_frame["n"]==str(i)]
    
            f = plt.figure()
            sns.boxplot(x="proportion observed (x100%)",y=y_name,data=n_subframe)
            if save:
                f.savefig(f_name)
    
    else:
        if observed:
            f_name = f"Observed_boxplot.pdf"
            y_name = "observed L2 agent errors"

        else:
            y_name = "unobserved L2 agent errors"
            f_name = f"Unobserved_boxplot.pdf"

        f = plt.figure()
        sns.catplot(x="proportion observed (x100%)",y=y_name,col="n",kind="box", data=frame)
        plt.tight_layout()
        if save:
            plt.savefig(f_name)
    
 
if __name__ == "__main__":
    
    n_step=10
    n_min = 10
    n_max = 30
    p_step=0.25
    p_min = 0.25
    p_max = 1.0
    
    plot1 = False
    plot2 = True
    n= np.arange(n_min,n_max+n_step,n_step)
    prop = np.arange(p_min,p_max+p_step,p_step)
    
    save=True
    if plot1:
        O,U = grand_L2_matrix(n,prop,n_step)
        grand_L2_plot(O,n,prop,n_step,p_step,True,save)
        grand_L2_plot(U,n,prop,n_step,p_step,False,save)
    if plot2:
        obs_frame ,uobs_frame = boxplot_parser(n,prop)
        boxplot_plots(n,prop,obs_frame,False,True,save)        
        boxplot_plots(n,prop,uobs_frame,False,False,save)