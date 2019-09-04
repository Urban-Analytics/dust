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

import glob
import seaborn as sns
import warnings

plt.rcParams.update({'font.size':20})

"""
function to take instanced clases output from arc ukf scripts and produce grand mean plots.
These plots take MSEs for each mean and variance into one grand MSE 
"""

#plt.style.use("dark_background")

class HiddenPrints:
    "suppress repeat printing"
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def l2_parser(instance,prop):
    "extract arrays of real paths, predicted paths, l2 distances between them."
    "HiddenPrints suppresses plots class from spam printing figures"
    matplotlib.use("Agg")
    actual,preds,full_preds = instance.data_parser(False)
    plts = plots(instance)
    a_u,b_u,plot_range = plts.plot_data_parser(actual,preds,False)
    a_o,b_o,plot_range = plts.plot_data_parser(actual,preds,True)    

    distances_obs,oindex,agent_means,t_mean_obs = plts.L2s(a_o,b_o)
    if prop<1:
        distances_uobs,uindex,agent_means,t_mean_uobs = plts.L2s(a_u,b_u)
    else:
        distances_uobs = []
    matplotlib.use("module://ipykernel.pylab.backend_inline")    
    preds[np.isnan(actual)]=np.nan
    
    return actual,preds,distances_obs,distances_uobs



def grand_L2_matrix(n,prop,n_step):
    o_L2 = np.ones((len(n),len(prop)))*np.nan
    u_L2 = np.ones((len(n),len(prop)))*np.nan
    

    for i in n:
        i_index = (i-n.min())//n_step  
        files={}
        for j in prop:
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
    f,ax=plt.subplots(figsize=(8,8))
    cmap = cm.viridis
    cmap.set_bad("black") #set nans for unobserved full prop to black
    data2 = np.ma.masked_where(np.isnan(data),data) #needed to get bad black squares in imshow
    data2=np.flip(data2,axis=0)
    im=ax.imshow(data2,interpolation="nearest",cmap=cmap,origin="lower")
    #ax.set_xlim([n.min()-n_step/20,n.max()+n_step/20])
    #ax.set_ylim([prop.min()-p_step/20,prop.max()+p_step/20])
    #ax.set_xticks((n+n_step/2)[:-1])
    ax.set_xticks(np.arange(len(n)))
    ax.set_yticks(np.arange(len(prop)))
    ax.set_xticklabels(n)
    ax.set_yticklabels(prop.round(2))
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Proportion of Agents Observed")
    plt.title("Grand L2s Over Varying Agents and Percentage Observed")
    b = im.get_extent()
    ax.set_xlim([b[0]-len(n)/40,b[1]+len(n)/40])
    ax.set_ylim([b[2]-len(prop)/40,b[3]+len(prop)/40])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",pad=0.05)
    cbar=plt.colorbar(im,cax,cax)
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
 
if __name__ == "__main__":
    
    n_step=5
    n_min = 10
    n_max = 50
    p_step=0.2
    p_min = 0.2
    p_max = 1.0
    
    n= np.arange(n_min,n_max+n_step,n_step)
    prop = np.arange(p_min,p_max+p_step,p_step)
    
    O,U = grand_L2_matrix(n,prop,n_step)
    save=True
    grand_L2_plot(O,n,prop,n_step,p_step,True,save)
    grand_L2_plot(U,n,prop,n_step,p_step,False,save)
   
