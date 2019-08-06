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


"""
function to take instanced clases output from arc ukf scripts and produce grand mean plots.

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

    distances_obs,oindex,agent_means,t_mean_obs = plts.MAEs(a_o,b_o)
    if prop<1:
        distances_uobs,uindex,agent_means,t_mean_uobs = plts.MAEs(a_u,b_u)
    else:
        distances_uobs = []
    matplotlib.use("module://ipykernel.pylab.backend_inline")    
    preds[np.isnan(actual)]=np.nan
    
    return actual,preds,distances_obs,distances_uobs



def grand_MAE_matrix(n,prop,n_step):
    o_MAE = np.ones((len(n),len(prop)))*np.nan
    u_MAE = np.ones((len(n),len(prop)))*np.nan
    

    for i in n:
        i_index = (i-n.min())//n_step  
        files={}
        for j in prop:
            files[j.round(2)] = glob.glob(f"ukf_results/ukf_agents_{i}_prop_{j.round(2)}*")

        for _ in files.keys():
            o_MAE2=[]
            u_MAE2 = []
            for file in files[_]:
                f = open(file,"rb")
                u = pickle.load(f)
                f.close()
                actual,pred,do,du = l2_parser(u,float(_))#
                o_MAE2.append(np.nanmean(do))
                u_MAE2.append(np.nanmean(du))
        
            o_MAE[i_index,int(_*len(n))-1]=np.nanmean(o_MAE2)
            u_MAE[i_index,int(_*len(n))-1]=np.nanmean(u_MAE2)
            
    return o_MAE,u_MAE

def grand_MAE_plot(data,n,prop,n_step,p_step,observed):
    n_range = n.max()-n.min()
    prop_range = prop.max()-prop.min()
    
    f,ax=plt.subplots(figsize=(8,8))
    cmap = cm.viridis
    cmap.set_bad("black")
    data2 = np.ma.masked_where(np.isnan(data),data)
    im=ax.imshow(data2,interpolation="none",aspect="auto",cmap=cmap,extent = [10,35,0,1])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",pad=0.05)
    ax.set_xlim([n.min()-n_range/40,n.max()+n_step+n_range/40])
    ax.set_ylim([prop.min()-p_step-prop_range/40,prop.max()+prop_range/40])
    ax.set_xticks(n+n_step/2)
    ax.set_xticklabels(n)
    ax.set_yticks((prop-p_step/2))
    ax.set_yticklabels(prop.round(2))
    ax.set_xlabel("Number of Agents")
    cbar=plt.colorbar(im,cax,cax)
    if observed:
        cbar.set_label("Observed MAE")
        ax.set_title("Observed Agent MAEs")
        ax.set_ylabel("Proportion of Agents Observed (x100%)")
        plt.savefig("Observed_Grand_MAES.pdf")

    else:
        cbar.set_label("Unobserved MAE")
        ax.set_title("Unobserved Agent MAEs")
        ax.set_ylabel("Proportion of Agents Observed (x100%)")
        plt.savefig("Unobserved_Grand_MAES.pdf")
 
if __name__ == "__main__":
    
    n_step=5
    n_min = 10
    n_max = 30
    p_step=0.2
    p_min = 0.2
    p_max = 1.0
    
    n= np.arange(n_min,n_max+n_step,n_step)
    prop = 2*np.arange(p_min,p_max+p_step,p_step)/2
    
    O,U = grand_MAE_matrix(n,prop,n_step)
    #grand_MAE_plot(O,n,prop,n_step,p_step,True)
    #grand_MAE_plot(U,n,prop,n_step,p_step,False)
   
