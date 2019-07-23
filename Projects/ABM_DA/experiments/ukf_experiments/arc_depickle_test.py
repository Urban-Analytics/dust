import pickle
import sys
import os
sys.path.append("../../stationsim")
sys.path.append("../..")

from stationsim.ukf import ukf,ukf_ss,plots
from stationsim_model import Model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import glob
import seaborn as sns
import warnings

plt.style.use("dark_background")

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def l2_parser(instance,prop):
    matplotlib.use("Agg")
    actual,preds = instance.data_parser(False)
    plts = plots(instance)
    with HiddenPrints():        
        distances_obs,t_mean_obs = plts.diagnostic_plots(actual,preds,True,False)
        if prop<1:
            distances_uobs,t_mean_uobs = plts.diagnostic_plots(actual,preds,False,False)
        else:
            distances_uobs = []
    matplotlib.use("module://ipykernel.pylab.backend_inline")    
    preds[np.isnan(actual)]=np.nan
    
    return actual,preds,distances_obs,distances_uobs

def grand_mean_plot(data,f_name):
    "take list of dataframes take means and stack them by row"
    reg_frames = []
    for i,frame in enumerate(data):
        mean_frame = np.ones((frame.shape[0],2))*np.nan
        mean_frame[:,0] = np.arange(0,frame.shape[0],1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category = RuntimeWarning)
            mean_frame[:,1]=(np.nanmean(frame,1))
        reg_frames.append(mean_frame)
    
    grand_frame = np.vstack(reg_frames)
    
    sns.lineplot(grand_frame[:,0],grand_frame[:,1])
    plt.savefig(f_name)
    plt.show()
    plt.close()
    
if __name__ == "__main__":
    
    n=50
    prop = 0.8
    actual = []
    preds = []
    d_obs = []
    d_uobs = []

    files = glob.glob(f"ukf_results/ukf_agents_{n}_prop_{prop}-0*")
    
    for file in files:
        
        f = open(file,"rb")
        u = pickle.load(f)
        f.close()
    
        a,b,d1,d2 = l2_parser(u,prop)#
        actual.append(a)
        preds.append(b)
        d_obs.append(d1)
        if prop<1:
            d_uobs.append(d2)
    plts = plots(u)
    grand_mean_plot(d_obs,f"obs_{n}_{prop}.pdf")
    if prop<1:
        grand_mean_plot(d_uobs,f"uobs_{n}_{prop}.pdf")
        plts.trajectories(a)
        plts.pair_frames(a,b)
    
    
        



