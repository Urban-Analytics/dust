"""
detailed diagnostics using multiple runs of a fixed number of agents for both 
arc_ukf.py and arc_ukf_agg.py. At each time point we sample the 
mean agent errors from each run as a population of means.
The mean and variance of this sample are plotted to demonstrate
the average error and uncertainty of the UKF over time. 
If the population is fully observed (as always with the aggregate case)
then only one plot is produced. 
Otherwise both observed and unobserved plots are produced.

download all class instances from arc
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/ukf* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""

import pickle
import sys
import os
sys.path.append("../../stationsim")
sys.path.append("../..")

from stationsim.ukf import plots as plots

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import glob
import seaborn as sns
import warnings

plt.rcParams.update({'font.size':20})
"""
function to take instanced clases output from arc ukf scripts and produce grand mean plots.

"""


class HiddenPrints:
    "suppress repeat printing of random plots"
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        
def l2_parser(instance,prop):
    "extract arrays of real paths, predicted paths, L2s between them."
    "HiddenPrints suppresses plots class from spam printing figures"
    matplotlib.use("Agg")
    actual,preds,full_preds = instance.data_parser(False)
    actual = actual[1:,:]
    plts = plots(instance)
    a_u,b_u,plot_range = plts.plot_data_parser(actual,preds,False)
    a_o,b_o,plot_range = plts.plot_data_parser(actual,preds,True)    

    distances_obs,oindex,agent_means,t_mean_obs = plts.L2s(a_o,b_o)
    if prop<1:
        distances_uobs,uindex,agent_means,t_mean_uobs = plts.L2s(a_u,b_u)
    else:
        distances_uobs = []
    
    return actual,preds,distances_obs,distances_uobs

def grand_mean_plot(data,f_name,instance,save):
    """
    take list of L2 dataframes and produces confidence plot for given number
    of agents and proportion over time.
    
    This function is a bit odd as it essentially 
    stacks a bunch of frames into two massive columns.
    Blame sns.lineplot.
    
    some entries have time stamps but are NaN.
    due to all (un)observed agents finished but other set still going
    typically happens when one agent is very slow.
    
    Can have ending of graphs with no confidence interval due to only 1 run being
    left by the end.
    """
    
    reg_frames = []
    for i,frame in enumerate(data):
        mean_frame = np.ones((frame.shape[0],2))*np.nan
        mean_frame[:,0] = np.arange(0,frame.shape[0],1)*instance.filter_params["sample_rate"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",category = RuntimeWarning)
            mean_frame[:,1]=(np.nanmean(frame,1))
        reg_frames.append(mean_frame)
    
    grand_frame = np.vstack(reg_frames)
    """
    plots using regression style line plot from seaborn
    easiest way to build confidence intervals 
    for mean and variance of average agent error at each time point
    """
    f = plt.figure()
    sns.lineplot(grand_frame[:,0],grand_frame[:,1],lw=3)
    plt.xlabel("Time (steps)")
    plt.ylabel("L2 Distribution over Time")
    plt.title("L2s over time")
    if save:
        plt.savefig(f_name)
    
if __name__ == "__main__":
    "parameters for which number of agents and proportion observed to plot for"
    n=10
    prop = 0.25
    actuals = []
    preds = []
    d_obs = []
    d_uobs = []
    instances=[]
    files = glob.glob(f"ukf_results/ukf_agents_{n}_prop_{prop}-*")
    
    for file in files:
        
        f = open(file,"rb")
        u = pickle.load(f)
        f.close()
    
        actual,pred,d1,d2 = l2_parser(u,prop)#
        actuals.append(actual)
        preds.append(pred)
        d_obs.append(d1)
        instances.append(u)
        if prop<1:
            d_uobs.append(d2)
    #plts = plots(u)
    save_plots =False
    if len(files)>1:
        #observed grand L2
        grand_mean_plot(d_obs,f"L2_obs_{n}_{prop}.pdf",u,save_plots)
        if prop<1:
            #unobserved grand L2
            grand_mean_plot(d_uobs,f"L2_uobs_{n}_{prop}.pdf",u,save_plots)
            #plts.trajectories(actual)
            #plts.pair_frames(actual,preds)
    else:
        actual,pred,full_preds=u.data_parser(False)
        actual = actual[1:,:]
        plts=plots(u)
        "single test diagnostics"
        save_plots=False
        if prop<1:
            "unobserved agents then observed agents"
            distances,t_mean = plts.diagnostic_plots(actual,pred,False,save_plots)
        
        "all observed just one plot"
        distances2,t_mean2 = plts.diagnostic_plots(actual,pred,True,save_plots)
        #plts.pair_frames(actual,full_preds) #basic animation
        #plts.pair_frames_stack_ellipse(actual,full_preds) #covariance and l2 trajectories. TAKES FOREVER




