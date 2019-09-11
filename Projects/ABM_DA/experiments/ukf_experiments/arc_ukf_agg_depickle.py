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
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/agg* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
"""

import pickle
import sys
import os
sys.path.append("../../stationsim")
sys.path.append("../..")

from stationsim.ukf import plots as plots
from stationsim.ukf_aggregate import agg_plots as agg_plots

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

#%%

        
def l2_parser(instance,prop):
    "extract arrays of real paths, predicted paths, L2s between them."
    actual,preds,full_preds,truth = instance.data_parser(False)
    plts = plots(instance)
    truth[np.isnan(actual)]=np.nan #make empty values to prevent mean skewing in diagnostic plots
    
    true_o,b_o,plot_range = plts.plot_data_parser(truth,preds,True)    
    true_u,b_u,plot_range= plts.plot_data_parser(truth,preds,False)

    distances_obs,oindex,agent_means,t_mean_obs = plts.L2s(true_o,b_o)

    
    return distances_obs

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
    plt.ylabel("L2 Distribution for Aggregated Agents over Time")
    plt.title("L2s over time")
    if save:
        plt.savefig(f_name)
    
if __name__ == "__main__":
    "parameters for which number of agents and proportion observed to plot for"
    n=10
    bin_size = 5

    distances = [] #l2 distance matrices for each run observed/unobserved
    instances=[]
    files = glob.glob(f"ukf_results/agg_ukf_agents_{n}_bin_{bin_size}-1")
    
    for file in files:
        
        f = open(file,"rb")
        u = pickle.load(f)
        f.close()
    
        distance = l2_parser(u,prop)#
        distances.append(distance)
        instances.append(u)


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
        print("just one run for given params. giving single run diagnostics")
        actual,pred,full_preds,truth=u.data_parser(False)
        
        truth[np.isnan(actual)]=np.nan #make empty values to prevent mean skewing in diagnostic plots
        pred[np.isnan(actual)]=np.nan #make empty values to prevent mean skewing in diagnostic plots

        plts=plots(u)
        "single test diagnostics"
        save_plots=False

        "all observed just one plot"
        distances2,t_mean2 = plts.diagnostic_plots(truth,pred,True,save_plots)
        #plts.pair_frames(actual,full_preds) #basic animation
        #plts.pair_frames_stack_ellipse(actual,full_preds) #covariance and l2 trajectories. TAKES FOREVER




