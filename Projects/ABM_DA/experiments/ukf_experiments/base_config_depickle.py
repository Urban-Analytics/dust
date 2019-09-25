import glob
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.patheffects as pe

"""
Translates results from base ukf experiments into plot.
plot indicates which performs best of noisy observations,
stationsim predictions and UKF predictions.


import data from arc with following in linux terminal
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
change medrclaa to relevant username
"""

def base_data_parser(n,rates,noises,run_id):
    errors = []
    rates2 = []
    noises2 = []
    run_id2 = []
    
    """
    pull data from pickles into columns.
    each row has a rate, noise, run_id (out of 10,30) 
    and mean L2 error for obs stationsim and ukf respectively
    """
    
    for i in rates:
        for j in noises:
            for k in run_id:
                file = glob.glob(f"/ukf_results/agents_{n}_rate_{i}_noise_{j}_base_config_errors_{k}")
                if file != []:
                    file=file[0]
                    rates2.append(i)
                    noises2.append(j)
                    run_id2.append(k)
                    
                    f = open(file,"rb")
                    error = pickle.load(f)
                    f.close()
                    errors.append(error)
    
    "lists to numpy arrays"
    
    run_id2 = np.vstack(run_id2)
    rates2 = np.vstack(rates2)
    noises2 = np.vstack(noises2)
    errors2 = np.vstack(errors)
    
    "convert to pandas for aggregation functions"
    "mean aggregate 30 runs for each noise and rate pair"
    data = pd.DataFrame(np.concatenate((run_id2,rates2,noises2,errors2),axis=1))
    data.columns = ["run_id","rates","noise","obs","prediction","ukf"]
    data2 = data.groupby(["rates","noise"],as_index=False).agg("mean")
    
    "calculate best (minimum) obs, preds, or ukf L2 error for each rate/noise pair"
    " assign 0,1, or 2 to represent obs, preds, or ukf performing best respectively"
    best = []
    for i in range(len(data2.index)):
        row = np.array([data2["obs"][i],data2["prediction"][i],data2["ukf"][i]])
        best.append(np.where(row==np.nanmin(row))[0][0])
    
    
    data2["best"] = best
    
    "convert best errors into numpy array with rate rows and noise columns for imshow"
    "produces a len(rates)xlen(noises) matrix."
    "rows represent rates, columns represent noises"
    "matrix entry i,j gives best estimate for each rates/noise pair"
    
    best_array = np.ones((len(rates),len(noises)))*np.nan
    #rows are rates,columns are noise
    for i,rate in enumerate(rates):
        for j,noise in enumerate(noises):
            rate_rows = data2.loc[data2["rates"]==rate]
            rate_noise_row = rate_rows.loc[rate_rows["noise"]==noise]
            if len(rate_noise_row.values) != 0:
                best_array[i,j]= rate_noise_row["best"]

    return data2,best_array

def plot_1(data2,best_array,b,rates,noises,save):
        f,ax = plt.subplots(figsize=(8,8))
        "cbar axis"
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right",size="5%",pad=0.05)
        colours = ["yellow","orangered","skyblue"]
        "custom discrete 3 colour map"
        cmap = colors.ListedColormap(colours)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = colors.LinearSegmentedColormap.from_list("custom_map",cmaplist,cmap.N)
        bounds = [0,1,2,3]
        norm = colors.BoundaryNorm(bounds,cmap.N)
        
        "imshow plot and colourbar"
        im = ax.imshow(best_array,origin="lower",cmap = cmap,norm=norm)
        #"""alternative continous contour plot idea for more "spatially real" mapping"""
        #grid = np.meshgrid(noises,rates)
        #im = plt.contourf(grid[0],grid[1],best_array,cmap=cmap,levels=[0,1,2,3])
        plt.ylim([0,2])
        cbar = plt.colorbar(im,cax=cax,ticks=np.arange(0,len(bounds)-1,1)+0.5,boundaries = [0,1,2,3])
        cbar.set_label("Best Error")
        cbar.set_alpha(1)
        cbar.draw_all()
        
        "labelling"
        cbar.set_ticklabels(("Obs","Preds","UKF"))
        ax.set_xticks(np.arange(len(noises)))
        ax.set_yticks(np.arange(len(rates)))
        ax.set_xticklabels(noises)
        ax.set_yticklabels(rates)
        ax.set_xticks(np.arange(-.5,len(noises),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(rates),1),minor=True)
        ax.grid(which="minor",color="k",linestyle="-",linewidth=2)
        ax.set_xlabel("Noise (std)")
        ax.set_ylabel("Sampling Frequency")
        ax.set_title("base ukf configuration experiment")
        if save:
            plt.savefig(f"{n}_base_config_test.pdf")
    

"""
plot 2 - plots errors for all 3 estimates simultaneously indicating
which is best (minimum) for each noise pair.
modular x axis allows for pandas style double indexing.
each sampling rate is plotted first on the x axis
for each rate every noise is also plotted in list order
"""
    
def plot_2(data2,best_array,n,rates,noises,save):
        g = plt.figure(figsize=(10,8))
        colours = ["yellow","orangered","skyblue"]
        ax1 = g.add_subplot(111)
        "line plots"
        l1=plt.plot(data2["obs"],label="obs",
                 color = colours[0],linewidth = 4,path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
        l2=plt.plot(data2["prediction"],label="prediction",linestyle=(0,(1,1)),
                 color=colours[1],linewidth = 4,path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
        l3=plt.plot(data2["ukf"],label = "ukf",linestyle=(0,(0.5,1,2,1)),
                 color=colours[2],linewidth = 4,path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
        plt.legend()
        
        "base line"
        ax1.axhline(y=0,color="grey",alpha=0.5,linestyle=":")
        
        "which estimate performs best vertical lines"
        for i in range(len(data2["best"])):
            best = data2["best"][i]
            if best ==0:
                plt.plot(np.array([i,i]),np.array([0,data2["obs"][i]]),
                         color=colours[0],linewidth = 3,
                         path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
            elif best==1:
                ax1.plot(np.array([i,i]),np.array([0,data2["prediction"][i]]),
                         color=colours[1],linestyle=":",linewidth = 3,
                         path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
            elif best==2:
                ax1.plot(np.array([i,i]),np.array([0,data2["ukf"][i]]),
                         color=colours[2],linestyle=(0,(0.5,1,2,1)),linewidth = 3,
                         path_effects=[pe.Stroke(linewidth=6, foreground='k'), pe.Normal()])
          
        "lines from max estimate to noise label"
        for j in range(2*len(noises)):
            maxim = np.nanmax((data2[["obs","prediction","ukf"]].loc[[j]]).to_numpy())
            noise = j%len(noises)
            plt.plot(np.array([j,j]),np.array([maxim,((maxim/(maxim+0.5))*5)+1]),
                     color="grey",linestyle="-")
            plt.text(j-0.1,((maxim/(maxim+0.5))*5)+1,str(noises[noise]),fontsize=18)
        
        plt.text(1,4,"Noises",fontsize=18)
        
        "labelling"
        ax1.tick_params(labelsize=28)
        plt.ylabel(f"Mean Agent L2 over {n} runs",fontsize=24)
        "split x axis labels into two"
        #ax2 = ax1.twiny()
        "noise labels"
        #ax2.set_xticks(np.arange(0,data2.shape[0],1))
        #ax2.set_xticklabels(noises*len(rates),fontsize=16)
        #plt.setp(ax2.get_xticklabels(),rotation=90)
        #ax2.set_xlabel("Noise (std)")
        "rate labels"
        ax1.set_xlabel("Sampling Frequency",fontsize=24)
        ax1.set_xticks(np.arange(0,len(rates)*len(noises),len(noises)))
        ax1.set_xticklabels(rates,fontsize=28)
        
        plt.tight_layout()
        if save:
            plt.savefig(f"{n}_error_trajectories.pdf")

if __name__ == "__main__":
    
    "parameters"
    n=30
    #rates = [1,2,5,10,20,50] #.2 to 1 by .2
    rates = [1,2,5,10] #.2 to 1 by .2
    noises = [0,0.25,0.5,1,2,5]
    run_id = np.arange(0,30,1) #20 runs
    plot1 =True #do plot 1
    plot2 = True # do plot2
    save =True # save plots
    
    data2,best_array = base_data_parser(n,rates,noises,run_id)
    
    if plot_1:
        plot_1(data2,best_array,n,rates,noises,save)
    if plot_2:
        plot_2(data2,best_array,n,rates,noises,save)
            
    
    "plot 1 gives visualisation of best_array"
