import glob
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
"""
Translates results from base ukf experiments into plot.
plot indicates which performs best of noisy observations,
stationsim predictions and UKF predictions.


import data from arc with following in linux terminal
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
change to relevant directories
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
                file = glob.glob(f"ukf_results/agents_{n}_rate_{i}_noise_{j}_base_config_errors_*{k}")
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


def draw_3d_2lines(data1, data2,data3, ymin, ymax):
    '''Given data1 as a list of plots, each plot being a list
       of (x, y) vertices, generate a 3-d figure where each plot
       is shown as a translucent polygon.
       If line_at_zero, a line will be drawn through the zero point
       of each plot, otherwise the baseline will be at the bottom of
       the plot regardless of where the zero line is.
       Give also data2 for the second plot
    '''

    # add_collection3d() wants a collection of closed polygons;
    # each polygon needs a base and won't generate it automatically.
    # So for each subplot, add a base at ymin.

    for p in data1:
        p.insert(0, (p[0], 0))
        p.append((p[-1], 0))
        p=np.array(p)
    for p in data2:
        p.insert(0, (p[0], 0))
        p.append((p[-1], 0))
        p=np.array(p)
       
    for p in data3:
        p.insert(0, (p[0], 0))
        p.append((p[-1], 0))
        p=np.array(p)

    

    poly1 = PolyCollection(data1,lw=1.5,linestyle = '-', edgecolors=colours[0])

    poly2 = PolyCollection(data2,lw=1.5,linestyle = '--', edgecolors=colours[1])

    poly3 = PolyCollection(data2,lw=1.5,linestyle = '--', edgecolors=colours[2])

    zs = range(len(data1))
    plt.tight_layout(pad=2.0, w_pad=10.0, h_pad=3.0)
    ax.add_collection3d(poly1, zs=zs, zdir='y')
    ax.add_collection3d(poly2, zs=zs, zdir='y')
    ax.add_collection3d(poly3, zs=zs, zdir='y')

 
def plot_2_3d(data2,best_array,n,rates,noises,save):
    """
    3d version of plot 2 given Minh's idea/code above
    need to convert data into desire list of plots
    """
    #first make list of plots
    colours = ["yellow","orangered","skyblue"]

    "init and some labelling"
    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111,projection='3d')
    ax.set_xlabel('Noise')
    ax.set_ylabel("Rate")
    ax.set_zlabel('Grand L2 Error')
    ax.view_init(45,225)
    
    "take each rate plot l2 error over each noise for preds obs and ukf"
    for i,rate in enumerate(rates):
        sub_data = data2.loc[data2["rates"]==rate]
        preds=list(sub_data["prediction"])
        ukf=list(sub_data["ukf"])
        obs=list(sub_data["obs"])
        l1=ax.plot(xs=noises,ys=[i]*len(noises),zs=obs,color=colours[0],linewidth=4,
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()],alpha=0.5)
        l2=ax.plot(xs=noises,ys=[i]*len(noises),zs=preds,color=colours[1],linewidth=4,
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()],alpha=0.5)
        l3=ax.plot(xs=noises,ys=[i]*len(noises),zs=ukf,color=colours[2],linewidth=4,
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()],alpha=0.5)
             
    "placeholder dummies for legend"
    s1=matplotlib.lines.Line2D([-1],[-1],color=colours[0],label="obs",linewidth=4,
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()])
    s2=matplotlib.lines.Line2D([-1],[-1],color=colours[1],label="preds",linewidth=4,
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()])
    s3=matplotlib.lines.Line2D([-1],[-1],color=colours[2],label="ukf",
                path_effects=[pe.Stroke(linewidth=6, foreground='k',alpha=0.5), pe.Normal()])

    "rest of labelling"
    ax.set_xticks(np.arange(0,len(noises)))
    ax.set_xticklabels(noises)
    ax.set_yticks(np.arange(0,len(rates)))
    ax.set_yticklabels(rates)
    ax.legend([s1,s2,s3],["obs","preds","ukf"])
    plt.tight_layout()
    "save?"
    if save:
        plt.savefig(f"3d_{n}_error_trajectories.pdf")


        
    

if __name__ == "__main__":
    
    "parameters"
    ns=[10,30]
    #rates = [1,2,5,10,20,50] #.2 to 1 by .2
    rates = [1,2,5,10] #.2 to 1 by .2
    noises = [0,0.25,0.5,1,2,5]
    run_id = np.arange(0,30,1) #20 runs
    plot1 =True #do plot 1
    plot2 = True # do plot2
    plot3=True
    save =True # save plots
    
    for n in ns:
        data2,best_array = base_data_parser(n,rates,noises,run_id)
        
        if plot_1:
            plot_1(data2,best_array,n,rates,noises,save)
        if plot_2:
            plot_2(data2,best_array,n,rates,noises,save)
        if plot3:
            plot_2_3d(data2,best_array,n,rates,noises,save)
        
    "plot 1 gives visualisation of best_array"
