import glob
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import colors

"""
Translates results from base ukf experiments into plot.
plot indicates which performs best of noisy observations,
stationsim predictions and UKF predictions.


import data from arc with following in linux terminal
scp medrclaa@arc3.leeds.ac.uk:/nobackup/medrclaa/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/* /home/rob/dust/Projects/ABM_DA/experiments/ukf_experiments/ukf_results/.
change medrclaa to relevant username
"""


if __name__ == "__main__":
    n=10
    rates = [1,2,5,10,20,50,100] #.2 to 1 by .2
    noises = [0,0.25,0.5,1,2,5,10,25,50,100]
    run_id = np.arange(0,5,1) #20 runs
    
    
    errors = []
    rates2 = []
    noises2 = []
    run_id2 = []
    
    "pull data from pickles into columns"
    
    for i in rates:
        for j in noises:
            for k in run_id:
                file = glob.glob(f"ukf_results/agents_{n}_rate_{i}_noise_{j}_base_config_errors_{k}")
                if file != []:
                    file=file[0]
                    rates2.append(i)
                    noises2.append(j)
                    run_id2.append(k)
                    
                    f = open(file,"rb")
                    error = pickle.load(f)
                    f.close()
                    errors.append(error)
    
    """this may seem counterintuitive but it preserves a 
    lot of information you need later for aggregating/imshow array"""
    
    run_id2 = np.vstack(run_id2)
    rates2 = np.vstack(rates2)
    noises2 = np.vstack(noises2)
    errors2 = np.vstack(errors)
    
    "convert to pandas for easier aggregation functions (afaik)"
    
    data = pd.DataFrame(np.concatenate((run_id2,rates2,noises2,errors2),axis=1))
    data.columns = ["run_id","rates","noise","actual","prediction","ukf"]
    data2 = data.groupby(["rates","noise"],as_index=False).agg("mean")
    
    "calculate best grand AED of obs preds or ukf for however many runs (5 for me)"
    
    best = []
    for i in range(len(data2.index)):
        row = np.array([data2["actual"][i],data2["prediction"][i],data2["ukf"][i]])
        best.append(np.where(row==np.nanmin(row))[0][0])
    
    data2["best"] = best
    
    "convert best errors into numpy array with rate rows and noise columns for imshow"
    
    best_array = np.ones((len(rates),len(noises)))*np.nan
    #rows are rates,columns are noise
    for i,rate in enumerate(rates):
        for j,noise in enumerate(noises):
            rate_rows = data2.loc[data2["rates"]==rate]
            rate_noise_row = rate_rows.loc[rate_rows["noise"]==noise]
            if len(rate_noise_row.values) != 0:
                best_array[i,j]= rate_noise_row["best"]
            
    "discrete matrix with y labels rates x labels noises  for imshow"
    
    
    
    
    f,ax = plt.subplots()
    "cbar axis"
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right",size="5%",pad=0.05)
    
    "custom discrete 3 colour map"
    cmap = colors.ListedColormap(["yellow","orangered","skyblue"])
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list("custom_map",cmaplist,cmap.N)
    bounds = [0,1,2,3]
    norm = colors.BoundaryNorm(bounds,cmap.N)
    
    "imshow plot and colourbar"
    im = ax.imshow(best_array,origin="lower",cmap = cmap,norm=norm)
    """alternative continous contour plot for more "real" mapping"""
    #grid = np.meshgrid(noises,rates)
    #im = plt.contourf(grid[0],grid[1],best_array,cmap=cmap,levels=[0,1,2,3])
    plt.ylim([0,2])
    cbar = plt.colorbar(im,cax=cax,ticks=np.arange(0,len(bounds)-1,1)+0.5,boundaries = [0,1,2])
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
    ax.set_xlabel("noise")
    ax.set_ylabel("sampling rate")
    ax.set_title("base ukf configuration experiment")
    plt.savefig(f"{n}_base_config_test.pdf")