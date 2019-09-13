import os
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
#plt.ioff() # Turn off interactive mode
import pandas as pd
import warnings
from scipy.interpolate import griddata # For interpolating across irregularly spaced grid
import pickle # For saving computationally-expensive operations
import seaborn as sns

# The following is to import the Particle Filter code
# (mostly we just read results that were created previously, but sometimes it's  useful to
# visualise additional experiments).
import sys
sys.path.append('../../stationsim')
sys.path.append('../..')
from stationsim.particle_filter import ParticleFilter
from stationsim.stationsim_model import Model
import time
import multiprocessing

# We need to tell the script which directory the results are in

# First try to get the directory of this .ipynb file
#root_dir = !echo %cd% # this might work under windows
root_dir = !pwd # under linux/mac # This works in linux/mac

# Now append the specific directory with results:
path = os.path.join(root_dir[0], "results","2/noise1")
print(f"Plotting results in directory: {path}")

# Need to set the number of particles and agents used in the experiments. These are set in the file that runs
# the experiments: ./run_pf.py
# Copy the lines near the top that set the number of particles and agents

# Lists of particles, agent numbers, and particle noise levels
num_par = list ( [1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + [3000, 5000, 7500, 10000])
num_age = [2, 5, 10, 15, 20, 30, 40, 50]

# Use log on y axis?
uselog = True

# Font sizes for figures. These match those in pf_experiments_plots.ipynb
# (from https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot)
SMALL_SIZE = 10
MEDIUM_SIZE = 11
BIGGER_SIZE = 13
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the figure title (when using axes)

# These are the basic parameter settings required.
# We will chance the number of particles and agents to see what the experiments are like

model_params = {
    'width': 200,
    'height': 100,
    'pop_total': 10, # IMPORTANT: number of agents
    'speed_min': .1,
    'separation': 2,
    'batch_iterations': 4000,  # Only relevant in batch() mode
    'do_history': False,
    'do_print': False,
}
# Model(model_params).batch() # Runs the model as normal (one run)



filter_params = {
    'number_of_particles': 10, #IMPORTANT: number of particles
    'number_of_runs': 1,  # Number of times to run each particle filter configuration
    'resample_window': 100,
    'multi_step': True,  # Whether to predict() repeatedly until the sampling window is reached
    'particle_std': 1.0, # Noise added to particles
    'model_std': 1.0, # Observation noise
    'agents_to_visualise': 10,
    'do_save': True,
    'plot_save': False,
    'do_ani': True, # Do the animation (generatea plot at each data assimilation window)
    'show_ani': False, # Don't actually show the animation. They can be extracted later from self.animation
}

def run_variance_experiment(pop, particles, num_runs=10, num_cores=1):
    model_params['pop_total'] = pop
    filter_params['number_of_particles'] = particles
    filter_params['do_ani'] = False
    variances = []
    errors = []
    windows = []

    for i in range(num_runs):
        print(f"\n ********** Starting Experiment {i + 1}/{num_runs} **********")
        pf = ParticleFilter(Model, model_params, filter_params, numcores=num_cores )
        result = pf.step()  # Run the particle filter
        # Get the variances and errors *after* resampling
        assert (len(pf.variances) == len(pf.mean_errors))
        variances = variances + [pf.variances[j] for j in range(len(pf.variances)) if not pf.before_resample[j]]
        errors = errors + [pf.mean_errors[j] for j in range(len(pf.mean_errors)) if not pf.before_resample[j]]
        assert (len(variances) == len(errors))
        # The number of windows varies for each run
        windows = windows + list(range(1, len([x for x in pf.before_resample if x == True]) + 1))
        assert (len(windows) == len(variances))
        pf.pool.close()  # For some reason the pool doesn't always kill it's child processes (notebook problem?)

    print("********** Finished Experiments **********")
    return pd.DataFrame(list(zip(windows, errors, variances)), columns=["Window", "Error", "Variance"])

N = 10  # Number of runs for each experiment

# Experiment 1: 5 agents, 10 particles
results1 = run_variance_experiment(pop=20, particles=10, num_runs=N, num_cores=1)
# Also melt the results for the boxplots coming shortly
results1_melt = pd.melt(results1, id_vars=['Window'], value_vars=['Error', 'Variance'])

# Experiment 2: 30 agents, 10 particles
results2 = run_variance_experiment(pop=20, particles=10, num_runs=N, num_cores=8)
results2_melt = pd.melt(results2, id_vars=['Window'], value_vars=['Error', 'Variance'])

# Experiment 3: 30 agents, 5,000 particles
results3 = run_variance_experiment(pop=20, particles=10, num_runs=N, num_cores=9)
results3_melt = pd.melt(results3, id_vars=['Window'], value_vars=['Error', 'Variance'])

results4 = run_variance_experiment(pop=20, particles=10, num_runs=N, num_cores=10)
results4_melt = pd.melt(results4, id_vars=['Window'], value_vars=['Error', 'Variance'])

# plt.tight_layout()
fig, [ax1, ax2, ax3, ax4] = plt.subplots(nrows=4, ncols=1, figsize=(7, 7), dpi=128, facecolor='w', edgecolor='k')
fig.set_tight_layout(True)

for i, (axis, results, title) in enumerate([ \
        (ax1, results1_melt, "10 agents, 10 particles, 1 cores"), \
        (ax2, results2_melt, "10 agents, 10 particles, 8 cores"), \
        (ax3, results3_melt, "10 agents, 10 particles, 9 cores"), \
        (ax4, results4_melt, "10 agents, 10 particles, 10 cores")]):

    sns.boxplot(ax=axis, x="Window", y="value", hue="variable", data=results, linewidth=1, fliersize=1)
    axis.set_ylim(-1, 7)
    axis.set_xlim(-1, 11)
    axis.title.set_text(title)
    axis.set_ylabel("")
    # Overide the default seaborn legend (https://stackoverflow.com/questions/35538882/seaborn-boxplot-stripplot-duplicate-legend)
    handles, labels = axis.get_legend_handles_labels()  # Get the handles and labels.
    if i == 0:  # Make a legend on the first plot, but not on the others
        axis.legend(handles, labels, loc="upper right")
    else:
        axis.legend().set_visible(False)

# fig.savefig("figs_for_pf_paper/variance_results.png", bbox_inches="tight")