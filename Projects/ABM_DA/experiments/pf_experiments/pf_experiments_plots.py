# -*- coding: utf-8 -*-
"""
DEPRICATED. I don't think this works any more. To analyse the results of the experiments you should
use pf_experiments_plots.ipynb.

Reads the out files that are created by station_sim/particle_filter.py

This version accounts for two types of results being written for each 
experiment (before and after reweighting) and also doesn't assume that the 
agents v.s. particles heat map is evenly spaced.

@author: medkmin
@author: nickmalleson
"""

#%% Initialise and read files

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import warnings
from scipy.interpolate import griddata # For interpolating across irregularly spaced grid

# Needs to be set to location of results
path = os.path.join(sys.path[0], "results","2/noise1")

# Need to set the number of particles and agents used in the experiments. These are set in the file that runs
# the experiments: ./run_pf.py
# Copy the lines near the top that set the number of particles and agents
# TODO: work these out from the results file names

# Lists of particles, agent numbers, and particle noise levels
num_par = list ( [1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + [3000, 5000, 7500, 10000])
num_age = [2, 5, 10, 15, 20, 30, 40, 50]

# Use log on y axis?
uselog = True

# Type of interpolation i.e. 'nearest' of 'linear' (see help(griddata)))
interpolate_method = "nearest"
#interpolate_method = "linear"

# From now on refer to the lists of agents and particles using different names (TODO: refactor)
particles = num_par
agents = num_age


if not os.path.isdir(path):
    sys.exit("Directory '{}' does not exist".format(path))


def is_duplicate(fname, files):
    """
    Sees if `fname` already exists in the `duplicates` list. Needs to strip off the integers at the end of
    the file (these were added by the Particle Filter script to prevent overridding experiments).
    :param fname:
    :param duplicates:
    :return: True if this file exists already, false otherwise
    """
    regex = re.compile(r"(.*?)-\d*\.csv") # Find everthing before the numbers at the end of the filename
    fname_stripped = re.findall(regex, fname)[0]
    for f in files:
        if re.findall(regex, f)[0] == fname_stripped:
            return True # There is a duplicate
    return False # No duplicates found


files = []
duplicates = [] # List of diplicate files
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            fname = os.path.join(r, file)
            if is_duplicate(fname, files):
                duplicates.append(fname)
            else:
                files.append(fname)

if len(files) == 0:
    sys.exit("Found no files in {}, can't continue".format(path) )
elif len(duplicates) > 0:
    warnings.warn("Found {} duplicate files:\n\t{}".format(len(duplicates), "\n\t".join(duplicates)))
else:
    print("Found {} files".format(len(files)))


# Errors are a matrix of particles * agents. These are a tuple because the errors
# are calculated before and after resampling

def init_matrix():
    return ( np.zeros(shape=(len(particles),len(agents))), np.zeros(shape=(len(particles),len(agents))) )

min_mean_err = init_matrix()
max_mean_err = init_matrix()
ave_mean_err = init_matrix() # Mean of the mean errors
med_mean_err = init_matrix() # Median of the mean errors
min_abs_err  = init_matrix()
max_abs_err  = init_matrix()
ave_abs_err  = init_matrix() # Mean of the mean errors
med_abs_err  = init_matrix() # Median of the mean errors
min_var      = init_matrix()
max_var      = init_matrix()
ave_var      = init_matrix()
med_var      = init_matrix()

# Regular expressions to find the particle number and population total from json-formatted info at the start of the file
particle_num_re = re.compile(r".*?'number_of_particles': (\d*).*?")
agent_num_re = re.compile(r".*?'pop_total': (\d*).*?")

print("Reading files....",)
data_shape = None # Check each file has a consistent shape
for i, f in enumerate(files):

    file = open(f,"r").read()
    #data = pd.read_csv(f, header = 2).replace('on',np.nan)
    data = pd.read_csv(f, header=2).replace('on', np.nan)
    # Check that each file has a consistent shape
    if i==0:
        data_shape=data.shape
    if data.shape != data_shape:
        # If the columns are the same and there are only a few (20%) rows missing then just continue
        if ( data_shape[1] == data.shape[1] ) and ( data.shape[0] > int(data_shape[0] - data_shape[0]*0.2) ):
            warnings.warn("Current file shape ({}) does not match the previous one ({}). Current file is: \n\t{}. \n\tLess than 20% rows missing so continuing".format(
                                  str(data.shape), str(data_shape), f ))
        # Can exit if the shapes are too bad (turn this off for now)
        #else:
        #    sys.exit("Current file shape ({}) does not match the previous one ({}). Current file is: \n\t{}. \n\tNot continuing".format(
        #            str(data.shape), str(data_shape), f  ))

    # Filter by whether errors are before or after (NOW DONE LATER, THIS CAN GO)
    #data = data[ data.loc[:,'Before_resample?'] == before]

    # Find the particle number and population total from json-formatted info at the start of the file
    search = re.findall(particle_num_re, file)
    try:
        particle_num = int(search[0])
    except:
        sys.exit("Error: could not find the number of particles in the header for file \n\t{}\n\tFound: '{}'".format(f, search))
    search = re.findall(agent_num_re, file)
    try:
        agent_num = int(search[0])
    except:
        sys.exit("Error: could not find the number of agents in the header for file \n\t{}\n\tFound: '{}'".format(f, search))
        
    # Calculate the statstics before and after resampling
    for before in [0,1]:
        d = data[ data.loc[:,'Before_resample?'] == before] # Filter data

        data_mean =  d.mean() # Calculate the mean of all columns
        data_median = d.median() # and also sometimes use median

        min_mean_err[before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Mean_errors']
        max_mean_err[before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Mean_errors']
        ave_mean_err[before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_mean_errors']
        med_mean_err[before][particles.index(particle_num),agents.index(agent_num)] = data_median['Average_mean_errors']
        min_abs_err [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Absolute_errors']
        max_abs_err [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Absolute_errors']
        ave_abs_err [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_Absolute_errors']
        med_abs_err [before][particles.index(particle_num),agents.index(agent_num)] = data_median['Average_Absolute_errors']
        min_var     [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_variances']
        max_var     [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_variances']
        ave_var     [before][particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_variances']
        med_var     [before][particles.index(particle_num),agents.index(agent_num)] = data_median['Average_variances']
        
# There will never be zero error, so replace 0s with NA
data[data == 0] = np.nan        

print("...finished reading {} files".format(len(files)))


#%% Do some sanity checking

# Max error after resampling should be less than before resampling
print(f"\tMax errors (mean) (before/after): {max_mean_err[0].max()} / {max_mean_err[1].max()}")
assert max_mean_err[1].max() < max_mean_err[0].max() 
# Min error should not have gone up
print(f"\tMin errors (before/after): {min_mean_err[0].max()} / {min_mean_err[1].max()}")
assert min_mean_err[1].max() <= min_mean_err[0].max()
# Average errors should mostly have gone down (wont always happen due to addition of particle noise)
false_count = 0
total = 0
for b,a in zip(ave_mean_err[0], ave_mean_err[1]):
    total += len(b)
    for i in range(len(b)):
        if b[i] < a[i]:
            false_count += 1
print(f"\tIn {false_count} / {total} experiments resampling *increased* the mean error")


#%% Do plots

# First plot all of the locations in the grids for which we have data (these are
# not necessarily evenly spaced).
# See here for instructions on how to do heatmap with irregularly spaced data:
# https://scipy-cookbook.readthedocs.io/items/Matplotlib_Gridding_irregularly_spaced_data.html

# Define the grid.
# First need the points that the observations are taken at
x, y = [], []
for i in range(len(agents)):
    for j in range(len(particles)):
        x.append(agents[i])
        if uselog:
            y.append(np.log(particles[j]))
        else:
            y.append(particles[j])
x = np.array(x)
y = np.array(y)

# Now the grid to interpolate over (used later)
xi = np.linspace(0,max(agents)   ,100)
yi = None
if uselog:
    yi = np.geomspace(0.01,np.log(max(particles)),100)
else:
    yi = np.linspace(0,max(particles),100)

# Plot the point locations
plt.figure(0)
plt.scatter(x=x, y=y, marker='o',c='black',s=2)
plt.xlabel('Agents')
plt.ylabel('Log Particles' if uselog else 'Particles')
plt.title("Sampling locations of experiments")


# Can restrict the number of agents and/or particles to look at in the plots
# (note this is an index into the actual number of agents/particles)
#min_particles = particles.index(10)
#min_particles = particles.index(10) # 1 means include all particles
#max_agents = agents.index(agents[len(agents)-1]) # agents[len(agents)-1] gives all agents


# Define the plots so that they can be plotted in a loop
plot_def = {
    "Min mean error" : min_mean_err,
    "Max mean error" : max_mean_err,
    "Avg mean error" : ave_mean_err,
    "Median mean error":  med_mean_err,
    "Min abs error"  : min_abs_err,
    "Max abs error"  : max_abs_err,
    "Avg abs error"  : ave_abs_err,
    "Median abs error": med_abs_err,
    "Min variance"   : min_var,
    "Max variance"   : max_var,
    "Avg variance"  : ave_var,
    "Median variance"  : med_var
    }

for before in [0,1]:
    for i, (title, d) in enumerate(plot_def.items()): # d is the data to plot (an np array)
        
        d = d[before] # d is a tuple (data before and data after)
        
        # The value of the statistic being visualised (e.g. mean_error) as a long list
        z = []
        for i in range(len(agents)):
            for j in range(len(particles)):
                z.append(d[j,i])
        assert len(x) == len(y) and len(x) == len(z)
        z = np.array(z)
        # Grid the data
        zi = griddata(points=(x, y),
                      values=z,
                      xi=(xi[None,:], yi[:,None]),
                      method=interpolate_method)
    
        plt.figure(i+1) # (+1 because there was a figure before)
        cs1 = plt.contour( xi,yi,zi,8,linewidths=0.5,colors='k')
        cs2 = plt.contourf(xi,yi,zi,8,cmap=plt.cm.jet)
        plt.colorbar() # draw colorbar
        #plt.scatter(x,y,marker='o',c=[cs2.get_cmap()(val) for val in z], s=2)
        plt.scatter(x,y,marker='o',c=z, cmap=cs2.get_cmap(), s=5)
        plt.xlabel('Agents')
        plt.ylabel('Log Particles' if uselog else 'Particles')
        plt.title(title+" ({} resampling)".format('before' if before==0 else 'after') )
        plt.show()

print("Finished.")
