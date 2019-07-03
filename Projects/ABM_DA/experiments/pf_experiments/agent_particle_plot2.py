# -*- coding: utf-8 -*-
"""
Reads the out files that are created by StationSim-ARCExperiments.py

This version accounts for two types of results being written for each 
experiment (before and after reweighting) and also doesn't assume that the 
agents v.s. particles heat map is evenly spaced.

Created on Thu Apr  4 14:09:12 2019

@author: medkmin (adapted by Nick Malleson)
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
path = os.path.join(sys.path[0], "results","")


# Need to set the number of particles and agents used in the experiments
# (these are set in StationSim-ARCExperiments.py)
# TODO: work these out from the results file names
particles  = list([1] + list(range(10, 50, 10)) + list(range(100, 501, 100)) + list(range(1000, 2001, 500)) + list(range(3000, 10001, 1500)) + [10000])
agents = list(range(1, 21, 3))

# Use log on y axis?
uselog = True

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


# Do this whole thing twice, calculing errors before and then after resampling
# (there's a neater way to do this, but a for loop will do for now)

for before in [0,1]:

    print("Calculating errors {} resampling:".format('before' if before == 0 else 'after'))

    # Errors are a matrix of particles * agents

    min_mean_err = np.zeros(shape=(len(particles),len(agents)))
    max_mean_err = np.zeros(shape=(len(particles),len(agents)))
    ave_mean_err = np.zeros(shape=(len(particles),len(agents)))
    med_mean_err = np.zeros(shape=(len(particles),len(agents))) # Median of the mean errors
    min_abs_err = np.zeros(shape=(len(particles),len(agents)))
    max_abs_err = np.zeros(shape=(len(particles),len(agents)))
    ave_abs_err = np.zeros(shape=(len(particles),len(agents)))
    med_abs_err = np.zeros(shape=(len(particles),len(agents)))
    min_var = np.zeros(shape=(len(particles),len(agents)))
    max_var = np.zeros(shape=(len(particles),len(agents)))
    ave_var = np.zeros(shape=(len(particles),len(agents)))
    med_var = np.zeros(shape=(len(particles),len(agents)))

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
            else:
                sys.exit("Current file shape ({}) does not match the previous one ({}). Current file is: \n\t{}. \n\tNot continuing".format(
                        str(data.shape), str(data_shape), f  ))

        #data.iloc[:,0] = pd.to_numeric(data.iloc[:,0]) # Not sure why this was necessary

        # Filter by whether errors are before or after
        data = data[ data.loc[:,'Before_resample?'] == before]

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

        data_mean =  data.mean() # Calculate the mean of all columns
        data_median = data.median() # and also sometimes use median

        min_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Mean_errors']
        max_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Mean_errors']
        ave_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_mean_errors']
        med_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_median['Average_mean_errors']
        min_abs_err [particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Absolute_errors']
        max_abs_err [particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Absolute_errors']
        ave_abs_err [particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_Absolute_errors']
        med_abs_err [particles.index(particle_num),agents.index(agent_num)] = data_median['Average_Absolute_errors']
        min_var     [particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_variances']
        max_var     [particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_variances']
        ave_var     [particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_variances']
        med_var     [particles.index(particle_num),agents.index(agent_num)] = data_median['Average_variances']

    print("...finished reading {} files".format(len(files)))


    # There will never be zero error, so replace 0s with NA
    data[data == 0] = np.nan

    #%% Plot full data

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

    for i, (title, d) in enumerate(plot_def.items()): # d is the data to plot (an np array)
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
                      method='nearest')

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
        if title=="Median abs error":
            print("HERE")

print("Finished.")