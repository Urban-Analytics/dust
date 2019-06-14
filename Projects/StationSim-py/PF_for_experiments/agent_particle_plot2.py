# -*- coding: utf-8 -*-
"""
Reads the out files that are created by StationSim-ARCExperiments.py
This version accounts for two types of results being written for each experiment: before and after reweighting.

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

# Needs to be set to location of results
#path = 'M:\Particle Filter\Model Results\HPC results\With noise = 10'
path = "/Users/nick/gp/dust/Projects/StationSim-py/PF_for_experiments/results/FewAgentExperiments/"

# Model the errors before or after resampling? 0 = before, 1= after
before = 0
print("Calculating errors {} resampling:".format('before' if before==0 else 'after'))

# Need to set the number of particles and agents used in the experiments 
# (these are set in StationSim-ARCExperiments.py)
particles  = list(range(1,49,1))  + list(range(50,501,50)) + list(range(600,2001,100)) + list(range(2500,4001,500))
agents = list(range(1,21,1))

if not os.path.isdir(path):
    sys.exit("Directory '{}' does not exist".format(path))

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
            
if len(files) == 0:
    sys.exit("Found no files in {}, can't continue".format(path) )
else:
    print("Found {} files".format(len(files)))
    


# Errors are a matrix of particles * agents

min_mean_err = np.zeros(shape=(len(particles),len(agents)))
max_mean_err = np.zeros(shape=(len(particles),len(agents)))
ave_mean_err = np.zeros(shape=(len(particles),len(agents)))
min_abs_err = np.zeros(shape=(len(particles),len(agents)))
max_abs_err = np.zeros(shape=(len(particles),len(agents)))
ave_abs_err = np.zeros(shape=(len(particles),len(agents)))
min_var = np.zeros(shape=(len(particles),len(agents)))
max_var = np.zeros(shape=(len(particles),len(agents)))
ave_var = np.zeros(shape=(len(particles),len(agents)))

print("Reading files....",)
for f in files:

    file = open(f,"r").read()
    data = pd.read_csv(f, header = 2).replace('on',np.nan)
    #data.iloc[:,0] = pd.to_numeric(data.iloc[:,0]) # Not sure why this was necessary
    
    # Filter by whether errors are before or after
    data = data[ data.loc[:,'Before_resample?'] == before]
    
    particle_num = int(re.findall('particles\': (\d{1,4})',file)[0])
    agent_num = int(re.findall('pop_total\': (\d{1,3})',file)[0])
    data_mean = data.mean() # Calculate the mean of all columns
    
    min_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Mean_errors']
    max_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Mean_errors']
    ave_mean_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_mean_errors']
    min_abs_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_Absolute_errors']
    max_abs_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_Absolute_errors']
    ave_abs_err[particles.index(particle_num),agents.index(agent_num)] = data_mean['Average_Absolute_errors']
    min_var[particles.index(particle_num),agents.index(agent_num)] = data_mean['Min_variances']
    max_var[particles.index(particle_num),agents.index(agent_num)] = data_mean['Max_variances']
    ave_var[particles.index(particle_num),agents.index(agent_num)] = data.mean()['Average_variances']
    
print("...finished reading")
    
#%% Plot full data

# Can restrict the number of agents and/or particles to look at in the plots
# (note this is an index into the actual number of agents/particles)
min_particles = particles.index(10)
max_agents = agents.index(agents[len(agents)-1]) # agents[len(agents)-1] gives all agents


# Define the plots so that they can be plotted in a loop
plot_def = {
    "Min mean error" : min_mean_err,
    "Max mean error" : max_mean_err,
    "Avg mean error" : ave_mean_err, 
    "Min abs error"  : min_abs_err,
    "Max abs error"  : max_abs_err,
    "Avg abs error"  : ave_abs_err,
    "Min variance"   : min_var,
    "Max variance"   : max_var,
    "Avg variance"  : ave_var
    }

for i, (title, data) in enumerate(plot_def.items()):
    plt.figure(i)
    plt.imshow( data[min_particles: , :max_agents ], aspect = 'auto', origin = 'lower', 
           extent = [min(agents),agents[max_agents], particles[min_particles], max(particles)])
    plt.xlabel('Agents')
    plt.ylabel('Particles')
    plt.title(title+" ({} resampling)".format('before' if before==0 else 'after') )
    plt.colorbar()


