# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:09:12 2019

@author: medkmin
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
path = "/Users/nick/gp/dust/Projects/StationSim-py/Particle_Filter_Experiments/results/noise-02"

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
    
particles = [1] + list(range(10,1010,10)) # Particles 10 -> 1000
agents = [1] + list(range(10,310,10)) # Agents 10 -> 300

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
    data.iloc[:,0] = pd.to_numeric(data.iloc[:,0])
    
    particle_num = int(re.findall('particles\': (\d{1,4})',file)[0])
    agent_num = int(re.findall('pop_total\': (\d{1,3})',file)[0])
    
    min_mean_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[0]
    max_mean_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[1]
    ave_mean_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[2]
    min_abs_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[3]
    max_abs_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[4]
    ave_abs_err[particles.index(particle_num),agents.index(agent_num)] = data.mean()[5]
    min_var[particles.index(particle_num),agents.index(agent_num)] = data.mean()[6]
    max_var[particles.index(particle_num),agents.index(agent_num)] = data.mean()[7]
    ave_var[particles.index(particle_num),agents.index(agent_num)] = data.mean()[-1]
    
print("...finished reading")
    
#%% Plot full data

#plt.figure(1)
#plt.imshow(min_mean_err, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Min Mean Error')
#plt.colorbar()

#plt.figure(2)
#plt.imshow(max_mean_err, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Max Mean Error')
#plt.colorbar()

#ave_mean_err = ave_mean_err[1:10,:]

# Can restrict the number of agents and/or particles to look at in the plots
# (note this is an index into the actual number of agents/particles)
min_particles = particles.index(10)
max_agents = agents.index(50)


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
    "Avg variance:"  : ave_var
    }

for i, (title, data) in enumerate(plot_def.items()):
    plt.figure(i)
    plt.imshow( data[min_particles: , :max_agents ], aspect = 'auto', origin = 'lower', 
           extent = [min(agents),agents[max_agents], particles[min_particles], max(particles)])
    plt.xlabel('Agents')
    plt.ylabel('Particles')
    plt.title(title)
    plt.colorbar()



  
#%% Plot with few agents

p = 0.2 # Proportion agents to include (only the first x %)

fig, ax = plt.subplots()
plt.imshow(ave_mean_err[ : , 0:int(p*len(agents))], aspect = 'auto', origin = 'lower')
plt.xlabel('Agents')
ax.set_xticklabels(range(0,300,10))
plt.ylabel('Particles')
plt.title('Average Mean Error')
plt.colorbar()



plt.figure(3)
#plt.imshow(ave_mean_err, aspect = 'auto', origin = 'lower', extent = [1,300,10,100])
plt.imshow(ave_mean_err[ : , 0:int(p*len(agents))], aspect = 'auto', origin = 'lower', extent = [1,int(p*300),10,100])
plt.xlabel('Agents')
plt.ylabel('Particles')
plt.title('Average Mean Error')
plt.colorbar()
        
plt.figure(6)
plt.imshow(ave_abs_err[ : , 0:int(p*len(agents))], aspect = 'auto', origin = 'lower', extent = [1,int(p*300),10,100])
plt.xlabel('Agents')
plt.ylabel('Particles')
plt.title('Average Absolute Error')        
plt.colorbar()
