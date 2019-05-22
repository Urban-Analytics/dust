# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:09:12 2019

@author: medkmin
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Needs to be set to location of results
path = 'M:\Particle Filter\Model Results\HPC results\With noise = 10'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.csv' in file:
            files.append(os.path.join(r, file))
    
particles = [1] + list(range(10,1010,10))
agents = [1] + list(range(10,310,10))

min_mean_err = np.zeros(shape=(len(particles),len(agents)))
max_mean_err = np.zeros(shape=(len(particles),len(agents)))
ave_mean_err = np.zeros(shape=(len(particles),len(agents)))
min_abs_err = np.zeros(shape=(len(particles),len(agents)))
max_abs_err = np.zeros(shape=(len(particles),len(agents)))
ave_abs_err = np.zeros(shape=(len(particles),len(agents)))
min_var = np.zeros(shape=(len(particles),len(agents)))
max_var = np.zeros(shape=(len(particles),len(agents)))
ave_var = np.zeros(shape=(len(particles),len(agents)))

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

plt.figure(3)
plt.imshow(ave_mean_err, aspect = 'auto', origin = 'lower', extent = [1,300,10,100])
plt.xlabel('Agents')
plt.ylabel('Particles')
plt.title('Average Mean Error')
plt.colorbar()

#plt.figure(4)
#plt.imshow(min_abs_err, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Min Absolute Error')
#plt.colorbar()
#
#plt.figure(5)
#plt.imshow(max_abs_err, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Max Absolute Error')    
#plt.colorbar()

#ave_abs_err = ave_abs_err[1:10,:]
        
plt.figure(6)
plt.imshow(ave_abs_err, aspect = 'auto', origin = 'lower', extent = [1,300,10,100])
plt.xlabel('Agents')
plt.ylabel('Particles')
plt.title('Average Absolute Error')        
plt.colorbar()
    
#plt.figure(7)
#plt.imshow(min_var, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Min Variance')
#plt.colorbar()
#
#plt.figure(8)
#plt.imshow(max_var, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Max Variance')
#plt.colorbar()
#
#plt.figure(9)
#plt.imshow(ave_var, aspect = 'auto', origin = 'lower', extent = [1,300,1,1000])
#plt.xlabel('Agents')
#plt.ylabel('Particles')
#plt.title('Average Variance')            
#plt.colorbar()    
  
    
