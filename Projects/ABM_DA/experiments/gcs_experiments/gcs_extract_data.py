#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created: 23/04/2020
@author: patricia-ternes
'''

from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from operator import itemgetter
import os

'''
    To use the data is necessary organize the data.
    Below some organization suggestions.
'''
by_frame = False
by_agent = False
plot_trajectories = True


#open and read original file
file_name = "trajectoriesNew/trajectoriesNew.mat"
file = loadmat(file_name)
trajectories = file["trks"][0]
'''
Organize data by line. Each line has:
['agentID', 'x', 'y', 'time']
'''
data = []
for i in range (len(trajectories)):
    x, y, t = trajectories[i]
    x, y, t = x.flatten(), y.flatten(), t.flatten()
    y = -1.*y + 455. # because for some reason the data is upside down
    for j in range (len(x)):
        values = [i, x[j], y[j], t[j]]  # indices: ['agentID', 'x', 'y', 'time']
        data.append(values)
        
    if(plot_trajectories):
        '''
        plot all trajectories
        '''
        plt.plot (x, y,'-',)
if(plot_trajectories):
    save_file = 'all_trajectories.png'
    plt.savefig (save_file)
    plt.close()

if (by_frame):
    '''
        Organize by frame
    '''
    directory = 'frames'
    if not(os.path.exists(directory)):
        os.mkdir(directory)

    data1 = sorted(data, key=itemgetter(3))  # sorted by: [0 = 'agentID', 1 = 'x', 2 = 'y', 3 = 'time']

    frame = 1
    save_file = open(directory+'/frame_'+ str(frame) +'.dat', 'w')
    print('# agentID', 'x', 'y', file=save_file)
    for line in data1:
        if (line[3] == frame):
            print(line[0], line[1], line[2], file=save_file)
        else:
            save_file.close()
            frame += 1
            save_file = open(directory+'/frame_'+ str(frame) +'.dat', 'w')
            print(line[0], line[1], line[2], file=save_file)
    save_file.close()

if (by_agent):
    '''
        Organize by agent
    '''
    directory = 'agents'
    if not(os.path.exists(directory)):
        os.mkdir(directory)

    data1 = sorted(data, key=itemgetter(0))  # sorted by: [0 = 'agentID', 1 = 'x', 2 = 'y', 3 = 'time']

    agent = 0
    save_file = open(directory+'/agent_'+ str(agent) +'.dat', 'w')
    print('# time', 'x', 'y', file=save_file)
    for line in data1:
        if (line[0] == agent):
            print(line[3], line[1], line[2], file=save_file)
        else:
            save_file.close()
            agent += 1
            save_file = open(directory+'/agent_'+ str(agent) +'.dat', 'w')
            print(line[3], line[1], line[2], file=save_file)
    save_file.close()
