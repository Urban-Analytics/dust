#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 17:13:47 2020

@author: medrclaa
"""

from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#give a file name with the path to whereever you've saved the .mat file e.g.
file_name = "trajectoriesNew/trajectoriesnew.mat"
#load file with loadmat you get 4 values. you only really want trks
file = loadmat(file_name)
#get trks (trajectories from loadmat dictionary). Note 0 here as we have a useless index
#layer.
trajectories = file["trks"][0]

"""
for the record python hates this type of array. Its probaly much easier in the
long term to reformat this into pandas.

To parse the array we have a few levels of indexing. The first level gives you 
each agent in the system. For example, if we take trajectories[0] it gives us
a sub tuple containing all the info for the first agent (0th) in the list. They seem to 
be listed as they appear chronologically (not confirmed this).

Within this sub tuple for the 0th agent we have another layer with three elements 
namely the x positions, y positions and time the observation was recorded (in seconds?).
Im not 100% sure how this was done but I know they use a key point tracker (see the paper).

For an example of this, I plot the x and y positions of the first 1000 agents using
matplotlib. Note the y coordinates are upside down (i dont know why).
"""

for i in range(1000):
    "plot directly from .mat files."
    "pull x and y for ith agent. flip y axis right way up"
    x, y = trajectories[i][0], -1 * trajectories[i][1]
    plt.plot(x,y)
    plt.close()
    
"""in terms of pandas my thoughts are to use the time stamps as the index and 
have two columns for each agents' x and y positions"""

#initial dummy data frame.
traj_frame = pd.DataFrame()

#only compiling 100 agents here. change 100 to len(trajectories) for all agents
#this will take forever.
for i in range(100):
    #pull single agents trajectories. split into x, y, and times.
    traj = trajectories[i]
    x = traj[0].astype(int)
    y = traj[1].astype(int)
    times = traj[2].ravel()
    
    #column names. ith agent columns 0 and 1 for x and y positions.
    x_column = f"{i}_0"
    y_column = f"{i}_1"
    #build new dataframe with x, y, and time indices
    new_columns = pd.DataFrame(np.hstack([x,y]), columns = [x_column, y_column], index = times)
    #add new agent's data to main data frame
    traj_frame = pd.concat([traj_frame, new_columns], axis = 1, sort = False)

"""
Then to plot the pandas frame we simply plot every 2 
columns as x y positions in matplotlib.
"""
    
    
for i in range(int(traj_frame.shape[1]/2)):
    #find columns to parse
    x_column = f"{i}_0"
    y_column = f"{i}_1"
    #parse and plot columns
    plt.plot(traj_frame[x_column], traj_frame[y_column])