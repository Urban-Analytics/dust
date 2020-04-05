# -*- coding: utf-8 -*-
"""
This function process the output data from the Social Force Model ABM into aggregated data for Emulators

@author: Minh Kieu, Leeds, Aug 2019
"""

import os
import numpy as np
print('running..')
door_in=10  #coordinate of the entrance door
door_out = 100 #coordinate of the exit door
tolerance = 0.5 #equals half of the width of the door
cell_size = 10 #the width of each cell, the height of the cell will aways be the heigth of the corridor

maxT = 300 #maximum time in s
stepT = 15 #each interval, in s

#indexes of the raw file from JCrowdSimulator
TimeIndex = 0
AgentID = 1
XPosIndex = 2
YPosIndex = 3
TotalForceIndex=4
CurrentSpeedIndex =5
MaxSpeedIndex=6
TimeThroughCorridorIndex= 7


#get a list of data files
#os.chdir("/Users/geomik/Dropbox/Minh_UoL/DA/Emulators/data")
os.chdir("/Users/MinhKieu/Documents/Research/Emulator_ABM/data/raw")
#prefixed = [filename for filename in os.listdir('.') if filename.startswith("HPP-cone-")]
prefixed = [filename for filename in os.listdir('.') if filename.startswith("HPP-cone-")]

#collect the data from each file
data = np.zeros(2*int((door_out-door_in)/cell_size)+2)
for f in prefixed:
    
    #Load data
    print(f)
    df = np.genfromtxt(f, delimiter=',', skip_header=1)
    df=df[df[:,XPosIndex]>10]
    df=df[df[:,XPosIndex]<100]

    
    #now loop through time
    for t in range(0,maxT*1000,stepT*1000):
        df_interval = df[(df[:,TimeIndex]>=t) & (df[:,TimeIndex]<t+stepT*1000),:]
        celldata = [t/1000]
        
        for c in range(0, int((door_out-door_in)/cell_size)):
            #print(c)
            flow_in = np.size(np.unique(df_interval[(df_interval[:,XPosIndex]>= door_in+c*cell_size-tolerance)& (df_interval[:,XPosIndex]< door_in+c*cell_size+tolerance),1]))
            celldata.append(flow_in)
            if flow_in ==0:
                celldata.append(0)        
            else: 
                #mean_speed = np.mean(df_interval[(df_interval[:,2]>= door_in+c*cell_size-tolerance)& (df_interval[:,2]< door_in+(c+1)*cell_size+tolerance),4])
                #celldata.append(mean_speed)
                #std_speed = np.std(df_interval[(df_interval[:,2]>= door_in+c*cell_size-tolerance)& (df_interval[:,2]< door_in+(c+1)*cell_size+tolerance),4])
                #celldata.append(std_speed)
                mean_force = np.mean(df_interval[(df_interval[:,XPosIndex]>= door_in+c*cell_size-tolerance)& (df_interval[:,XPosIndex]< door_in+(c+1)*cell_size+tolerance),TotalForceIndex])
                celldata.append(mean_force)
                #std_xforce = np.std(df_interval[(df_interval[:,2]>= door_in+c*cell_size-tolerance)& (df_interval[:,2]< door_in+(c+1)*cell_size+tolerance),5])
                #celldata.append(std_xforce)
                #celldata.extend([mean_xforce, mean_speed])
            #print(celldata)
        output = [np.mean(df_interval[:,TotalForceIndex])]
        #print(output)
        data = np.vstack((data,celldata+output))
        #data = np.vstack((data,celldata))

np.savetxt("agg-1-3-5-7-9.csv", data,fmt='%10.3f', delimiter=",")
#np.savetxt("val-15ped.csv", data,fmt='%10.3f', delimiter=",")
        
