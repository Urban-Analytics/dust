def plot_station():
    # plotting the station walls
    a = [0, 0, 53, 53, 0]; b = [0, 50, 50, 0, 0]; plt.plot(a,b,'-', linewidth=2,color='k')

    # plotting the station gate 0
    plt.plot(0,20,'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 1
    plt.plot(10,50,'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 2
    plt.plot(43,50, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 3
    plt.plot(53,5, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 4
    plt.plot(53,17.5, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 5
    plt.plot(53,32.5, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 6
    plt.plot(53,45, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 7
    plt.plot(8.6,0, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 8
    plt.plot(20.2,0, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 9
    plt.plot(32.8,0, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    # plotting the station gate 10
    plt.plot(44.4,0, 'D',linewidth=2, color='lime', ms=20, markeredgecolor='k',markeredgewidth=2);

    plt.text(-1.5, -1.5, '0', fontsize=14)
    plt.text(-1-4.5, 50-.5, '50 m', fontsize=14)
    plt.text(53-2, -2, '53 m', fontsize=14)
    

import os
import numpy as np
import matplotlib.pyplot as plt

directory = 'agents_complete_trails/'
directory = './'
files = os.listdir(directory)

directory1 = 'trails/'
if not(os.path.exists(directory1)):
    os.mkdir(directory1)
    
for agent_file in files:
    file_name = directory + agent_file
    t, x, y = np.loadtxt(file_name,unpack=True)
    fig, ax = plt.subplots(num=None,figsize=(7.4, 7),dpi=128,facecolor='w',edgecolor='k')
    plot_station()
    x = x/14.0
    y = y/14.0
    plt.plot(x,y, 'k')
    plt.plot(x[0],y[0], 'o', ms=10, color='r')
    plt.axis('off')
    
    file_name =  directory1 + agent_file + '.png'
    plt.savefig(file_name, bbox_inches='tight')
    plt.close() 
