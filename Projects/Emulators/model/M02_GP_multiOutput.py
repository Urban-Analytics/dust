# -*- coding: utf-8 -*-
"""


"""

#!/usr/bin/env python3

from __future__ import print_function

# Adapted from
# https://stackoverflow.com/questions/13240633/matplotlib-plot-pulse-propagation-in-3d
# and rewritten to make it clearer how to use it on real data.

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
#from matplotlib import colors as mcolors

import os
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
os.chdir("/Users/MinhKieu/Documents/Research/Emulator_ABM")  #Mac


def draw_3d_2lines(data1, data2, ymin, ymax):
    '''Given data1 as a list of plots, each plot being a list
       of (x, y) vertices, generate a 3-d figure where each plot
       is shown as a translucent polygon.
       If line_at_zero, a line will be drawn through the zero point
       of each plot, otherwise the baseline will be at the bottom of
       the plot regardless of where the zero line is.
       Give also data2 for the second plot
    '''
    # add_collection3d() wants a collection of closed polygons;
    # each polygon needs a base and won't generate it automatically.
    # So for each subplot, add a base at ymin.

    for p in data1:
        p.insert(0, (p[0][0], 0))
        p.append((p[-1][0], 0))

    for p in data2:
        p.insert(0, (p[0][0], 0))
        p.append((p[-1][0], 0))

    facecolors = (1, 1, 1,0.5)
    edgecolors1 = (0, 0, 1, 1)
    edgecolors2 = (1, 0, 0, 0.8)

    poly1 = PolyCollection(data1,lw=1.5,linestyle = '-',
                          facecolors=facecolors, edgecolors=edgecolors1)

    poly2 = PolyCollection(data2,lw=1.5,linestyle = '--',
                          facecolors=facecolors, edgecolors=edgecolors2)

    zs = range(len(data1))
    plt.tight_layout(pad=2.0, w_pad=10.0, h_pad=3.0)

    ax.add_collection3d(poly1, zs=zs, zdir='y')
    ax.add_collection3d(poly2, zs=zs, zdir='y')

if __name__ == '__main__':

    #load simulated training data: #switch better same and narrow experiment
    data = np.genfromtxt('./data/agg-2-4-6.csv', delimiter=',')
    data = data[data[:,0]>100]
    #x: Time, count_gate x 9 (10 columns)
    #y: mean_speed,std_speed,mean_xforce,std_xforce  (4 columns)
    ## process data to make predictions: X= Xt, Y = [Speed,Social Force]
    x_train = data[:-1,(0,1,3,5,7,9,11,13,15,17)]
    y_train=data[1:,range(2,20,2)]
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(x_train)


    #Emulator: Gaussian Process Regression 
    import sys
    sys.path.append('/Users/MinhKieu/Documents/Research/Emulator_ABM/model')
    from MultiOutputGP import MultiOutputGP
    gp_multi = MultiOutputGP(X_train,y_train.T)
    gp_multi.learn_hyperparameters(n_tries=25)
    
    
    for ped in range(2,8):
    
        filename = './data/val-'+ str(ped) +'ped.csv'
        #print(filename)
        test_data = np.genfromtxt(filename, delimiter=',')
        #print(test_data)
        test_data = test_data[test_data[:,0]>100]
        #x: flow in at each `check point', or `cell': x_i^t
        #y: mean_speed,std_speed,mean_xforce,std_xforce
        x_test = test_data[:-1,(0,1,3,5,7,9,11,13,15,17)]
        X_test = sc.transform(x_test)
        y_test=test_data[1:,range(2,20,2)]
    
        #make predictions using the trained model
        y_pred, y_unc, _ = gp_multi.predict(X_test,do_unc=True, do_deriv=False)
    
        df_test = []
        df_pred = []
        for i in range(0,9):
            data_bin_test= []
            data_bin_pred= []
            #mean_y = np.mean(y_test[:,i])
            for j in range(0,17):
                data_bin_test.append((120+j*60,y_test[j][i]))
                data_bin_pred.append((120+j*60,y_pred[i][j]))
            df_pred.append(data_bin_pred)
            df_test.append(data_bin_test)
        
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(1,1,1, projection='3d')
        ax.view_init(35,-6)
        draw_3d_2lines(df_test,df_pred, 0, 1.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Cell')
        ax.set_zlabel('Social Force')

        ax.set_xlim3d(120, 1140)
        ax.set_ylim3d(0, 9)
        ax.set_title('Demand='+str(ped/2)+' pedestrian/s')

        if ped < 5:
            ax.set_zlim3d(0, 2)

        if ped == 5:
            ax.set_zlim3d(0, 5)
        if ped > 5:
            ax.set_zlim3d(0, 15)


        plotname = './figures/Cell-GP' + str(ped) + 'ped.pdf'
        plt.savefig(plotname)
        #plt.show()
        plt.clf()

import dill                            #pip install dill --user
filename = './data/Cell-GP.pkl'
dill.dump_session(filename)

# and to load the session again:
#dill.load_session(filename)