# -*- coding: utf-8 -*-
"""
Created on Thu May 23 11:13:26 2019

@author: RC

first attempt at a square root UKF class
class built into 5 steps
-init
-Prediction SP generation
-Predictions
-Update SP generation
-Update

SR filter generally the same as regular UKF for efficiency 
but more numerically stable wrt rounding errors 
and preserving PSD covariance matrices

based on
citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
"""



import numpy as np
from ukf import ukf



if __name__ == "__main__":
    """
            a - alpha scaling parameter determining how far apart sigma points are spread. Typically between 1e-4 and 1
            b - beta scaling paramater incorporates prior knowledge of distribution of state space. b=2 optimal for gaussians
            k - kappa scaling parameter typically 0 for state space estimates and 3-dim(x) for parameter estimation
            init_x- initial state space
    """

    
    "here is a very simply test for srukf using 10 agents and 10 time steps with linear motion for each "
    
    def fx(x):
        x+= np.ones(x.shape)
        return x
    
    def hx(x):
        return x
    
    n_agents = 250
    ys = np.zeros((10,2*n_agents))
    for i in range(1,int(ys.shape[0])):
        ys[i,:] = ys[i-1,:]+ np.ones(2*n_agents) + 0.5*np.random.normal(size=ys.shape[1])

    ukf_params = {
            "a":0.1,#alpha between 1 and 1e-4 typically
            "b":2,#beta set to 2 for gaussian 
            "k":0,#kappa usually 0 for state estimation and 3-dim(state) for parameters
            }
    
    Q = np.eye(2*n_agents)
    R = np.eye(2*n_agents)
    
    xs = []
    Ps=[]
    xs.append(ys[0,:])
    u = ukf(ukf_params,ys[0,:],fx=fx,hx=hx,Q=Q,R=R)

    for j in range(1,10):
       
        
        u.predict()
        y = ys[j,:]
        u.update(y)
        xs.append(u.x)
        Ps.append(u.P)
        
        
        
    res = np.array(xs)-np.array(ys)
    
    import matplotlib.pyplot as plt
    xs=np.array(xs)
    ys=np.array(ys)
    
    f = plt.figure()
    agent_means = np.mean(np.abs(res),axis=0)
    plt.hist(np.sqrt(agent_means[0::2]**2+agent_means[1::2]**2))
    plt.title("MAE histogram per agent")
    
    f=plt.figure()
    time_means = np.mean(res,axis=1)
    plt.plot(np.sqrt(time_means[0::2]**2+time_means[1::2]**2))
    plt.title("MAE over time across agents")

    f=plt.figure()
    plt.plot(xs[:,::2],xs[:,1::2])
    plt.title("kf paths")

    f=plt.figure()
    plt.plot(ys[:,::2],ys[:,1::2])
    plt.title("true paths")
    
