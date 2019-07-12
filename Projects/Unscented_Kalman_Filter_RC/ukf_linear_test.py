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
from scipy.spatial import distance as dist
from matplotlib.patches import Ellipse


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
    
    n_agents = 1
    np.random.seed(8)
    n_steps=5
    ys = np.zeros((n_steps+1,2*n_agents))
    for i in range(1,int(ys.shape[0])):
        ys[i,:] = ys[i-1,:]+ np.ones(2*n_agents) + 0.25*np.random.normal(size=ys.shape[1])

    ukf_params = {
            "a":10,#alpha between 1 and 1e-4 typically
            "b":2,#beta set to 2 for gaussian 
            "k":0,#kappa usually 0 for state estimation and 3-dim(state) for parameters
            }
    
    Q = np.eye(2*n_agents)
    R = np.eye(2*n_agents)
    P = np.array([[1,0],[0,1]])

    xs = []
    Ps=[]
    xs.append(ys[0,:])
    u = ukf(ukf_params,ys[0,:],fx,hx,P,Q,R)
    for j in range(1,n_steps+1):
        u.predict()
        y = ys[j,:]
        u.update(y)
        xs.append(u.x)
        Ps.append(u.P)
    
        
    res = np.array(xs)-np.array(ys)
    distances = np.zeros((n_steps,n_agents))
    for i in range(n_agents):
        for j in range(n_steps+1):
            distances[j:,i] = dist.euclidean(res[j,(2*i)],res[j,(2*i)+1])
    import matplotlib.pyplot as plt
    xs=np.array(xs)[1:,:]
    ys=np.array(ys)[1:,:]
    
def plots():    
    
    f = plt.figure()
    agent_means = np.mean(np.abs(distances),axis=0)
    plt.hist(np.sqrt(agent_means[0::2]**2+agent_means[1::2]**2))
    plt.title("MAE histogram per agent")
    
    f=plt.figure()
    time_means = np.mean(distances,axis=1)
    plt.plot(time_means)
    plt.title("MAE over time across agents")

    f=plt.figure()
    plt.plot(xs[:,::2],xs[:,1::2])
    plt.title("kf paths")

    f=plt.figure()
    plt.plot(ys[:,::2],ys[:,1::2])
    plt.title("true paths")
    
    
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]



    nstd = 1
    f = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    sigmas = u.Sigmas(u.x,u.P)
    ax.scatter(sigmas[0,:],sigmas[1,:])
    ax.scatter(ys[4,0],ys[4,1],color="r")
    vals, vecs = eigsorted(u.P)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(u.x[0],u.x[1]),width=w, height=h,angle=theta, color='white')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    
    vals, vecs = eigsorted(np.eye(2)*5/8)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell2 = Ellipse(xy=(ys[4,0],ys[4,1]),width=w, height=h,angle=theta, color='red')
    ell2.set_facecolor('none')
    ax.add_artist(ell2)
    
    
    ax.set_xlim([3,7])
    ax.set_ylim([5,10])
    plt.show()
    
    
def ukf_ex(a,b,k):
    def fx(x):
        x+= np.ones(x.shape)
        return x
    
    def hx(x):
        return x
    ukf_params = {

        "a":a,
        "b":b,
        "k":k,
        "d_rate" : 10, 

        }
    
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    print(ukf_params)
    n_agents = 1
    np.random.seed(12)
    n_steps=5
    ys = np.zeros((n_steps+1,2*n_agents))
    for i in range(1,int(ys.shape[0])):
        ys[i,:] = ys[i-1,:]+ np.ones(2*n_agents) + 0.25*np.random.normal(size=ys.shape[1])


    Q = np.eye(2*n_agents)
    R = np.eye(2*n_agents)
    P = np.array([[1,0],[0,1]])

    xs = []
    Ps=[]
    xs.append(ys[0,:])
    u2 = ukf(ukf_params,ys[0,:],fx,hx,P,Q,R)
    for j in range(1,n_steps+1):
        u2.predict()
        y = ys[j,:]
        u2.update(y)
        xs.append(u2.x)
        Ps.append(u2.P)
    
        
    res = np.array(xs)-np.array(ys)
    distances = np.zeros((n_steps,n_agents))
    for i in range(n_agents):
        for j in range(n_steps+1):
            distances[j:,i] = dist.euclidean(res[j,(2*i)],res[j,(2*i)+1])
    import matplotlib.pyplot as plt
    xs=np.array(xs)[1:,:]
    ys=np.array(ys)[1:,:]
    
    nstd = 1
    f = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, aspect='equal')
    sigmas = u2.Sigmas(u2.x,u2.P)
    ax.scatter(sigmas[0,:],sigmas[1,:],color = "yellow")
    ax.scatter(ys[4,0],ys[4,1],color="r")
    vals, vecs = eigsorted(u2.P)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell = Ellipse(xy=(u2.x[0],u2.x[1]),width=w, height=h,angle=theta, color="cyan",linewidth=4)
    ell.set_facecolor('none')
    ax.add_artist(ell)
    
    vals, vecs = eigsorted(np.eye(2)*5/16)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell2 = Ellipse(xy=(ys[4,0],ys[4,1]),width=w, height=h,angle=theta, color='orangered',linewidth=4)
    ell2.set_facecolor('none')
    ax.add_artist(ell2)
    
    
    ax.set_xlim([3,7])
    ax.set_ylim([3,6])
    plt.savefig(f"ukf{a}.pdf")


def ensemble_ex(n):
    np.random.seed(12)
    pf_ys=[]
    n_particles = n
    n_steps=5
    n_agents=1
    truth_ys  = np.zeros((n_steps+1,2*n_agents))
    for i in range(1,int(truth_ys.shape[0])):
        truth_ys[i,:] = truth_ys[i-1,:]+ np.ones(2*n_agents) + 0.5*np.random.normal(size=truth_ys.shape[1])

    for _ in range(n_particles):
        ys = np.zeros((n_steps+1,2*n_agents))
        for i in range(1,int(ys.shape[0])):
            ys[i,:] = ys[i-1,:]+ np.ones(2*n_agents) + 0.5*np.random.normal(size=ys.shape[1])
        pf_ys.append(ys[-1,:])

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    pf_ys=np.vstack(pf_ys)
    x = pf_ys[:,0]
    y= pf_ys[:,1]

    distances = np.sqrt((x-truth_ys[-1,0])**2+(y-truth_ys[-1,1])**2)
    index = np.where(distances<2)
    x= x[index]
    y=y[index]
    nstd=1
    f = plt.figure(figsize=(12,12))
    ax = plt.subplot(111, aspect='equal')        
    ax.scatter(x,y,color="yellow",label="Particles")
    ax.scatter(truth_ys[-1,0],truth_ys[-1,1], color='orangered',label = "Truth")
    vals, vecs = eigsorted(np.cov(x,y))
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ax.scatter(np.mean(x),np.mean(y), color='cyan',label="PF Truth Estimate")
    ell = Ellipse(xy=(np.mean(x),np.mean(y)),width=w, height=h,angle=theta, color='cyan',linewidth=4.0)
    ell.set_facecolor('none')
    ax.add_artist(ell)

    vals, vecs = eigsorted(np.eye(2)*5/4)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    w, h = 2 * nstd * np.sqrt(vals)
    ell2 = Ellipse(xy=(truth_ys[-1,0],truth_ys[-1,1]),width=w, height=h,angle=theta, color='orangered',linewidth=4.0)
    ell2.set_facecolor('none')
    ax.add_artist(ell2)

    ax.set_xlim([3,7])
    ax.set_ylim([3,6])
    plt.legend()
    plt.savefig(f"pf{n}.pdf")
    
ukf_ex(0.1,2,0)
ukf_ex(1,2,0)
ukf_ex(1.5,2,0)

ensemble_ex(10)
ensemble_ex(1000)
ensemble_ex(100000)


    