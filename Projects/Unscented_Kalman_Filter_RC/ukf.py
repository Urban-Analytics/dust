"""
generalised ukf class for (hopefully) any state space
"""
import numpy as np
from choldate import cholupdate,choldowndate
#"pip install git+git://github.com/jcrudy/choldate.git"
"""
for cholesky update/downdate.
see cholupdate matlab equivalent not converted into numpy.linalg from LAPACK yet (probably never will be)
but I found this nice package in the meantime.
"""
#!! look into **kwargs functionality to generalise to any F/H
class ukf:
    
    def __init__(self,srukf_params,init_x,fx,hx,P,Q,R):
        """
        x - state
        n - state size 
        P - state covariance
        fx - transition function
        hx - measurement function
        lam - lambda paramter
        g - gamma parameter
        wm/wc - unscented weights for mean and covariances respectively.
        Q,R -noise structures for fx and hx
        xs,Ps - lists for storage
        """
        
        #init initial state
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = P
        #self.P = np.linalg.cholesky(self.x)
        self.fx=fx
        self.hx=hx
        
        #init further parameters based on a through el
        self.lam = srukf_params["a"]**2*(self.n+srukf_params["k"]) - self.n #lambda paramter calculated viar
        self.g = np.sqrt(self.n+self.lam) #gamma parameter

        
        #init weights based on paramters a through el
        main_weight =  1/(2*(self.n+self.lam))
        self.wm = np.ones(((2*self.n)+1))*main_weight
        self.wm[0] *= 2*self.lam
        self.wc = self.wm.copy()
        self.wc[0] += (1-srukf_params["a"]**2+srukf_params["b"])

    
            
        self.Q=Q
        self.R=R

        self.xs = []
        self.Ps = []

    def Sigmas(self,mean,S):
        """sigma point calculations based on current mean x and  UT (upper triangular) 
        decomposition S of covariance P"""
        
     
        sigmas = np.ones((self.n,(2*self.n)+1)).T*mean
        sigmas=sigmas.T
        sigmas[:,1:self.n+1] += self.g*S #'upper' confidence sigmas
        sigmas[:,self.n+1:] -= self.g*S #'lower' confidence sigmas
        return sigmas 

    def predict(self,**fx_args):
        """
        - calculate sigmas using prior mean and UT element of covariance S
        - predict interim sigmas X for next timestep using transition function Fx
        - predict unscented mean for next timestep
        - calculate interim S using concatenation of all but first column of Xs
            and square root of process noise
        - cholesky update to nudge on unstable 0th row
        - calculate futher interim sigmas using interim S and unscented mean
        """
        #calculate NL projection of sigmas
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.P)) #calculate current sigmas using state x and UT element S
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
        wnl_sigmas = nl_sigmas*self.wm
            
        xhat = np.sum(wnl_sigmas,axis=1)#unscented mean for predicitons
        
        """
        should be a faster way of doing this
        covariance estimation for prior P as a sum of the outer products of 
        (sigmas - unscented mean) weighted by wc
        """
        Pxx =  self.wc[0]*np.outer((nl_sigmas[:,0].T-xhat),(nl_sigmas[:,0].T-xhat))+self.Q
        for i in range(1,len(self.wc)): 
            Pxx += self.wc[i]*np.outer((nl_sigmas[:,i].T-self.x),nl_sigmas[:,i].T-xhat)
            
        self.P = Pxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z,**hx_args):     
        """
        Does numerous things in the following order
        - calculate interim sigmas using Sxx and unscented mean estimate
        - calculate measurement sigmas Y = h(X)
        - calculate unscented mean of Ys
        - calculate qr decomposition of concatenated columns of all but first Y scaled 
            by w1c and square root of sensor noise to calculate interim S
        - cholesky update to nudge interim S on potentially unstable 0th 
            column of Y
        - calculate sum of scaled cross covariances between Ys and Xs Pxy
        - calculate kalman gain
        - calculate x update
        - calculate S update
        """
        
        """
        posterior sigmas using above unscented interim estimates for x and P
        """
        sigmas = self.Sigmas(self.x,np.linalg.cholesky(self.P)) #update using Sxx and unscented mean
        nl_sigmas = np.apply_along_axis(self.hx,0,sigmas)
        wnl_sigmas = nl_sigmas*self.wm

        """
        unscented estimate of posterior mean using said posterior sigmas
        """
        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        
        
        "similar weighted estimates as Pxx for cross covariance and posterior covariance"
        Pyy =  self.wc[0]*np.outer((nl_sigmas[:,0].transpose()-yhat),(nl_sigmas[:,0].transpose()-yhat))+self.R
        for i in range(1,len(self.wc)):
            Pyy += self.wc[i]*np.outer((nl_sigmas[:,i].transpose()-yhat),(nl_sigmas[:,i].transpose()-yhat))
        

        Pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),(nl_sigmas[:,0].transpose()-yhat))
        for i in range(1,len(self.wc)):
            Pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),(nl_sigmas[:,i].transpose()-yhat))
            
        "kalman gain"
        K = np.matmul(Pxy,np.linalg.inv(Pyy))
 
        #update xhat
        self.x += np.matmul(K,(z-yhat))
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        Pxx = self.P
        Pxx -= np.matmul(K,np.matmul(Pyy,K.T))
        
        self.P = Pxx
        self.Ps.append(self.P)
        self.xs.append(self.x)
        
        
        
    def batch(self):
        """
        batch function maybe build later
        """
        return
