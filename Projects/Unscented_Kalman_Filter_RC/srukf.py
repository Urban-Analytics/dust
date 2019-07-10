import numpy as np
from choldate import cholupdate,choldowndate
import multiprocessing
#"pip install git+git://github.com/jcrudy/choldate.git"
"""
for cholesky update/downdate.
see cholupdate matlab equivalent not converted into numpy.linalg from LAPACK yet (probably never will be)
but I found this nice package in the meantime.
"""

class srukf:
    
    def __init__(self,srukf_params,init_x,fx,hx,Q,R):
        """
        x - state
        n - state size 
        P - Initial state covariance. This is generally the only 
        cholesky decomposition used to get a ballpark initial value.
        S- UT element of P. 
        fx - transition function
        hx - measurement function
        lam - lambda paramter
        g - gamma parameter
        wm/wc - unscented weights for mean and covariances respectively calculated using dimension size 
        and srukf parameters a,b and k.
        Q,R -noise structures for fx and hx
        sqrtQ,sqrtR - similar to P only square rooted once and propagated through S
        for efficiency
        xs,Ss - lists for storage
        """
        
        #init initial state
        self.x = init_x #!!initialise some positions and covariances
        self.n = self.x.shape[0] #state space dimension

        self.P = np.eye(self.n)
        #self.P = np.linalg.cholesky(self.x)
        self.S = np.linalg.cholesky(self.P)
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
        self.sqrtQ = np.linalg.cholesky(self.Q)
        self.sqrtR = np.linalg.cholesky(self.R)
         
        self.xs = []
        self.Ss = []

    def Sigmas(self,mean,S):
        """
        sigma point calculations based on current mean x and  UT (upper triangular) 
        decomposition S of covariance P
        !! this should probably be profiled to find the most efficient method as it is 
        called A LOT. this current method has proven more efficient than equivalent 
        for loops thus far and any further improvement would be massive.
        in:
            some mean and confidence stucutre S
        out: 
            sigmas based on above mean and S
        """
        
     
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
        - calculate qr decomposition of concatenated columns of all but the first sigma scaled 
            by the 1st (not 0th) wc and the square root of process noise (sqrtQ) 
            to calculate another interim S.
        - cholesky up(/down)date to nudge interim S on potentially unstable 0th 
            column of Y. if 0th wc weight +ve update else downdate
        - calculate futher interim sigmas using interim S and unscented mean
        in:
            -prior x and S
            -fx
            -wm,wc
        out:
            -interim x and S
        """
        "nl_sigmas calculated using either apply along axis or multithreading"
        #calculate NL projection of sigmas
        sigmas = self.Sigmas(self.x,self.S) #calculate current sigmas using state x and UT element S
        nl_sigmas = np.apply_along_axis(self.fx,0,sigmas)
        #p = multiprocessing.Pool()
        #nl_sigmas = np.vstack(p.map(self.fx,[sigmas[:,j] for j in range(sigmas.shape[1])])).T
        #p.close()
        wnl_sigmas = nl_sigmas*self.wm
            
        xhat = np.sum(wnl_sigmas,axis=1)#unscented mean for predicitons
        
        
        """
        a qr decompositions of the compound matrix contanining the weighted predicted sigmas points
        and the matrix square root of the additive process noise.

        the qr decomposition is wierdly worded in the reference paper here and 
        i have written this to follow it as to avoid confusion mainly with myself.
        in particular np.linalg.qr solves for a different qr decomposition form
        A^T = QR (as opposed to A=QR). this gives R as a pseudo UT mxn (m>n)** rectangular matrix.
        from this only bottom n rows are take as an actual UT matrix used as an interim S
        ** m sigmas points, n state space dimension
        """
        
        Pxx =np.vstack([np.sqrt(self.wc[1])*(nl_sigmas[:,1:].T-xhat),self.sqrtQ])
        Sxx = np.linalg.qr(Pxx,mode="r")
        Sxx = Sxx[Sxx.shape[0]-len(self.x):,:]
        "up/downdating as necessary depending on sign of first covariance weight"
        u =  np.sqrt(np.sqrt(np.abs(self.wc[0])))*(nl_sigmas[:,0]-xhat)
        if self.wc[0]>0:    
            cholupdate(Sxx,u)
        if self.wc[0]<0:    
            choldowndate(Sxx,u)   
            
        self.S = Sxx #update Sxx
        self.x = xhat #update xhat
    
    def update(self,z):     
        """
        Does numerous things in the following order
        - calculate interim sigmas using Sxx and unscented mean estimate
        - calculate measurement sigmas Y = h(X)
        - calculate unscented mean of Ys
        - calculate qr decomposition of concatenated columns of all but the first sigma scaled 
            by the 1st (not 0th) wc and the square root of sensor noise (sqrtR) to calculate another interim S
        - cholesky up(/down)date to nudge interim S on potentially unstable 0th 
            column of Y. if 0th wc weight +ve update else downdate
        - calculate sum of scaled cross covariances between Ys and Xs Pxy
        - calculate kalman gain through double back propagation 
        (uses moore-penrose pseudo inversion to improve stability in inverting near singular matrix)
        - calculate x update
        - calculate posterior S by cholesky downdating interim S Sxx on each column of matrix U
        
         in:
            -interim x and S
            -hx
            -wm,wc
        out:
            -posterior x and S
        """
        sigmas = self.Sigmas(self.x,self.S) #update using Sxx and unscented mean
        nl_sigmas = np.apply_along_axis(self.hx,0,sigmas)
        wnl_sigmas = nl_sigmas*self.wm

        yhat = np.sum(wnl_sigmas,axis=1) #unscented mean for measurements
        """
        a qr decompositions of the compound matrix contanining the weighted measurement sigmas points
        and the matrix square root of the additive sensor noise.
        
        same wierd formatting as Sxx above.
        """
        Pyy =np.vstack([np.sqrt(self.wc[1])*(nl_sigmas[:,1:].T-yhat),self.sqrtR])
        Syy = np.linalg.qr(Pyy,mode='r')
        Syy = Syy[Syy.shape[0]-len(z):,:]

        u =  np.sqrt(np.sqrt(np.abs(self.wc[0])))*(nl_sigmas[:,0]-yhat)
        #fourth root squares back to square root for outer product uu.T
        
        if self.wc[0]>0:    
            cholupdate(Syy,u)
        if self.wc[0]<0:    
            choldowndate(Syy,u)   
        
        #!! do this with quadratic form may be much quicker
        Pxy =  self.wc[0]*np.outer((sigmas[:,0].T-self.x),(nl_sigmas[:,0].T-yhat))
        for i in range(1,len(self.wc)):
            Pxy += self.wc[i]*np.outer((sigmas[:,i].T-self.x),(nl_sigmas[:,i].T-yhat))
            
        "line 1 is standard matrix inverse. generally slow for large spaces"
        "lines 2-3 are double back prop avoiding true inversion and much quicker/stable."
        #K = np.matmul(Pxy,np.linalg.inv(np.matmul(Syy,Syy.T)))
        #K = np.dot(Pxy,np.linalg.pinv(Syy.T))
        #K = np.dot(K,np.linalg.pinv(Syy))
        
        #double back prop
        K = np.linalg.lstsq(Syy,Pxy.T,rcond=1e-8)[0].T
        K = np.linalg.lstsq(Syy.T,K.T,rcond=1e-8)[0].T

        U = np.matmul(K,Syy)
        
        #update xhat
        self.x =self.x + np.matmul(K,(z-yhat))
        
        "U is a matrix (not a vector) and so requires dim(U) updates of Sxx using each column of U as a 1 step cholup/down/date as if it were a vector"
        Sxx = self.S
        for j in range(U.shape[1]):
            choldowndate(Sxx,U[:,j])
        
        self.S = Sxx
        self.Ss.append(self.S)
        self.xs.append(self.x)
        
        
        
    def batch(self):
        """
        batch function maybe build later
        """
        return

