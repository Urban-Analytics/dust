import numpy as np
import matplotlib.pyplot as plt

k=0
ps=[5,25,50]
x = np.arange(0,4,0.01)

f = plt.figure(figsize=(12,12))
styles = ["--",":","-."]
for i,p in enumerate(ps):
    lambd = x**2*(p+k) - p
    y1 = lambd/(p+lambd)
    y2 = 1/(2*(p+lambd))
    plt.ylim([-2,2])
    equal = np.sqrt(((2*p)+1)/(2*(k+p)))
    #plt.xlim([-equal-0.5,equal+0.5])
    plt.plot(x,y2,label=f"Weighting for Outer Sigmas for p={p}",ls = styles[i],linewidth=4)

plt.plot(x,y1,label=f"Weighting for Central Sigma",linewidth=4)

plt.axhline(y=0,color="k",ls="--",alpha=0.4)
plt.axhline(y=1,color="k",ls="--",alpha=0.4)
plt.axvline(x=0,color="k",ls="--",alpha=0.4)

#plt.axvline(x=equal,color="r",linestyle="--")
#plt.axvline(x=-equal,color="r",linestyle="--")

plt.legend(fontsize = 18)
plt.xlabel("Alpha")
plt.ylabel("Weighting of Central and Outer Sigma Points")
#plt.title(f"Parameter Alpha's effect on MSSP Weighting")
plt.savefig("alphas_plot.pdf")