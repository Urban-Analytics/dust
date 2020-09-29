import os

files = os.listdir()

framesTaken = []
import numpy as np
for i in range(len(files)-1):
	time, x, y = np.loadtxt(files[i],unpack=True)
	framesTaken.append(float(len(time)))

print (framesTaken)
import math
framesTaken = np.asarray(framesTaken)
framesTaken /= 25.

import matplotlib.pyplot as plt
fig, ax = plt.subplots(num=None,figsize=(5.5, 3.5),dpi=128,facecolor='w',edgecolor='k')
n_bins = int(math.ceil(max(framesTaken) - min(framesTaken))/4)
n, bins, patches = plt.hist(framesTaken, bins=n_bins, density=False, color='orange', alpha=1., edgecolor='black', lw = 1)
plt.xlim(0, 150)
plt.xlabel('Time (s)', fontsize=12200)
plt.ylabel("Frequency", fontsize=12) 
plt.savefig('real_time_his.png')
