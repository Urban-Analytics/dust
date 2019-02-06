# Graph the results from the StationSim / Keanu experiments. See https://github.com/nickmalleson/keanu-post-hackathon/releases/tag/v0.1

#%% Imports
import pandas as pd
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

%matplotlib inline


os.chdir("/Users/nick/conferences/2018/ABMUS2018/abstract-nick/results0.1/")


INPUT = "results_no_observe"

#%% Read the data

# Read once to get the number of columns
csv = pd.read_csv(INPUT+".csv")
col_names = ["t"+str(x) for x in range(len(csv.columns))]
col_names = ["sample"] + col_names

# Now read again, setting the column names correctly
csv = pd.read_csv(INPUT+".csv", header=None, names=col_names)

# Transpose (so each column is a sample) and fix the headers (not sure why transpose() breaks them)
df = csv.transpose()
df.columns = df.iloc[0] # Take headers from the first row 
df = df.drop(df.index[0]) # Drop the first row

df = df.dropna() # There are some NA rows at the end; drop these

#df = df.iloc[:,range(1,10)] # Reduce the number of samples (to test)

#%% Calculate the mean and confidence interval

_mean = [np.mean(row[1]) for row in df.iterrows() ]
#_sd =   [np.std(row[1]) for row in df.iterrows() ]

Z = 2.98 # 99%
#Z = 1.96 # 95%

upper_ci = [ _mean[i] + Z * (np.std(row[1]) / np.sqrt(len(row[1]))) for i,row in enumerate(df.iterrows()) ]
lower_ci = [ _mean[i] - Z * (np.std(row[1]) / np.sqrt(len(row[1]))) for i,row in enumerate(df.iterrows()) ]

#lower_ci = [ m - err for m,err in zip(_mean,moe) ]
#upper_ci = [ m + err for m,err in zip(_mean,moe) ]


#%% Plot

plt.rcParams["figure.figsize"] = [11,7] # Figure size

lines = [] # lines on the plot. Remember these because we need one for the legend

for column in df:
    x = np.arange(len(df[column]))
    y = df[column]
    lines += plt.plot(x, y, color="#999999", lw=0.5, label="Samples")

# Plot the and CIs (and make a 'proxy' for the legend for some reason)
# https://matplotlib.org/users/legend_guide.html#creating-artists-specifically-for-adding-to-the-legend-aka-proxy-artists

proxies= []
for (vals, colour, name, width) in [ 
        (_mean, "#000000", "Mean", 1.0),
        (upper_ci, "#FF4444", "Upper CI", 1.0),
        (lower_ci, "#4444FF", "Lower CI", 1.0)
        ]:
    plt.plot(                 np.arange(len(vals)), vals, color=colour, label=name, lw=width)
    proxies += [mlines.Line2D(np.arange(len(vals)), vals, color=colour, label=name, lw=width)]

plt.legend( handles = [lines[0]]+proxies, loc='upper right', frameon=False, fontsize=11)
plt.xlabel("Number of iterations", fontsize=11)
plt.ylabel("Number of agents", fontsize=11)
plt.title ("Number of agents per iteration in the\nsimulation across all samples", fontsize=14)


#plt.show()

plt.savefig(INPUT+'.pdf', dpi=128, bbox_inches="tight")


#%% WRITE OUT TO QUICKLY VERIFY IN R

#df.to_csv('results_temp.csv')
