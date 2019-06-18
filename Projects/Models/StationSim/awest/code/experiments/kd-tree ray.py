import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

x, y = np.mgrid[0:5, 2:8]
p = np.c_[x.ravel(), y.ravel()]
tree = cKDTree(p, compact_nodes=False)

fig, ax = plt.subplots()
ax.plot(p[:,0], p[:,1], '.')

l = p[[4, 24]]
plt.plot(l[0], l[1])



d = []


import numpy as np
import matplotlib.pyplot as plt
from plotly import offline as py
import plotly.tools as tls
py.init_notebook_mode()

t = np.linspace(0, 20, 500)
plt.plot(t, np.sin(t))

py.iplot(tls.mpl_to_plotly(plt.gcf()))
