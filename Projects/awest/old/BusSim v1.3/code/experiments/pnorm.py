# p-norm difference

import numpy as np
import matplotlib.pyplot as plt

I = 100000
high, low = 10, -10
x = np.random.random(size=(I, 2))
x = (high-low) * x - low
ds = []
for i in range(len(x)-1):
    x0 = x[i]
    x1 = x[i+1]
    n1 = np.linalg.norm(x0-x1, 1)
    n2 = np.linalg.norm(x0-x1, 2)
    d = (n1 / np.sqrt(2) - n2)
    ds.append(d)

plt.hist(ds)
plt.show()
