import matplotlib.pyplot as plt
import numpy as np

x = np.empty(100_000)
i = 0
while i < len(x):
    r = -2
    while r <= -2:
        r = np.random.normal()
    x[i] = r
    i += 1

y = np.fmax(-2, np.random.normal(size=100_000))


plt.hist(x, bins=100, alpha=.5)
plt.hist(y, bins=100, alpha=.5)
plt.show()
