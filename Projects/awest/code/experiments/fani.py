
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing.dummy import Pool

array = np.random.random(size=(10, 20, 30))
empty = np.ones(array[0].shape)

fig = plt.figure()
func = lambda i: plt.imshow(array[i])
frames = len(array)
ani = FuncAnimation(fig, func, frames)
plt.show()
