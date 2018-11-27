# NextRandom
'''
A class for organising random number usage
'''
import numpy as np


class NextRandom:

    def __init__(self):
        np.random.seed(303)
        self.random_number_usage = 0
        return

    def uniform(self, high=1, low=0, shape=1):
        r = np.random.random(shape)
        r = r * (high - low)
        self.random_number_usage += np.size(r)
        return r

    def gaussian(self, mu=0, sigma=1, shape=1):
        r = np.random.normal(shape)
        r = r * sigma + mu
        self.random_number_usage += np.size(r)
        return r

    def integer(self, high=1, low=0, shape=1):
        r = np.random.randint(low, high + 1, shape)
        self.random_number_usage += np.size(r)
        return r
