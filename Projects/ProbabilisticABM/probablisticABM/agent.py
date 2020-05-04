import torch
from torch import tensor
from pyro import distributions as dist
import pyro
import numpy as np


class Agent:
    """
	The agent class for this project represents a simple agent that has randomness embedded within it using
	probabilistic programming package, Pyro as well as some common functionality such as movement and  storing
	location, destination, destination bias, speed and sample size.
	"""

    def __init__(self, x=0, y=500, n_samples=1, bias=0.5, **kwargs):
        self.n_samples = n_samples
        self.destination_preference = bias
        self.destination = None
        self.xy = tensor([[x for _ in range(self.n_samples)],
                          [y for _ in range(self.n_samples)]])
        self.s = tensor([1. for _ in range(self.n_samples)])
        self.rv_v = pyro.sample('rv_s', dist.LogNormal(loc=self.s, scale=1.))
        self.steps_taken = 0

    def pick_destination(self, doors):
        """
		When provided with a list of doors, this function will randomly select one by its key. This then sets
		the destination to the xy location of the door object. Door objects are extended from the rectangle objects
		provided in matplotlib.
		:param doors:
		:return:
        """
        ids = [int(pyro.sample('destination',
                               dist.Bernoulli(probs=self.destination_preference))) for _ in range(self.n_samples)]
        doors = [doors[id] for id in ids]
        door_x = tensor([door.get_x() +
                         door.get_width() / 2 for door in doors])
        door_y = tensor([door.get_y() +
                         door.get_height() / 2 for door in doors])
        door = torch.stack([door_x, door_y])
        self.destination = door

    def step(self, posterior=None, obs=None):
        """
		This is the primary stepping function for the agent. Without any arguments, it will step the agent and update
		self.xy with the new location i.e(self.xy = move(self.xy, destination)) The self.xy attribute will be
		initialised as x, y = (0, 500) by default. The function will also take in observations and condition the built
		in Pyro normal distribution sampler on this input. If given a posterior object such as one build by  Empirical
		Marginal, it will it's mean and stddev attributes and use those as to shape the new xy position (Though this
		has not been fully tested and may yield strange results).
		:param posterior:
		:param obs:
		:return:
		"""
        self.xy = pyro.sample('xy', dist.Normal(loc=self.xy if posterior is None else posterior.mean[0],
                                                scale=tensor([[1.], [1.]]) if posterior is None else posterior.stddev[
                                                    0]),
                              obs=obs)
        self.xy = self.move(origin=self.xy, destination=self.destination)

    def move(self, origin, destination):
        """
		The Lerp function(adapted from Kieran's and Andrew's work) will calculate the line needed to get a specified
		destination and steps along this line using a static random variable velocity initialise when the agent is
		built.
		:param origin:
		:param destination:
		:return: loc
		"""
        x = destination[0] - origin[0]
        y = destination[1] - origin[1]
        distance = (x * x + y * y) ** .5
        loc = origin + self.rv_v * (destination - origin) / distance
        return loc

    def print_mean_agent_loc(self, i):
        print('Step:{} {}'.format(i, [np.mean(list(self.xy[0])), np.mean(list(self.xy[1]))]))
