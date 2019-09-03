import torch
from torch import tensor
from pyro import distributions as dist
from copy import deepcopy
import pyro
import numpy as np


class Agent:
	def __init__(self, x, y, n_samples=1, bias=0.5, **kwargs):
		self.n_samples = n_samples
		self.destination_preference = bias
		self.destination = None
		self.xy = tensor([[x for _ in range(self.n_samples)],
						  [y for _ in range(self.n_samples)]])
		self.s = tensor([1. for _ in range(self.n_samples)])
		self.rv_v = pyro.sample('rv_s', dist.LogNormal(loc=self.s, scale=1.))
		self.obs = None
		self.steps_taken = 0
		self.initial_state = deepcopy(self)

	def pick_destination(self, doors):
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
		self.xy = pyro.sample('xy', dist.Normal(loc=self.xy if posterior is None else posterior,
										   scale=tensor([[1.], [1.]])), obs=None if obs is None else obs)
		self.xy = self.move(origin=self.xy, destination=self.destination)

	def move(self, origin, destination):
		x = destination[0] - origin[0]
		y = destination[1] - origin[1]
		distance = (x*x + y*y)**.5
		loc = origin + self.rv_v * (destination - origin) / distance
		return loc

	def print_mean_agent_loc(self, i):
		print('Step:{} {}'.format(i, [np.mean(list(self.xy[0])), np.mean(list(self.xy[1]))]))

	def reset_agent(self):
		return self.initial_state