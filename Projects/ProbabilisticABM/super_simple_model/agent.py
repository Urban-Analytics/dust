import torch
from torch import tensor
from pyro import distributions as dist
import pyro
import numpy as np


class Agent:
	def __init__(self, x, y, n_samples, **kwargs):
		self.n_samples = n_samples
		self.destination_preference = .5
		self.destination = None
		self.xy = tensor([[x for _ in range(self.n_samples)],
						  [y for _ in range(self.n_samples)]])
		self.s = tensor([1. for _ in range(self.n_samples)])
		self.rv_v = pyro.sample('rv_s', dist.LogNormal(loc=self.s, scale=1.))
		self.obs = None

	def pick_destination(self, doors):
		ids = [int(pyro.sample('destination', dist.Bernoulli(probs=self.destination_preference))) for _ in range(self.n_samples)]
		doors = [doors[id] for id in ids]
		door_x = tensor([door.get_x() + door.get_width() / 2 for door in doors])
		door_y = tensor([door.get_y() + door.get_height() / 2 for door in doors])
		door = torch.stack([door_x, door_y])
		self.destination = door

	def model(self, pred=None, obs=None):
		self.xy = pyro.sample('xy', dist.Normal(loc=self.xy if pred is None else pred,
										   scale=tensor([[1.], [1.]])), obs=obs)
		self.xy = self.move(origin=self.xy, destination=self.destination)

	def guide(self, pred=None):
		self.xy = pyro.sample('xy', dist.Normal(loc=self.xy if pred is None else pred,
												scale=tensor([[1.], [1.]])))
		self.xy = self.move(origin=self.xy, destination=self.destination)

	def move(self, origin, destination):
		x = destination[0] - origin[0]
		y = destination[1] - origin[1]
		distance = (x*x + y*y)**.5
		loc = origin + self.rv_v * (destination - origin) / distance
		return loc

	def print_median_agent_loc(self, i):
		print('Step:{} [x,y] {}'.format(i, [np.median(self.xy[0]), np.median(self.xy[1])]))