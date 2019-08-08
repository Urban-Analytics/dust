import torch
from torch import tensor
from pyro import distributions as dist
import pyro


class Agent:
	def __init__(self, x, y, n_samples, **kwargs):
		self.n_samples = n_samples
		self.destination_preference = .5
		self.rv_destination = int(pyro.sample('destination', dist.Bernoulli(probs=self.destination_preference)))
		self.destination = None
		self.xy = tensor([[x for _ in range(self.n_samples)],
						  [y for _ in range(self.n_samples)]])
		self.s = tensor([1. for _ in range(self.n_samples)])
		self.rv_v = pyro.sample('rv_s', dist.LogNormal(loc=self.s, scale=1.))

	def step(self, pred=None, obs=None):
		xy = pyro.sample('xy', dist.Normal(loc=self.xy if pred is None else pred,
										   scale=tensor([[5.], [5.]])), obs=obs)
		return self.move(origin=xy, destination=self.destination)

	def pick_destination(self, doors):
		door = doors[self.rv_destination]
		door_x = tensor([door.get_x() + door.get_width() / 2])
		door_y = tensor([door.get_y() + door.get_height() / 2])
		door = torch.stack([door_x, door_y])
		self.destination = door

	def move(self, origin, destination):
		x = destination[0] - origin[0]
		y = destination[1] - origin[1]
		distance = (x*x + y*y)**.5
		loc = origin + self.rv_v * (destination - origin) / distance
		return loc