import torch
from torch import tensor
from pyro import distributions as dist
import pyro


class Agent:
	def __init__(self, x, y, n_samples, **kwargs):
		self.n_samples = n_samples
		self.destination = pyro.sample('destination', dist.Bernoulli(.5))
		self.s = tensor([1. for _ in range(self.n_samples)])

		self.xy = tensor([[x for _ in range(self.n_samples)],
						  [y for _ in range(self.n_samples)]])

		self.rv_v = pyro.sample('rv_s', dist.Normal(loc=self.s, scale=1.))

	def step(self, pred=None, obs=None):
		xy = pyro.sample('xy', dist.Normal(loc=self.xy if pred is None else pred,
										   scale=tensor([[0.], [0.]])), obs=obs)

		# PLACE HOLDER DOORS
		door_x = tensor([1000.])
		door_y = tensor([500.*1.33])
		door = torch.stack([door_x, door_y])

		return self.move(origin=xy, destination=door)

	def move(self, origin, destination):
		x = destination[0] - origin[0]
		y = destination[1] - origin[1]
		distance = (x*x + y*y)**.5
		loc = origin + self.s * (destination - origin) / distance
		return loc