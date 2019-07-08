import numpy as np
import os

import torch
from pyro import distributions as dist
import pyro


class Agent:
	def __init__(self, x, y, n_samples, **kwargs):
		self.n_samples = n_samples
		self.xy = torch.tensor([[x for _ in range(self.n_samples)],
								[y for _ in range(self.n_samples)]])
		self.s = pyro.sample('s', dist.Normal(loc=torch.tensor([[1. for _ in range(self.n_samples)],
																[0. for _ in range(self.n_samples)]]),
											  scale=torch.as_tensor([[1.], [.25]])))

	def step(self, pred=None, obs=None, noise=0.):
		xy = pyro.sample('xy', dist.Normal(loc=self.xy if pred is None else pred,
										   scale=torch.as_tensor([1.])), obs=obs)

		n = pyro.sample('n', dist.Normal(loc=torch.as_tensor([0. for _ in range(self.n_samples)]),
										 scale=torch.as_tensor([noise])))
		return (xy + self.s) + n
