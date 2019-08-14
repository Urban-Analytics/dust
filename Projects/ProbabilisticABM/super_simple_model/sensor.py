import numpy as np
import torch
import pyro
import pyro.distributions as dist


class Sensor:
	def __init__(self, freq=1, n_samples=1):
		self.observation_freq = freq
		self.n_samples = n_samples

	def observe(self, t=1, agent=None):
		if (t % self.observation_freq) == 0:
			print('Observing!')
			obs = pyro.sample('my_obs', dist.Normal(loc=torch.tensor([[np.median(agent.xy[0]) for _ in range(self.n_samples)],
																   [np.median(agent.xy[1]) for _ in range(self.n_samples)]]),
												 scale=torch.tensor([[10.], [10.]])))

			n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0. for _ in range(self.n_samples)],
																	   [0. for _ in range(self.n_samples)]]),
													 scale=1.))

			obs = obs + n
		else:
			obs = None
		return obs
