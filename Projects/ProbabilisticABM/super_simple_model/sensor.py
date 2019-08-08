import numpy as np
import torch
import pyro
import pyro.distributions as dist


class Sensor:
	def __init__(self, freq, n_samples):
		self.observation_freq = 100
		self.n_samples = n_samples

	def observe(self, t, location):
		if (t % self.observation_freq) == 0:
			print('Observing!')
			obs = pyro.sample('obs', dist.Normal(loc=torch.tensor([[np.median(location[0]) for _ in range(self.n_samples)],
																   [np.median(location[1]) for _ in range(self.n_samples)]]),
												 scale=torch.tensor([[10.], [10.]])))

			# Have a look at using a whole tensor of noise.
			n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0. for _ in range(self.n_samples)],
																	   [0. for _ in range(self.n_samples)]]),
													 scale=1.))

			obs = obs + n
		else:
			obs = None
		return obs
