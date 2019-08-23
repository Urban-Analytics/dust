import numpy as np
import torch
import pyro
import pyro.distributions as dist


class Sensor:
	def __init__(self, freq=1, n_samples=1):
		self.observation_freq = freq
		self.n_samples = n_samples
		self.scale = 0.
		self.noise = 0.
		self.observations = None

	def observe(self, t=1, agent=None):
		if (t % self.observation_freq) == 0:
			self.activate_sensor()
			print('Observing!')
			obs = pyro.sample('my_obs', dist.Normal(loc=torch.tensor([[np.median(agent.xy[0]) for _ in range(self.n_samples)],
																   [np.median(agent.xy[1]) for _ in range(self.n_samples)]]),
												 scale=torch.tensor([[self.scale], [self.scale]])))

			n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0. for _ in range(self.n_samples)],
																	   [0. for _ in range(self.n_samples)]]),
													 scale=self.noise))

			self.observations.append(obs + n)
			self.print_detail(obs)

	def print_detail(self, obs):
		print('Observation X Mean: {} Scale: {}'.format(np.mean(list(obs[0])), self.scale))
		print('Observation Y Mean: {} Scale: {}'.format(np.mean(list(obs[1])), self.scale))

	def activate_sensor(self):
		if self.observations is None:
			self.observations = []
