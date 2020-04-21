import numpy as np
import torch
import pyro
import pyro.distributions as dist
from torch import tensor


class Sensor:
    def __init__(self, freq=1, n_samples=1):
        self.observation_freq = freq
        self.n_samples = n_samples
        self.scale = 1.
        self.noise = 0.
        self.obs = None

    def observe(self, t=1, agent=None):
        if (t % self.observation_freq) == 0:
            self.activate_sensor()
            obs = pyro.sample('my_obs',
                              dist.Normal(loc=torch.tensor([[np.median(agent.xy[0]) for _ in range(self.n_samples)],
                                                            [np.median(agent.xy[1]) for _ in range(self.n_samples)]]),
                                          scale=torch.tensor([[self.scale], [self.scale]])))

            n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0. for _ in range(self.n_samples)],
                                                                       [0. for _ in range(self.n_samples)]]),
                                                     scale=self.noise))

            self.obs.append(obs + n)

    def aggregate_obs(self, step):
        if self.obs is not None:
            obs_median = tensor([[np.mean(list(self.obs[step][0]))], [np.mean(list(self.obs[step][1]))]])
            return obs_median

    def print_detail(self, t):
        print('Observation\n '
              'X Mean: {}\n '
              'STD: {}\n '
              'Y Mean: {}\n '
              'STD: {}\n'.format(np.mean(list(self.obs[t][0])),
                                 np.std(list(self.obs[t][0])),
                                 np.mean(list(self.obs[t][1])),
                                 np.std(list(self.obs[t][1]))))

    def activate_sensor(self):
        if self.obs is None:
            self.obs = []
