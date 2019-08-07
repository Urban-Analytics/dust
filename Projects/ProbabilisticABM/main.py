from super_simple_model.visualiser import Visualiser
import super_simple_model.renderer as renderer
from super_simple_model.agent import Agent
from super_simple_model.door import Door
from super_simple_model.environment import Environment

import pyro
import pyro.distributions as dist
import torch
import numpy as np

n_samples =1000
steps = 1000
observation_freq = 1001
environment = Environment()
visualiser = Visualiser(environment)


def main():
    renderer.clear_output_folder()

    width, height = 20, 100
    environment.doors = [Door(id=0, xy=(1000 - width / 2, (500 * 1.33) - height / 2),
                              width=width, height=height, fill=True),
                         Door(id=1, xy=(1000 - width / 2, (500 * 0.66) - height / 2),
                              width=width, height=height, fill=True)]

    agent = Agent(x=0., y=500., n_samples=n_samples)
    agent.pick_destination(doors=environment.doors)
    location = agent.step()
    obs = None

    for i in range(steps):
        if i != 0:
            obs = observe(i)

        location = agent.step(pred=location,
                              obs=obs)

        print_agent_loc(i, location)

        visualiser.plot_agent(location[0], location[1], median=False)
        visualiser.plot_environment(environment)
        visualiser.save_plt()
        visualiser.clear_frame()

    renderer.render_agent()


# TODO Should move this to Agent class.
def print_agent_loc(i, location):
    print('Step:{} [x,y] {}'.format(i, [np.median(location[0]), np.median(location[1])]))


def observe(i):
    if (i % observation_freq) == 0:
        print('Observing!')
        obs = pyro.sample('obs', dist.Normal(loc=torch.tensor([[float(i)], [500.]]),
                                             scale=torch.tensor([[100.], [100.]])))

        # Have a look at using a whole tensor of noise.
        n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0.], [0.]]), scale=10.))

        obs = obs + n
    else:
        obs = None
    return obs


if __name__ == '__main__':
    main()
