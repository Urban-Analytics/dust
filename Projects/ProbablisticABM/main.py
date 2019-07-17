from super_simple_model.visualiser import Visualiser
import super_simple_model.renderer as renderer
from super_simple_model.agent import Agent
from super_simple_model.door import Door

import pyro
import pyro.distributions as dist
import torch
import numpy as np

n_samples=1000
steps=100
delta=301
visualiser = Visualiser(environment=(1000, 1000))


def main():
    renderer.clear_output_folder()

    agent = Agent(x=0., y=500., n_samples=n_samples)
    width, height = 20, 100
    location = agent.step()

    for i in range(steps):
        doors = [Door(xy=(1000-width/2, (500*1.33)-height/2), width=width, height=height, fill=True),
                 Door(xy=(1000-width/2, (500*.66)-height/2), width=width, height=height, fill=True)]

        obs = None

        location = agent.step(pred=location,
                              obs=obs)

        print_agent_loc(location)

        visualiser.plot_agent(location[0], location[1], median=False)
        visualiser.plot_doors(doors=doors)
        visualiser.save_plt()
        visualiser.clear_frame()

    renderer.render_agent()


def print_agent_loc(location):
    print('[x,y] {}'.format([np.median(location[0]), np.median(location[1])]))


def observe(i):
    if (i % delta) == 0:
        print('Observing!')
        obs = pyro.sample('obs', dist.Normal(loc=torch.tensor([[float(i)], [500.]]),
                                             scale=torch.tensor([[10.], [10.]])))

        n = pyro.sample('obs_noise', dist.Normal(loc=torch.tensor([[0.], [0.]]), scale=100.))

        obs = obs + n
    else:
        obs = None
    return obs


if __name__ == '__main__':
    main()
