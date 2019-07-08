from super_simple_model.visualiser import Visualiser
import super_simple_model.renderer as renderer
from super_simple_model.agent import Agent

import pyro
import pyro.distributions as dist
import torch
import numpy as np

n_samples=1000
visualiser = Visualiser()


def main():
    renderer.clear_output_folder()

    agent = Agent(x=0., y=500., n_samples=n_samples)

    location = agent.step()
    for i in range(1000):
        if (i % 50) == 0:
            print('Observing!')
            obs = pyro.sample('obs', dist.Normal(loc=torch.tensor([[float(i)], [500]]),
                                                 scale=torch.tensor([[20.], [10]])))
        else:
            obs = None

        location = agent.step(pred=location,
                              obs=obs,
                              noise=1.)

        print('[x,y] {}'.format([np.median(location[0]), np.median(location[1])]))

        visualiser.plot_agent(location[0], location[1])
        visualiser.save_plt()
        visualiser.clear_frame()

    renderer.render_agent()


if __name__ == '__main__':
    main()
