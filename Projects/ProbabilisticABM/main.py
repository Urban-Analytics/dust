from super_simple_model.visualiser import Visualiser
import super_simple_model.renderer as renderer
from super_simple_model.agent import Agent
from super_simple_model.door import Door
from super_simple_model.environment import Environment
from super_simple_model.sensor import Sensor

import pyro
import pyro.distributions as dist
import torch
import numpy as np

n_samples = 10000
steps = 400
environment = Environment()
visualiser = Visualiser(environment)
sensor = Sensor(freq=100, n_samples=n_samples)


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

    for t in range(steps):
        if t != 0:
            obs = sensor.observe(t, location)

        location = agent.step(pred=location,
                              obs=obs)

        print_agent_loc(t, location)

        visualiser.plot_agent(location[0], location[1], median=False)
        visualiser.plot_environment()
        visualiser.save_plt()
        visualiser.clear_frame()

    renderer.render_agent()


# TODO Should move this to Agent class.
def print_agent_loc(i, location):
    print('Step:{} [x,y] {}'.format(i, [np.median(location[0]), np.median(location[1])]))


if __name__ == '__main__':
    main()
