from super_simple_model.visualiser import Visualiser
from super_simple_model.agent import Agent
from super_simple_model.door import Door
from super_simple_model.environment import Environment
from super_simple_model.sensor import Sensor
import super_simple_model.renderer as renderer

import matplotlib as plt
import numpy as np
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam as Adam


pyro.set_rng_seed(101)

n_samples = 1
steps = 400

environment = Environment()

agent = Agent(x=0., y=500., n_samples=n_samples)
width, height = 20, 100

sensor = Sensor(freq=1, n_samples=n_samples)

visualiser = Visualiser(environment=environment,
						agent=agent)

environment.doors = [Door(id=0, xy=(1000 - width / 2, (500 * 1.33) - height / 2),
						  width=width, height=height, fill=True),
					 Door(id=1, xy=(1000 - width / 2, (500 * 0.66) - height / 2),
						  width=width, height=height, fill=True)]

agent.pick_destination(doors=environment.doors)

adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
optimizer = Adam(adam_params)


def main():
	renderer.clear_output_folder()

	agent.obs = sensor.observe(agent=agent)
	conditioned_position = agent.model(obs=agent.obs)
	posterior = SVI(model=agent.model, guide=agent.guide, optim=optimizer, loss=Trace_ELBO(),  num_samples=1000, num_steps=1000)

	marginal = pyro.infer.EmpiricalMarginal(posterior.run(), sites='xy')

	visualiser.plot_agent(median=False)
	visualiser.plot_environment()
	visualiser.save_plt()
	visualiser.clear_frame()




if __name__ == '__main__':
	main()
