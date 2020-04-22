from probablisticABM.visualiser import Visualiser
from probablisticABM.agent import Agent
from probablisticABM.door import Door
from probablisticABM.environment import Environment
from probablisticABM.sensor import Sensor
import probablisticABM.renderer as renderer

import pyro
from pyro.infer import Importance

pyro.set_rng_seed(7)

n_steps = 20
environment = Environment()
width, height = 20, 100
environment.doors = [Door(key=0, xy=(1000 - width / 2, (500 * 1.33) - height / 2),
                          width=width, height=height, fill=True),
                     Door(key=1, xy=(1000 - width / 2, (500 * 0.66) - height / 2),
                          width=width, height=height, fill=True)]


def main():
    """
    This main routine will run the agent using pyro's EmpiricalMarginal and Importance methods as opposed to the
    main.py which uses a custom loop to step the agent. This main routine runs but is less flexible when it comes to
    visualising due to the samples in the posterior being private. It was also found that trying to pass the
    posterior back as the prior did not yield clear results but did step the model forward in the expected direction.

	This should be looked into further.
	:return:
	"""
    posterior = None
    sensor = build_observations()
    agent = Agent(x=0., y=500.)
    agent.pick_destination(doors=environment.doors)

    # This is the inference algorithm. From what is understood this simply just samples the stochastic
    # function and builds a distribution.
    infer = Importance(model=agent.step, num_samples=1000)

    for step in range(n_steps):
        # Assimilate observation and update the prior with the posterior distribution.
        if (step % 10) == 0:  # For every n steps, calibrate.
            obs = sensor.aggregate_obs(step)
            print('\nAssimilating_Observation at Step {}'.format(step))
            sensor.print_detail(step)
        else:
            # Else do not make an observation and simply run the model forward using the agents internal position.
            obs = None

        # This contains the attributes such as mean and std for the posterior which then can be used to pass back as
        #
        posterior = pyro.infer.EmpiricalMarginal(infer.run(posterior=posterior, obs=obs), sites=['xy'])
        print_agent_loc(agent, step)
        print_posterior(posterior)


def print_agent_loc(agent, step):
    print('Step:{} xy: ({}, {})'.format(step, agent.xy[0].item(), agent.xy[1].item()))


def print_posterior(posterior):
    if posterior is not None:
        print('Posterior Mean: ({}, {})'.format(posterior.mean[0][0].item(), posterior.mean[0][1].item()))


def build_observations(n_samples=1000, steps=n_steps):
    """
	Runs the model to generate a list of synthetic observations that are stored inside the sensor object.
	:param n_samples:
	:param steps:
	:return: sensor
	"""
    n_samples = n_samples
    steps = steps
    agent = Agent(x=0., y=500., bias=0.0, n_samples=n_samples)
    sensor = Sensor(freq=1, n_samples=n_samples)
    agent.pick_destination(doors=environment.doors)
    renderer.clear_output_folder()

    for t in range(steps):
        if t != 0:
            sensor.observe(t, agent)

        agent.step()
    return sensor


if __name__ == '__main__':
    main()
