from probablisticABM.visualiser import Visualiser
from probablisticABM.agent import Agent
from probablisticABM.door import Door
from probablisticABM.environment import Environment
from probablisticABM.sensor import Sensor
import probablisticABM.renderer as renderer

environment = Environment()
width, height = 20, 100
environment.doors = [Door(key=0, xy=(1000 - width / 2, (500 * 1.33) - height / 2),
                          width=width, height=height, fill=True),
                     Door(key=1, xy=(1000 - width / 2, (500 * 0.66) - height / 2),
                          width=width, height=height, fill=True)]


def main():
    n_samples = 100
    n_steps = 400
    agent = Agent(x=0., y=500., n_samples=n_samples)

    renderer.clear_output_folder()

    agent.pick_destination(doors=environment.doors)

    sensor = build_observations(n_samples=n_samples, steps=n_steps)

    for step in range(n_steps):
        # For every n steps, calibrate.
        if (step % 10) == 0 and step != 0:
            obs = sensor.aggregate_obs(step)
            print('\nAssimilating_Observation at Step {}'.format(step))
            sensor.print_detail(step)
        else:
            obs = None

        agent.step(obs=obs)
        agent.print_mean_agent_loc(step)

        visualiser = Visualiser(environment=environment, agent=agent)
        visualiser.plot_agent(median=False)
        visualiser.plot_environment()
        visualiser.save_plt()
        visualiser.clear_frame()

    renderer.render_agent()


def build_observations(n_samples=1000, steps=400):
    """
	Runs the model to generate a list of observations that are stored inside the sensor object.
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
