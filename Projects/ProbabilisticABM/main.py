from super_simple_model.visualiser import Visualiser
from super_simple_model.agent import Agent
from super_simple_model.door import Door
from super_simple_model.environment import Environment
from super_simple_model.sensor import Sensor
import super_simple_model.renderer as renderer


n_samples = 1000
steps = 400

environment = Environment()

agent = Agent(x=0., y=500., n_samples=n_samples)
width, height = 20, 100

sensor = Sensor(freq=200, n_samples=n_samples)

visualiser = Visualiser(environment=environment, agent=agent)

environment.doors = [Door(id=0, xy=(1000 - width / 2, (500 * 1.33) - height / 2),
						  width=width, height=height, fill=True),
					 Door(id=1, xy=(1000 - width / 2, (500 * 0.66) - height / 2),
						  width=width, height=height, fill=True)]


def main():
	renderer.clear_output_folder()

	agent.pick_destination(doors=environment.doors)
	obs = None

	for t in range(steps):
		if t != 0:
			obs = sensor.observe(t, agent)

		agent.step(pred=agent.xy,
				   obs=obs)

		agent.print_median_agent_loc(t)

		visualiser.plot_agent(median=False)
		visualiser.plot_environment()
		visualiser.save_plt()
		visualiser.clear_frame()

	renderer.render_agent()


if __name__ == '__main__':
	main()
