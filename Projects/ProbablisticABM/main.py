from super_simple_model.visualiser import Visualiser
import super_simple_model.renderer as renderer
from super_simple_model.agent import Agent


def main():
    renderer.clear_output_folder()
    agent = Agent(x=1, y=500, sample_size=1000, noise=10)
    visualiser = Visualiser()

    i = 0
    while True:
        x, y = agent.get_sample_position()
        visualiser.plot_agent(x, y)
        visualiser.save_plt()
        visualiser.clear_frame()
        agent.step()
        # Issue with recursion depth if i is too large.
        if i > 300:
            break
        i += 1
    renderer.render_agent()


if __name__ == '__main__':
    main()
