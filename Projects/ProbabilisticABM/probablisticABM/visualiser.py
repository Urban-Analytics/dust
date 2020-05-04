import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import numpy as np
import time


class Visualiser:
    """
	The Visualiser class provides methods to visualise the output of the abm simulation.
	"""
    plt.figure(figsize=(12, 12))

    def __init__(self, environment=None, agent=None, **kwargs):
        self.environment = environment
        self.agent = agent
        self.x, self.y = self.agent.xy
        self.bounds = (1000, 1000) if environment is None else self.environment.get_env_size()
        self.left, self.width = 0.1, 0.65
        self.bottom, self.height = 0.1, 0.65
        self.spacing = 0.02
        self.rect_scatter = [self.left, self.bottom, self.width, self.height]
        self.rect_histx = [self.left, self.bottom + self.height + self.spacing, self.width, 0.2]
        self.rect_histy = [self.left + self.width + self.spacing, self.bottom, 0.2, self.height]
        self.ax_scatter = None
        self.ax_histx = None
        self.ax_histy = None

    def plot_agent(self, **kwargs):
        """
		Main routine to generate the custom scatter plot with corresponding distributions of the current agent state
		location(s). Should you wish to add other elements to the plot, simply append a new plotting function to the
		file and call it here.
		:param kwargs:
		:return:
		"""
        self.ax_scatter = plt.axes(self.rect_scatter)
        self.ax_scatter.tick_params(direction='in', top=True, right=True)
        self.ax_histx = plt.axes(self.rect_histx)
        self.ax_histx.tick_params(direction='in', labelbottom=False)
        self.ax_histy = plt.axes(self.rect_histy)
        self.ax_histy.tick_params(direction='in', labelleft=False)

        self.plot_prior_scatter()

        self.ax_scatter.set_xlim((self.bounds[0] * -.2, self.bounds[0]))
        self.ax_scatter.set_ylim((0, self.bounds[1]))

        self.ax_scatter.legend(loc='upper right')
        self.ax_scatter.grid(True)

        n, bins, patches = self.ax_histx.hist(self.x, bins=100, density=1)
        self.build_heatmap(n, patches)
        n, bins, patches = self.ax_histy.hist(self.y, bins=100, orientation='horizontal', density=1)
        self.build_heatmap(n, patches)

        self.ax_histx.set_xlim(self.ax_scatter.get_xlim())
        self.ax_histy.set_ylim(self.ax_scatter.get_ylim())

    def plot_prior_scatter(self):
        self.ax_scatter.scatter(self.x,
                                self.y,
                                s=3,
                                label='Prior')

    def plot_doors(self, doors=None):
        """ Adds patch collection to the scatter plot. In this case we use doors extended from the rectangle patches.
		:param doors:
		:return:
		"""
        doors = PatchCollection(doors)
        self.ax_scatter.add_collection(doors)

    def plot_environment(self):
        self.plot_doors(self.environment.doors)

    @staticmethod
    def build_heatmap(n, patches):
        """
		Defines the heatmap style for the distribution plots.
		:param n:
		:param patches:
		:return:
		"""
        fracs = n / n.max()
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.inferno(np.linalg.norm(thisfrac))
            thispatch.set_facecolor(color)

    @staticmethod
    def save_plt():
        """
		Wrapper for saving the current plot to disk.
		:return:
		"""
        plt.savefig('output/{}.png'.format(time.time()))

    @staticmethod
    def clear_frame():
        """
		Clear the current frame/plot.
		:return:
		"""
        plt.clf()
