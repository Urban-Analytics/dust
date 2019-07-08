import matplotlib.pyplot as plt
import numpy as np
import time


# This file will need some refactoring at some point.

class Visualiser:
	plt.figure(figsize=(12, 12))

	# TODO Rebuild the way this class initialises itself.
	def __init__(self):
		self.ENVIRONMENT = (1000, 1000)
		self.left, self.width = 0.1, 0.65
		self.bottom, self.height = 0.1, 0.65
		self.spacing = 0.02

	def plot_agent(self, x, y):
		rect_scatter = [self.left, self.bottom, self.width, self.height]
		rect_histx = [self.left, self.bottom + self.height + self.spacing, self.width, 0.2]
		rect_histy = [self.left + self.width + self.spacing, self.bottom, 0.2, self.height]

		ax_scatter = plt.axes(rect_scatter)
		ax_scatter.tick_params(direction='in', top=True, right=True)
		ax_histx = plt.axes(rect_histx)
		ax_histx.tick_params(direction='in', labelbottom=False)
		ax_histy = plt.axes(rect_histy)
		ax_histy.tick_params(direction='in', labelleft=False)

		ax_scatter.scatter(x,
						   y,
						   s=1,
						   label='Potential Locations')

		ax_scatter.scatter(np.median(x),
						   np.median(y),
						   s=10,
						   c='red',
						   marker='x',
						   label='Centre')

		ax_scatter.set_xlim((self.ENVIRONMENT[0] * -.2,
							 self.ENVIRONMENT[0]))
		ax_scatter.set_ylim((0,
							 self.ENVIRONMENT[1]))

		ax_scatter.legend(loc='upper right')
		ax_scatter.grid(True)

		n, bins, patches = ax_histx.hist(x, bins=100, density=1)
		self.build_heatmap(n, patches)
		n, bins, patches = ax_histy.hist(y, bins=100, orientation='horizontal', density=1)
		self.build_heatmap(n, patches)

		ax_histx.set_xlim(ax_scatter.get_xlim())
		ax_histy.set_ylim(ax_scatter.get_ylim())

	@staticmethod
	def build_heatmap(n, patches):
		fracs = n / n.max()
		for thisfrac, thispatch in zip(fracs, patches):
			color = plt.cm.inferno(np.linalg.norm(thisfrac))
			thispatch.set_facecolor(color)

	@staticmethod
	def save_plt():
		plt.savefig('output/{}.png'.format(time.time()))

	@staticmethod
	def clear_frame():
		plt.clf()
