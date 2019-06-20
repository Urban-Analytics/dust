# MesaFree
'''
A model building framework.
'''
from random import shuffle, random
import matplotlib.pyplot as plt


class Agent:

	def __init__(self, model, unique_id):
		self.unique_id = unique_id
		self.location = (random() * model.width, random() * model.height)
		return

	def step(self, model):
		self.move(model)
		return

	def move(self, model):
		x, y = self.location
		x += 1
		x %= model.width
		y += 1
		y %= model.height
		self.location = (x, y)
		return

class Model:

	def __init__(self, width=500, height=500, population=10):
		self.step_id = 0
		# Parameters
		self.width = width
		self.height = height
		self.population = population
		self.agents = list([Agent(self, unique_id) for unique_id in range(self.population)])
		return

	def step(self):
		self.step_id += 1
		[agent.step(self) for agent in self.agents]
		return

	def get_state(self, do_ravel=True):
		state = zip(*[agent.location for agent in self.agents])
		return state

	def set_state(self, state):
		for i in range(self.population):
			self.agents[i].location = state[2*i:2*i+2]
		return


if __name__ == '__main__':
	model = Model(population=100)
	for _ in range(100):
		model.step()
		[plt.plot(*agent.location, '.k') for agent in model.agents]
		plt.plot(*model.agents2state(), '.k')
		plt.axis([0, model.width, 0, model.height])
		plt.pause(.05)
		plt.clf()
