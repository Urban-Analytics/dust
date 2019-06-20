# Screen Saver
'''
An agent based model so basic, a child made it.
'''
import numpy as np
import matplotlib.pyplot as plt


class Agent:

	def __init__(self, unique_id):
		self.unique_id = unique_id
		self.status = True
		self.location = np.random.uniform(size=2)
		self.velocity = np.random.uniform(-1, +1, size=2) / 200
		return

	def step(self):
		self.move()
		return

	def move(self):
		self.location += self.velocity
		# Bounce (oobb)
		for i in range(len(self.location)):
			if self.location[i] < 0 or 1 < self.location[i]:
				self.velocity[i] = -self.velocity[i]
				self.status = False
			else:
				self.status = True
		return


class Model:

	def __init__(self, params={}):
		# Default Params
		self.population = 100
		# Dictionary Params Edit
		self.params = (params,)
		[setattr(self, key, value) for key, value in params.items()]
		# Model
		self.boundaries = np.array([[0, 0], [1, 1]])
		# Init Agents
		self.agents = list([Agent(unique_id) for unique_id in range(self.population)])
		return

	def step(self):
		[agent.step() for agent in self.agents]
		return

	# Filtering Interface

	def mask(self):
		mask = np.array([agent.status for agent in self.agents])
		active = np.sum(mask)
		mask = np.ravel(np.stack([mask, mask], axis=1))  # Two pieces of data per agent
		return mask, active

	def get_state(self, do_ravel=True):
		state = [agent.location for agent in self.agents]
		if do_ravel:
			state = np.ravel(state)
		else:
			state = np.array(state)
		return state

	def set_state(self, state, noise=False):
		for i in range(len(self.agents)):
			self.agents[i].location = state[2 * i:2 * i + 2]
			if noise:
				self.agents[i].location += np.random.normal(0, noise, size=2)
		return

	def ani(self, agents=None, colour=None, alpha=1):
		state = self.get_state(do_ravel=False)[:agents].T
		plt.scatter(state[0], state[1], marker='.', color=colour, alpha=alpha)
		plt.axis((0, 1, 0, 1))
		return

	# Batch

	def batch(self, iterations=10, do_ani=False):
		for _ in range(iterations):
			self.step()
			if do_ani:
				plt.clf()
				self.ani()
				plt.pause(1 / 4)
		return


if __name__ == '__main__':
	model = Model()
	model.batch(100, True)
