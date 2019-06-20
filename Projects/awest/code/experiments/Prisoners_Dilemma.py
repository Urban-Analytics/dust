# Prisoner's Dilemma
'''
Here we intend to create an ABM in concordance with:
Kristian Lindgren's 1991 paper Evolutionary Pheonomena in Simple Dynamics
'''
import numpy as np
import matplotlib.pyplot as plt
# from multiprocessing.dummy import Pool

results = [[(3, 3), (0, 5)], [(5, 0), (1, 1)]]
def game(a: bool, b: bool):
	# Prisoner's Dilemma game
	# a, b are boolean
	# 0: cooperate
	# 1: defect
	return results[a][b]


class Agent:

	def __init__(self, unique_id: int, games: int):
		self.unique_id = unique_id
		self.score = 0
		self.decision = np.random.randint(2)
		return

	def step(self, model):
		for agent in model.agents[self.unique_id+1:]:
			for game_id in range(model.games):
				a, b = game(self.decision, agent.decision)
				self.stratergise(game_id, agent.decision)

				model.scores[self.unique_id] += a
				model.scores[agent.unique_id] += b
		return

	def stratergise(self, game_id, verses):
		# redefine self.decision using memory and verses memory
		if np.random.rand() < .001:
			self.decision = np.random.randint(2)
		return


class Model:

	def __init__(self, population=100, games=5):
		self.population = population
		self.games = games
		self.agents = [Agent(unique_id, self.games) for unique_id in range(self.population)]
		return

	def step(self):
		self.scores = [0 for _ in range(self.population)]
		[agent.step(self) for agent in self.agents]
		self.resample()
		return

	def resample(self):
		# Threshold
		mean = sum(self.scores) / self.population
		std_dev = (sum([(score - mean)**2 for score in self.scores]) / self.population)**.5
		threshold = mean - std_dev

		# Sort
		ind_sort = sorted(enumerate(self.scores), key=lambda x:x[1])
		ind = [i[0] for i in ind_sort]
		sort = [i[1] for i in ind_sort]
		plt.plot(ind, '.')
		plt.plot(sort, '.')
		plt.show()

		# Resample
		index = [i[0] for i in ind_sort]
		j = 0
		for i, s in ind_sort:
			if s < threshold:
				index[i] = j
			else:
				j += 1
		return

	def batch(self, i=10):
		for _ in range(i):
			self.step()
			plt.hist(self.scores, bins=10)
			plt.pause(1)
		return


Model().batch(1)
