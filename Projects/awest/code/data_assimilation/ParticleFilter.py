# Particle Filter for Agent-Based Modelling
#
# Requirement:
# 	model0                         is your agent-based model
# 	model.step()                   advance the model
# 	state = model.get_state()   get a state vector of the abm equivalent to the observation vector
# 	model.set_state(state)      give a state vector to the particle-model maybe with added noise
# 	mask, activity = model.mask()  often agents are inactive in abms, because state vector shouldn't change size a boolean mask is required for statistics generation
# Requests:
# 	model.params                   for reinitialising models (if there is initial randomness) the parameters will need reloading
# 	model.unique_id                is applied with this filter, maybe this
# 	model.ani()                    is used for animation
#
# TODO:
# 	mask ani
# 	async
# 	await
# 	wait


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from copy import deepcopy
from multiprocessing.dummy import Pool

class ParticleFilter:

	def __init__(self, model0, particles=10, window=1, do_copies=True, do_save=False):
		self.time = 0
		# Params
		self.window = window
		self.particles = particles
		# Model
		self.models = [deepcopy(model0) for _ in range(self.particles)]
		if not do_copies:
			[model.__init__() for model in self.models]
		for unique_id in range(self.particles):
			self.models[unique_id].unique_id = unique_id
		# Save
		self.do_save = do_save  # save stats
		if self.do_save:
			self.active = []
			self.mask = []
			self.mean = []
			self.var = []
			self.err = []
		return

	def step(self, state_obs):
		self.time += self.window
		states = np.array([model.get_state() for model in self.models])
		states = self.predict(states)
		weights = self.reweight(states, state_obs)
		states, weights = self.resample(states, weights)
		if self.do_save: self.save(states, weights, state_obs)
		return

	def predict(self, states):
		[self.models[i].set_state(states[i]) for i in range(self.particles)]
		Pool().map(lambda model: [model.step() for _ in range(self.window)], self.models)
		states = np.array([model.get_state() for model in self.models])
		return states

	def reweight(self, states, state_obs):
		if states.shape[1] != state_obs.shape[0]:
			print('Warning - Not equal states and state_obs lengths.\nSee documentation.\n\nShortening quick fix applied.')
			states = states[:, :len(state_obs)]
		distance = np.linalg.norm(states - state_obs, axis=1)
		weights = 1 / np.fmax(1e-99, distance)
		weights /= np.sum(weights)
		return weights

	def resample(self, states, weights):
		offset = (np.arange(self.particles) + np.random.uniform()) / self.particles
		cumsum = np.cumsum(weights)
		i, j = 0, 0
		indexes = np.empty(self.particles, 'i')
		while i < self.particles and j < self.particles:
			if offset[i] < cumsum[j]:
				indexes[i] = j
				i += 1
			else:
				j += 1
		states = states[indexes]
		weights = weights[indexes]
		return states, weights

	def save(self, states, weights, state_obs):
		active = np.array([model.mask()[1] for model in self.models], 'i')
		self.active.append(active)
		mask = np.array([model.mask()[0] for model in self.models], 'b')
		self.mask.append(mask)
		mean = np.average(np.where(mask, states, np.nan), weights=weights[mask], axis=0)
		self.mean.append(np.average(mean))
		var = np.average((states - mean)**2, weights=weights, axis=0)
		self.var.append(np.average(var))
		err = np.linalg.norm(mean - state_obs)
		self.err.append(err)
		return

	def file(self, do_file=True):
		if self.do_save:
			mask = np.array(self.mask, 'b')
			mean = np.array(self.mean, 'f')
			var = np.array(self.var, 'f')
			if np.any(var) < 0: print('Warning - Negative variance')
			if np.any(var) == np.nan: print('Warning - A NaN variance')
			err = np.array(self.err, 'f')
			if np.any(err) < 0: print('Warning - Negative error')
			if np.any(err) == np.nan: print('Warning - A NaN variance')
			if not do_file:
				return mask, mean, var, err
			else:
				np.savez('pf_data', mask, mean, var, err)
		else:
			print('Warning - Cannot file as do_save is: ', self.do_save)
		return

	def plot(self, do_plot=True):
		if self.do_save:
			mask, mean, var, err = self.file(False)

			# Expectation
			plt.figure()
			plt.plot(mean)
			plt.xlabel('step id')
			plt.ylabel('mean')

			# Error
			plt.figure()
			plt.plot(err)
			plt.fill_between(range(len(err)), err-var, err+var, alpha=.5)
			plt.xlabel('step id')
			plt.ylabel('error')

			# Mask Ani
			fig = plt.figure()
			plt.title('mask (white = active, black = inactive)')
			plt.ylabel('particle')
			plt.xlabel('state')
			plt.set_cmap('gray')
			func = lambda i: plt.imshow(mask[i])
			frames = len(mask)
			ani = FuncAnimation(fig, func, frames, repeat=False)

			plt.show()
		else:
			print('Warning - Cannot do_plot as do_save is: ', self.do_save)
		return

	def batch(self, model0, iterations=None, do_ani=False, agents=None):
		if iterations is None:
			iterations = model0.batch_iterations
		for i in range(iterations):
			model0.step()
			self.step(model0.get_state())
			if do_ani:
				self.ani(model0, agents)
		self.plot()
		# self.file()  # not needed example batching
		return

	def ani(self, model0, agents=None):
		plt.clf()
		fig = plt.figure(1)
		[model.ani(agents=agents, colour='r', alpha=.3) for model in self.models]
		model0.ani()
		plt.pause(1/30)
		return


if __name__ == '__main__':
	import sys
	sys.path.append("..")
	from models.screensaver import Model
	model = Model()
	pf = ParticleFilter(model, particles=10, window=2, do_copies=False, do_save=True)
	pf.batch(model, iterations=50, do_ani=True, agents=None)
