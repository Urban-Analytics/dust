# StationSim (pronounced Mike's Model)
'''
A genuinely interacting agent based model.

TODO:
	ani gates
	ani markersize
	ani save
	difference between pf and pf_km
	removal of stationsim_km
	update sspmm.md?

speed_desire -> speed_max (to fit speed_min)
classmethods out - statics and internal are in
default params are defined using dictionaries as to keiran's plan
gates construtor methods from keiran are used
lerp deleted and lerp_vector increases speed dramatically
norm edited to kerian's improved euclidean distance

TLDR: Speed updates and back to functioning with PF
'''
# Imports
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import names


# Agent
class Agent:

	def __init__(self, model, unique_id):
		self.unique_id = unique_id
		self.name = names.get_full_name()
		# Required
		self.status = 0  # 0 Not Started, 1 Active, 2 Finished
		# Location
		self.location = model.loc_entrances[np.random.randint(model.entrances)]
		self.location[1] += model.entrance_space * (np.random.uniform() - .5)
		self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
		# Parameters
		self.speed_max = 0
		while self.speed_max <= model.speed_min:
			self.speed_max = np.random.normal(model.speed_desire_mean, model.speed_desire_std)
		self.wiggle = min(model.max_wiggle, self.speed_max)
		self.speeds = np.arange(self.speed_max, model.speed_min, -model.speed_step)
		self.time_activate = int(np.random.exponential(model.entrance_speed * self.speed_max))
		if model.do_save:
			self.wiggles = 0  # number of wiggles this agent has experienced
			self.collisions = 0  # number of speed limitations/collisions this agent has experienced
			self.history_loc = []
		return

	def __repr__(self):
		return '\nObject ID: {}, sspmm Agent: {} {}'.format(hex(id(self)), hex(self.unique_id), self.name)

	def step(self, model):
		if self.status == 0:
			self.activate(model)
		elif self.status == 1:
			self.move(model)
			self.exit_query(model)
			self.save(model)
		return

	def activate(self, model):
		if not self.status and model.time > self.time_activate:
			self.status = 1
			model.pop_active += 1
			self.time_start = model.time
		return

	@staticmethod
	def distance(loc1, loc2):
		# Euclidean distance between two 2D points.
		x = loc1[0] - loc2[0]
		y = loc1[1] - loc2[1]
		norm =  (x*x + y*y)**.5
		# The default np.linalg.norm(loc1-loc2) was not use because it took 2.45s while this method took 1.71s.
		return norm

	def move(self, model):
		lerp_vector = (self.loc_desire - self.location) / self.distance(self.loc_desire, self.location)
		for speed in self.speeds:
			# Direct
			new_location = self.location + speed * lerp_vector
			if not self.collision(model, new_location):
				break
			else:
				if model.do_save:
					self.collisions += 1
			if speed == self.speeds[-1]:
				if model.do_save:
					self.wiggles += 1
				# Wiggle
				new_location = self.location + self.wiggle*np.random.randint(-1, 1 +1, 2)
				# Rebound
				if not model.is_within_bounds(new_location):
					new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
		# Move
		self.location = new_location
		return

	def collision(self, model, new_location):
		if not model.is_within_bounds(new_location):
			collide = True
		elif self.neighbourhood(model, new_location):
			collide = True
		else:
			collide = False
		return collide

	def neighbourhood(self, model, new_location):
		neighbours = False
		neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
		for neighbouring_agent in neighbouring_agents:
			agent = model.agents[neighbouring_agent]
			if agent.status == 1 and self.unique_id != agent.unique_id and new_location[0] <= agent.location[0]:
				neighbours = True
				break
		return neighbours

	def exit_query(self, model):
		if self.distance(self.location, self.loc_desire) < model.exit_space:
			self.status = 2
			model.pop_active -= 1
			model.pop_finished += 1
			if model.do_save:
				time_delta = model.time - self.time_start
				model.time_taken.append(time_delta)
				time_delta -= (self.distance(self.location, self.loc_desire) - model.exit_space) / self.speed_max
				model.time_delay.append(time_delta)
		return

	def save(self, model):
		if model.do_save:
			self.history_loc.append(self.location)
		return


# Model
class Model:

	def __init__(self, params=dict()):
		self.unique_id = None
		# Default Params
		self.params = {
			'width': 200,
			'height': 100,
			'pop_total': 100,
			'entrances': 3,
			'entrance_space': 2,
			'entrance_speed': 4,
			'exits': 2,
			'exit_space': 1,
			'speed_min': .1,
			'speed_desire_mean': 1,
			'speed_desire_std': 1,
			'separation': 4,
			'max_wiggle': 1,
			'batch_iterations': 2_000,
			'do_save': False,
			'do_plot': False,
			'do_print': True,
			'do_ani': False
		}
		# Params Edit
		for key in params.keys():
			if key in self.params:
				self.params[key] = params[key]
			else:
				print('BadKeyWarning: {} is not a model parameter.'.format(key))
		[setattr(self, key, value) for key, value in self.params.items()]
		# Functional Params
		self.speed_step = (self.speed_desire_mean - self.speed_min) / 3  # 3 - Average number of speeds to check
		self.boundaries = np.array([[0, 0], [self.width, self.height]])
		# Model Variables
		self.time = 0
		self.pop_active = 0
		self.pop_finished = 0
		self.loc_entrances = None
		self.loc_exits = None
		self.initialise_gates()
		self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
		self.tree = None
		if self.do_save:
			self.time_taken = []
			self.time_delay = []
		return

	def __repr__(self):
		text = 'sspmm Model {}'.format(self.unique_id)
		align_max = max([len(key) for key, _ in self.params.items()])
		align = [align_max-len(key) for key, _ in self.params.items()]
		text += ''.join('\n  {}{}: {}'.format(' '*align[i], key, val) for i, (key, val) in enumerate(self.params.items()))
		text += '\nObject ID: {}'.format(hex(id(self)))
		return text

	def step(self):
		if self.pop_finished < self.pop_total and self.time:
			self.kdtree_build()
			[agent.step(self) for agent in self.agents]
		self.time += 1
		self.mask()
		return

	def initialise_gates(self):
		# Initialise the locations of the entrances and exits.
		self.loc_entrances = self.initialise_gates_generic(self.height, self.entrances, 0)
		self.loc_exits = self.initialise_gates_generic(self.height, self.exits, self.width)
		return

	@staticmethod
	def initialise_gates_generic(height, n_gates, x):
		# General method for initialising gates.
		gates = np.zeros((n_gates, 2))
		gates[:, 0] = x
		if n_gates == 1:
			gates[0, 1] = height/2
		else:
			gates[:, 1] = np.linspace(height/4, 3*height/4, n_gates)
		return gates

	def is_within_bounds(self, new_location):
		return all(self.boundaries[0] <= new_location) and all(new_location <= self.boundaries[1])

	def kdtree_build(self):
		state = self.get_state(do_ravel=False)
		self.tree = cKDTree(state)
		return

	def get_state(self, do_ravel=True):
		state = [agent.location for agent in self.agents]
		if do_ravel:
			state = np.ravel(state)
		else:
			state = np.array(state)
		return state

	def set_state(self, state, noise=False):
		for i, agent in enumerate(self.agents):
			agent.location = state[2*i : 2*i+2]
			if noise:
				agent.location += np.random.normal(0, noise, size=2)
		return

	def mask(self):
		mask = np.array([agent.status==1 for agent in self.agents])
		active = np.sum(mask)
		mask = np.ravel(np.stack([mask, mask], axis=1))  # Two pieces of data per agent
		return mask, active

	def ani(self, agents=None, colour='k', alpha=1, show_separation=False):
		# Design for use in PF
		wid = 8  # image size
		hei = wid * self.height / self.width
		if show_separation:
			# the magic formular for marksize scaling
			magic = 1.9  # dependant on the amount of figure space used
			markersizescale = magic*72*hei/self.height
		plt.figure(1, figsize=(wid, hei))
		plt.axis(np.ravel(self.boundaries, 'F'))
		plt.axes().set_aspect('equal')
		x = np.arange(10)
		for agent in self.agents[:agents]:
			if agent.status == 1:
				if show_separation:
					plt.plot(*agent.location, marker='.', markersize=markersizescale*self.separation, color=colour, alpha=.05)
				plt.plot(*agent.location, marker='.', markersize=2, color=colour, alpha=alpha)
		plt.xlabel('Corridor Width')
		plt.ylabel('Corridor Height')
		return

	def ani_save(self):
		pass

	def save_plot(self):
		# Trails
		plt.subplot(2, 1, 1)
		for agent in self.agents:
			if agent.status == 0:
				colour = 'r'
			elif agent.status == 1:
				colour = 'b'
			else:
				colour = 'm'
			locs = np.array(agent.history_loc).T
			plt.plot(locs[0], locs[1], color=colour, linewidth=.5)
		plt.axis(np.ravel(self.boundaries, 'F'))
		plt.xlabel('Corridor Width')
		plt.ylabel('Corridor Height')
		plt.legend(['Agent trails', 'Finished Agents'])

		# Time Taken, Delay Amount
		plt.subplot(2, 1, 2)
		plt.hist(self.time_taken, alpha=.5, label='Time taken')
		plt.hist(self.time_delay, alpha=.5, label='Time delay')
		plt.xlabel('Time')
		plt.ylabel('Number of Agents')
		plt.legend()

		plt.show()
		return

	def save_stats(self):
		print()
		print('Stats:')
		print('    Finish Time: ' + str(self.time))
		print('    Active / Finished / Total agents: ' + str(self.pop_active) + '/' + str(self.pop_finished) + '/' + str(self.pop_total))
		print('    Average time taken: {:.2f}s'.format(np.mean(self.time_taken)))
		print('    Average time delay: {:.2f}s'.format(np.mean(self.time_delay)))
		print('    Interactions/Agent: {:.2f}'.format(np.mean([agent.collisions for agent in self.agents])))
		print('    Wiggles/Agent: {:.2f}'.format(np.mean([agent.wiggles for agent in self.agents])))
		return

	def batch(self):
		for i in range(self.batch_iterations):
			self.step()
			if self.do_ani:
				plt.clf()
				self.ani(show_separation=True)
				plt.pause(1/30)
			if self.pop_finished == self.pop_total:
				if self.do_print:
					print('Everyone made it!')
				break
		if self.do_save:
			if self.do_print:
				self.save_stats()
			if self.do_plot:
				self.save_plot()
		return


# Batches
def animated_batch():
	params = {
		'batch_iterations': 200,
		'do_ani': True,
		'do_save': True,
		'do_print': False,
		'do_plot': True,
		#'false_param': 'expect a warning'
		}
	model = Model(params)
	model.batch()
	return

def parametric_study():
	import time
	print('Process Time (seconds), Time Taken (steps), Time Delay (steps), |, Interactions (per Agent), Wiggles (per Agent), |, None Default Params')
	for pop, sep in [(100, 4), (300, 3), (700, 2)]:
		t = time.time()
		params = {
			'pop_total': pop,
			'separation': sep,
			}
		model = Model(dict(params, **{'do_save': True, 'do_print': False}))
		model.batch()
		print('{:.2f}, {:.2f}, {:.2f}, |, {:.2f}, {:.2f}, |, '.format(time.time()-t, np.mean(model.time_taken), np.mean(model.time_delay), np.mean([agent.collisions for agent in model.agents]), np.mean([agent.wiggles for agent in model.agents]))+str(params))
	return


if __name__ == '__main__':
	# animated_batch()
	# parametric_study()
	# Model().batch()
	pass
