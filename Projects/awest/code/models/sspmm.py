# sspmm
'''
A genuine Agent-Based Model designed to contain many ABM features.
v7.3 (lit)
'''
# Imports
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


# Agent
class Agent:

	def __init__(self, model, unique_id):
		self.unique_id = unique_id
		# Required
		self.status = 0  # 0 Not Started, 1 Active, 2 Finished
		# Location
		self.loc_start = model.loc_entrances[np.random.randint(model.entrances)]
		self.loc_start[1] += model.entrance_space * np.random.uniform(-1,+1)
		self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
		self.location = self.loc_start
		# Parameters
		self.speed_max = 0
		while self.speed_max <= model.speed_min:
			self.speed_max = np.random.normal(model.speed_mean, model.speed_std)
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
		norm = (x*x + y*y)**.5
		# The default np.linalg.norm(loc1-loc2) was not use because it took 2.45s while this method took 1.71s.
		return norm

	def move(self, model):
		direction = (self.loc_desire - self.location) / self.distance(self.loc_desire, self.location)
		for speed in self.speeds:
			# Direct
			new_location = self.location + speed * direction
			if not self.collision(model, new_location):
				break
			else:
				if model.do_save:
					self.collisions += 1
			# Wiggle
			if speed == self.speeds[-1]:
				if model.do_save:
					self.wiggles += 1
				new_location = self.location + self.wiggle*np.random.randint(-1, 1+1, 2)
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
		# if model.width-self.location[0] < model.exit_space:  # saves a small amount of time
		if self.distance(self.location, self.loc_desire) < model.exit_space:
			self.status = 2
			model.pop_active -= 1
			model.pop_finished += 1
			if model.do_save:
				time_delta = model.time - self.time_start
				model.time_taken.append(time_delta)
				time_delta -= (self.distance(self.loc_start, self.loc_desire) - model.exit_space) / self.speed_max
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
			'entrance_space': 1,
			'entrance_speed': 4,
			'exits': 2,
			'exit_space': 1,
			'speed_min': .1,
			'speed_mean': 1,
			'speed_std': 1,
			'speed_steps': 3,
			'separation': 5,
			'max_wiggle': 1,
			'iterations': 1_800,
			'do_save': False,
			'do_plot': False,
			'do_print': True,
			'do_ani': False
			}
		# Params Edit
		self.params0 = dict()
		for key in params.keys():
			if key in self.params:
				if self.params[key] is not params[key] and 'do_' not in key:
					self.params0[key] = params[key]
				self.params[key] = params[key]
			else:
				print('BadKeyWarning: {} is not a model parameter.'.format(key))
		[setattr(self, key, value) for key, value in self.params.items()]
		# Constants
		self.speed_step = (self.speed_mean - self.speed_min) / self.speed_steps
		self.boundaries = np.array([[0, 0], [self.width, self.height]])
		init_gates = lambda x,y,n: np.array([np.full(n,x), np.linspace(0,y,n+2)[1:-1]]).T
		self.loc_entrances = init_gates(0, self.height, self.entrances)
		self.loc_exits = init_gates(self.width, self.height, self.exits)
		# Variables
		self.time = 0
		self.pop_active = 0
		self.pop_finished = 0
		self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
		self.tree = None
		if self.do_save:
			self.time_taken = []
			self.time_delay = []
		self.is_within_bounds = lambda loc: all(self.boundaries[0] <= loc) and all(loc <= self.boundaries[1])
		return

	def step(self):
		if self.pop_finished < self.pop_total and self.time:
			self.tree = cKDTree([agent.location for agent in self.agents])
			[agent.step(self) for agent in self.agents]
		self.time += 1
		return

	def get_state(self, sensor='location'):
		if sensor is None:
			state = [(agent.status, *agent.location) for agent in self.agents]
			state = np.append(self.time, np.ravel(state))
		elif sensor is 'location':
			state = [agent.location for agent in self.agents]
			state = np.ravel(state)
		return state

	def set_state(self, state, sensor='location', noise=False):
		if sensor is None:
			self.time = int(state[0])
			state = np.reshape(state[1:], (self.pop_total, 3))
			for i, agent in enumerate(self.agents):
				agent.status = int(state[i,0])
				agent.location = state[i,1:]
		elif sensor is 'location':
			state = np.reshape(state, (self.pop_total, 2))
			for i, agent in enumerate(self.agents):
				agent.location = state[i,:]
		return

	def mask(self):
		mask = np.array([agent.status==1 for agent in self.agents])
		active = np.sum(mask)
		mask = np.ravel(np.stack([mask, mask], axis=1))  # Two pieces of data per agent
		return mask, active

	def ani(self, agents=None, colour='k', alpha=1, show_separation=False, show_axis=True):
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
		if not show_axis:
			plt.axis('off')
		return

	def get_plot(self):
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

	def get_stats(self):
		statistics = {
			'Finish Time': self.time,
			'Total': self.pop_total,
			'Active': self.pop_active,
			'Finished': self.pop_finished,
			'Time Taken': np.mean(self.time_taken),
			'Time Delay': np.mean(self.time_delay),
			'Interactions': np.mean([agent.collisions for agent in self.agents]),
			'Wiggles': np.mean([agent.wiggles for agent in self.agents]),
			}
		return statistics

	def batch(self):
		for i in range(self.iterations):
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
				print(self.get_stats())
			if self.do_plot:
				self.get_plot()
		return


# Batches
def animated_batch():
	params = {
		'iterations': 400,
		'do_ani': False,
		'do_save': True,
		'do_print': True,
		'do_plot': True,
		#'false_param': 'expect a warning'
		}
	model = Model(params)
	model.batch()
	return

def parametric_study():
	import time
	analytics = {}

	for pop, sep in [(100, 5), (300, 3), (700, 2)]:
		params = {
			'pop_total': pop,
			'separation': sep,
			}
		t = time.time()
		model = Model({**params, 'do_save':True,'do_print':False})
		model.batch()
		analytics[str(model.params0)] = {
			'Process Time': time.time()-t,
			**model.get_stats()
			}
	for s in (9,5,2,1):
		params = {'speed_steps': s}
		t = time.time()
		model = Model({**params, 'do_save':True,'do_print':False})
		model.batch()
		analytics[str(model.params0)] = {
			'Process Time': time.time()-t,
			**model.get_stats()
			}

	csv_str, lines = '', []
	for i,row in enumerate(analytics):
		if i==0:
			header = ', '.join(k for k,_ in analytics[row].items()) + ',\n'
		line = ', '.join(f'{v}' for _,v in analytics[row].items()) + f', {row}'
		lines.append(line)
	csv_str = header + '\n'.join(lines)
	print(csv_str)
	print(csv_str, file=open('test.csv', 'w'))
	return


if __name__ == '__main__':
	# animated_batch()
	parametric_study()
	pass
