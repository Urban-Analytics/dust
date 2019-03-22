# StationSim (pronounced Mike's Model)
'''
A genuinely interacting agent based model.

TODO:
	time scaling  1) time_id, step_id  2) dt*new_location
	Add gates too animation
	repr

@classmethod -> global methods
parametric_study
lerp5
markersizescale

'''
# todo
# fixme
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import names

# Agent object and methods
class Agent:

	def __init__(self, model, unique_id):
		self.unique_id = unique_id
		# Required
		self.name = names.get_full_name()
		self.status = 0  # 0 Not Started, 1 Active, 2 Finished
		# Location
		self.location = model.loc_entrances[np.random.randint(model.entrances)]
		self.location[1] += model.entrance_space * (np.random.uniform() - .5)
		self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
		# Parameters
		self.speed_desire = 0
		while self.speed_desire <= model.speed_min:
			self.speed_desire = np.random.normal(model.speed_desire_mean, model.speed_desire_std)
		self.wiggle = min(model.wiggle, self.speed_desire)
		self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)
		self.time_activate = int(np.random.exponential(model.entrance_speed * self.speed_desire))
		if model.do_save:
			self.wiggles = 0  # number of wiggles this agent has experienced
			self.collisions = 0  # number of speed limitations/collisions this agent has experienced
			self.history_loc = []
		return

	def __repr__(self):
		print('sspmm Agent: {}'.format(self.unique_id))
		print('Object ID: {}'.format(hex(id(self))))
		return ''


def agent_step(agent, model):
	if agent.status == 0:
		activate(agent, model)
	elif agent.status == 1:
		move(agent, model)
		exit_query(agent, model)
		save(agent, model)
	return

def activate(agent, model):
	if not agent.status and model.time_id > agent.time_activate:
		agent.status = 1
		model.pop_active += 1
		agent.time_start = model.time_id
	return

def move(agent, model):
	for speed in agent.speeds:
		# Direct
		new_location = lerp(agent.loc_desire, agent.location, speed)
		if not collision(agent, model, new_location):
			break
		else:
			if model.do_save:
				agent.collisions += 1
		if speed == agent.speeds[-1]:
			if model.do_save:
				agent.wiggles += 1
			# Wiggle
			new_location = agent.location + agent.wiggle*np.random.randint(-1, 1 +1, 2)
			# Rebound
			if not is_within_bounds(model, new_location):
				new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])

	# Move
	agent.location = new_location
	return

def is_within_bounds(model, new_location):
	return all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])

def collision(agent, model, new_location):
	if not is_within_bounds(model, new_location):
		collide = True
	elif neighbourhood(agent, model, new_location):
		collide = True
	else:
		collide = False
	return collide

def neighbourhood(agent, model, new_location):
	neighbours = False
	neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
	for neighbouring_agent in neighbouring_agents:
		agent2 = model.agents[neighbouring_agent]
		if agent2.status == 1 and agent.unique_id != agent2.unique_id and new_location[0] <= agent2.location[0]:
			neighbours = True
			break
	return neighbours

sqrt2 = np.sqrt(2)
def lerp(loc1, loc2, speed):
	reciprocal_distance = sqrt2 / sum(abs(loc1 - loc2))
	loc = loc2 + speed * (loc1 - loc2) * reciprocal_distance
	return loc

rsqrt2 = 1 / np.sqrt(2)
def exit_query(agent, model):
	if sum(abs(agent.location - agent.loc_desire)) * rsqrt2 < model.exit_space:
		agent.status = 2
		model.pop_active -= 1
		model.pop_finished += 1
		if model.do_save:
			time_delta = model.time_id - agent.time_start
			model.time_taken.append(time_delta)
			time_delta -= (np.linalg.norm(agent.location - agent.loc_desire) - model.exit_space) / agent.speed_desire
			model.time_delay.append(time_delta)
	return

def save(agent, model):
	if model.do_save:
		agent.history_loc.append(agent.location)
	return


# Model object and methods
class Model:

	def __init__(self, params=dict()):
		self.unique_id = None
		# Default Params
		self.width = 200
		self.height = 100
		self.pop_total = 100
		self.entrances = 3
		self.entrance_space = 2
		self.entrance_speed = 4
		self.exits = 2
		self.exit_space = 1
		self.speed_min = .1
		self.speed_desire_mean = 1
		self.speed_desire_std = 1
		self.separation = 4
		self.wiggle = 1
		self.batch_iterations = 2_000
		self.do_save = False
		self.do_plot = False
		self.do_print = True
		self.do_ani = False
		# Dictionary Params Edit
		self.params = params
		[setattr(self, key, value) for key, value in self.params.items()]
		# Functional Params
		self.speed_step = (self.speed_desire_mean - self.speed_min) / 3  # 3 - Average number of speeds to check
		self.boundaries = np.array([[0, 0], [self.width, self.height]])
		# Model Variables
		self.time_id = 0
		self.step_id = 0
		self.pop_active = 0
		self.pop_finished = 0
		self.loc_entrances = None
		self.loc_exits = None
		initialise_gates(self)
		self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
		self.tree = None
		if self.do_save:
			self.time_taken = []
			self.time_delay = []
		return

	def __repr__(self):
		print('sspmm Model {}'.format(self.unique_id))
		[print(key, value) for key, value in self.params.items()]
		print('Object ID: {}'.format(hex(id(self))))
		return ''

def step(model):
	if model.pop_finished < model.pop_total and model.step_id:
		kdtree_build(model)
		[agent_step(agent, model) for agent in model.agents]
	model.time_id += 1
	model.step_id += 1
	mask(model)
	return

def initialise_gates(model):
	# Entrances
	model.loc_entrances = np.zeros((model.entrances, 2))
	model.loc_entrances[:, 0] = 0
	if model.entrances == 1:
		model.loc_entrances[:, 1] = model.height / 2
	else:
		model.loc_entrances[:, 1] = np.linspace(model.height / 4, 3 * model.height / 4, model.entrances)
	# Exits
	model.loc_exits = np.zeros((model.exits, 2))
	model.loc_exits[:, 0] = model.width
	if model.exits == 1:
		model.loc_exits[0, 1] = model.height / 2
	else:
		model.loc_exits[:, 1] = np.linspace(model.height / 4, 3 * model.height / 4, model.exits)
	return

def kdtree_build(model):
	state = agents2state(model, do_ravel=False)
	model.tree = cKDTree(state)
	return

def agents2state(model, do_ravel=True):
	state = [agent.location for agent in model.agents]
	if do_ravel:
		state = np.ravel(state)
	else:
		state = np.array(state)
	return state

def state2agents(model, state, noise=False):
	for i, agent in enumerate(model.agents):
		agent.location = state[2 * i:2 * i + 2]
		if noise:
			agent.location += np.random.normal(0, noise, size=2)
	return

def mask(model):
	mask = np.array([agent.status == 1 for agent in model.agents])
	active = np.sum(mask)
	mask = np.ravel(np.stack([mask, mask], axis=1))  # Two pieces of data per agent, not none agent data in state
	return mask, active

def batch(model):
	for i in range(model.batch_iterations):
		step(model)
		if model.do_ani:
			ani(model)
		if model.pop_finished == model.pop_total:
			if model.do_print:
				print('Everyone made it!')
			break
	if model.do_save:
		if model.do_print:
			save_stats(model)
		if model.do_plot:
			save_plot(model)
	return

def ani(model, agents=None, colour='k', alpha=1, show_separation=True):
    # Design for use in PF
    wid = 8  # image size
    hei = wid * model.height / model.width
    if show_separation:
        # the magic formular for marksize scaling
        magic = 1.8  # dependant on the amount of figure space used
        markersizescale = magic*72*hei/model.height
    plt.figure(1, figsize=(wid, hei))
    plt.clf()
    plt.axis(np.ravel(model.boundaries, 'F'))
    plt.axes().set_aspect('equal')
    for agent in model.agents[:agents]:
        if agent.status == 1:
            if show_separation:
                plt.plot(*agent.location, marker='.', markersize=markersizescale*model.separation, color=colour, alpha=.05)
            plt.plot(*agent.location, marker='.', markersize=2, color=colour, alpha=alpha)
    plt.xlabel('Corridor Width')
    plt.ylabel('Corridor Height')
    plt.pause(1 / 30)
    return

def save_plot(model):
	# Trails
	plt.figure()
	for agent in model.agents:
		if agent.status == 0:
			colour = 'r'
		elif agent.status == 1:
			colour = 'b'
		else:
			colour = 'm'
		locs = np.array(agent.history_loc).T
		plt.plot(locs[0], locs[1], color=colour, linewidth=.5)
	plt.axis(np.ravel(model.boundaries, 'F'))
	plt.xlabel('Corridor Width')
	plt.ylabel('Corridor Height')
	plt.legend(['Agent trails', 'Finished Agents'])
	# Time Taken, Delay Amount
	plt.figure()
	plt.hist(model.time_taken, alpha=.5, label='Time taken')
	plt.hist(model.time_delay, alpha=.5, label='Time delay')
	plt.xlabel('Time')
	plt.ylabel('Number of Agents')
	plt.legend()

	plt.show()
	return

def save_stats(model):
	print()
	print('Stats:')
	print('    Finish Time: ' + str(model.time_id))
	print('    Active / Finished / Total agents: ' + str(model.pop_active) + '/' + str(model.pop_finished) + '/' + str(model.pop_total))
	print('    Average time taken: {:.2f}s'.format(np.mean(model.time_taken)))
	print('    Average time delay: {:.2f}s'.format(np.mean(model.time_delay)))
	print('    Interactions/Agent: {:.2f}'.format(np.mean([agent.collisions for agent in model.agents])))
	print('    Wiggles/Agent: {:.2f}'.format(np.mean([agent.wiggles for agent in model.agents])))
	return

def parametric_study():
	import time
	print('CPU Time (seconds), Time Taken (steps), Time Delay (steps), |, Interactions (per Agent), Wiggles (per Agent), |, None Default Params')
	for pop, sep in [(100, 4), (300, 3), (700, 2)]:
		t = time.time()
		params = {
			'pop_total': pop,
			'separation': sep,
			}
		model = Model(dict(params, **{'do_save': True, 'do_print': False}))
		batch(model)
		print('{:.2f}, {:.2f}, {:.2f}, |, {:.2f}, {:.2f}, |, '.format(time.time()-t, np.mean(model.time_taken), np.mean(model.time_delay), np.mean([agent.collisions for agent in model.agents]), np.mean([agent.wiggles for agent in model.agents]))+str(params))
	return

if __name__ == '__main__':
	parametric_study()
