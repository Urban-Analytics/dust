# StationSim (pronounced Mike's Model)
'''
A interacting agent based model.
'''
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, model, unique_id):
        self.unique_id = unique_id
        self.active = False
        self.finished = False
        self.activate_time = np.random.exponential(model.entrance_speed)
        model.pop_active += 1
        # Location
        self.location = model.loc_entrances[np.random.randint(model.entrances)]
        self.location[1] += model.entrance_space * (np.random.uniform() - .5)
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
        # Parameters
        self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), model.speed_min)
        if model.do_save:
            self.start_time = model.time
            self.history_loc = []
        return

    def step(self, model):
        if self.active:
            self.move(model)
            self.exit_query(model)
            self.save(model)
        else:
            self.activate(model)
        return

    def activate(self, model):
        if not self.finished and model.time > self.activate_time:
            self.active = True
        return

    def move(self, model):
        speeds = np.linspace(self.speed_desire, model.speed_min, 5)
        for speed in speeds:
            if speed == model.speed_min:
                # Wiggle
                new_location = self.location + np.random.random_integers(-1, +1, 2)
            else:
                # Direct
                new_location = self.lerp(self.loc_desire, self.location, speed)
                if not self.collision(model, new_location):
                    break
        # Rebound
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
        # Move
        self.location = new_location
        return

    def collision(self, model, new_location):
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location, do_kd_tree=True):
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.active == 1 and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    def lerp(self, loc1, loc2, speed):
        distance = np.linalg.norm(loc1 - loc2)
        loc = loc2 + speed * (loc1 - loc2) / distance
        return loc

    def exit_query(self, model):
        if np.linalg.norm(self.location - self.loc_desire) < model.exit_space:
            self.active = False
            self.finished = True
            model.pop_active -= 1
            model.pop_finished += 1
            if model.do_save:
                model.time_taken.append(model.time - self.start_time)
        return

    def save(self, model):
        if model.do_save:
            self.history_loc.append(self.location)
        return


class Model:

    def __init__(self, params):
        self.params = (params,)
        [setattr(self, key, value) for key, value in params.items()]
        # Batch Details
        self.time = 0
        if self.do_save:
            self.time_taken = []
        # Model Parameters
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.initialise_gates()
        self.initialise_agents()
        return

    def step(self):
        self.time += 1
        if self.pop_finished < self.pop_total and self.time:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        return

    def initialise_gates(self):
        # Entrances
        self.loc_entrances = np.zeros((self.entrances, 2))
        self.loc_entrances[:, 0] = 0
        if self.entrances == 1:
            self.loc_entrances[0, 1] = self.height / 2
        else:
            self.loc_entrances[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.entrances)
        # Exits
        self.loc_exits = np.zeros((self.exits, 2))
        self.loc_exits[:, 0] = self.width
        if self.exits == 1:
            self.loc_exits[0, 1] = self.height / 2
        else:
            self.loc_exits[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4, self.exits)
        return

    def initialise_agents(self):
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        return

    def kdtree_build(self):
        state = self.agents2state(do_ravel=False)
        self.tree = cKDTree(state)
        return

    def agents2state(self, do_ravel=True):
        state = [agent.location for agent in self.agents]
        if do_ravel:
            state = np.ravel(state)
        else:
            state = np.array(state)
        return state

    def state2agents(self, state):
        for i in range(len(self.agents)):
            self.agents[i].location = state[2 * i:2 * i + 2]
        return

    def batch(self):
        for i in range(self.batch_iterations):
            self.step()
            if self.do_ani:
                self.ani()
            if self.pop_finished == self.pop_total:
                print('Everyone made it!')
                break
        if self.do_save:
            self.stats()
            self.plot()
        return

    def ani(self):
        plt.figure(1)
        plt.clf()
        for agent in self.agents:
            if agent.active == 1:
                plt.plot(*agent.location, '.k')
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.pause(1 / 30)
        return

    def plot(self):
        plt.figure()
        for agent in self.agents:
            if True or agent.unique_id < 50:
                locs = np.array(agent.history_loc).T
                plt.plot(locs[0], locs[1], linewidth=.5)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.figure()
        plt.hist(self.time_taken)
        plt.show()
        return

    def stats(self):
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time))
        print('Active / Finished / Total agents: ' + str(self.pop_active) + '/' + str(self.pop_finished) + '/' + str(self.pop_total))
        print('Average time taken: ' + str(np.mean(self.time_taken)) + 's')
        return
