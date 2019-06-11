#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 10:48:07 2019

@author: rob
"""
"""
KM's StationSim slightly modified for UKF.


Rob's Additions:
    -ideal_location for the transition step (definitely need this)
    -wiggle counters (at nick's request)
"""
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import datetime

sqrt2 = np.sqrt(2)  # required for Agent.lerp()

class Agent:
    """
    A class representing a generic agent for the StationSim ABM.
    """
    def __init__(self, model, unique_id):
        """
        Initialise a new agent.
        Creates a new agent and gives it a randomly chosen entrance, exit, and
        desired speed. All agents start with active state 0 ('not started').
        Their initial location (** HOW IS LOCATION REPRESENTED?? Answer: With (x,y) tuple-floats **) is set
        to the location of the entrance that they are assigned to.
        :param model: a pointer to the station sim model that is creating this agent
        """
        # Required
        self.unique_id = unique_id
        self.active = 0  # 0 Not Started, 1 Active, 2 Finished
        model.pop_active += 1

        # Choose at random at which of the entrances the agent starts
        self.location = model.loc_entrances[np.random.randint(model.entrances)] #pick an entrance
        self.location[1] += model.entrance_space * (np.random.uniform() - .5) #wiggle between +-1 a bit about the entrance
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)] #pick an exit
        self.ideal_location = self.location #Rob's addition. Where it would be without collisions. Useful prior prediction for each agent
        self.time_activate = np.random.exponential(model.entrance_speed) #when do agents enter model
        # The maximum speed that this agent can travel at:
        # self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)  # this is not a truncated normal distribution
        self.speed_desire = model.speed_min - 1
        while self.speed_desire <= model.speed_min:
            self.speed_desire = np.random.normal(model.speed_desire_mean, model.speed_desire_std)
        self.wiggle = min(self.speed_desire, model.wiggle)  # if they can wiggle faster than they can move they may beat the expected time
        self.wiggle_count = 0 #how many wiggles per agent per run
        # A few speeds to check; used if a step at the max speed would cause a collision
        self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)
        if model.do_save:
            self.history_loc = []
        self.time_expected = None
        self.time_start = None

    def step(self, model):
        """
        Iterate the agent. If they are inactive then it checks to see if they
        should become active. If they are active then then move (see
        self.move()) and, possibly, leave the model (see exit_query())).
        """
        if self.active == 0:
            self.activate(model)
        elif self.active == 1:
            self.move(model)
            self.exit_query(model)
            self.save(model)
  

    def activate(self, model):
        """
        Test whether an agent should become active. This happens when the model
        time is greater than the agent's activate time.
        """
        if not self.active and model.time_id > self.time_activate:
            self.active = 1
            self.time_start = model.time_id
            self.time_expected = (np.linalg.norm(self.location - self.loc_desire) - model.exit_space) / self.speed_desire

    def is_within_bounds(self, boundaries, new_location):
        """
        Check if new location is within the bounds of the model.
        """
        within0 = all(boundaries[0] <= new_location)
        within1 = all(boundaries[1] <= new_location)
        return within0 and within1

    def move(self, model):
        """
        Move the agent towards their destination. If the way is clear then the
        agent moves the maximum distance they can given their maximum possible
        speed (self.speed_desire). If not, then they iteratively test smaller
        and smaller distances until they find one that they can travel to
        without causing a colision with another agent.
        """
        
        
        for speed in self.speeds:
            # Direct
            new_location = Agent.lerp(self.loc_desire, self.location, speed)
            if speed == self.speed_desire:
                self.ideal_location = new_location

            if not Agent.collision(model, new_location):
                break
            elif speed == self.speeds[-1]:
                # Wiggle
                new_location = self.location + self.wiggle*np.random.randint(-1, 1+1, 2)
                self.wiggle_count+=1 #rob's addition adds 1 to the wiggle counter
        # Rebound
        if not self.is_within_bounds(model.boundaries, new_location):
            new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
        # Move
        self.location = new_location

    @classmethod
    def collision(cls, model, new_location):
        """
        Detects whether a move to the new_location will cause a collision
        (either with the model boundary or another agent).
        """
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            collide = True
        elif Agent.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    @classmethod
    def neighbourhood(cls, model, new_location):
        """
        XXXX WHAT DOES THIS DO??  Answer: This method finds whether or not nearby neighbours are a collision.
         :param model:        the model that this agent is part of
         :param new_location: the proposed new location that the agent will move to
                         (a XXXX - what kind of object/data is the location?  Answer: the standard (x,y) floats-tuple)
         :param do_kd_tree    whether to use a spatial index (kd_tree) (default true)
        """
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.active == 1 and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    @classmethod
    def lerp(cls, loc1, loc2, speed):
        """
        lerp - linear extrapolation
        Find the new position of after moving 'speed' distance from loc2 towards loc1.
            :param loc1: desired location
            :param loc2: current location
            :param speed: distance that can be covered in an iteration
            :return: The new location
        lerp is a intensively used method hence profiling and adjustments have been made, see 'github dust/Projects/awest/code/experiments/lerp.py' for more understanding.
        """
        #distance = np.linalg.norm(loc1 - loc2)     # frobenius norm: profiled at 8.05μs
        #distance = np.sqrt(sum((loc1 - loc2)**2))  # euclidean norm: profiled at 7.82μs
        #distance = sum(abs(loc1 - loc2))           # manhattan norm: profiled at 6.19μs
        reciprocal_distance = sqrt2 / sum(abs(loc1 - loc2))  # lerp5: profiled at 6.41μs
        loc = loc2 + speed * (loc1 - loc2) * reciprocal_distance
        return loc

    def exit_query(self, model):
        """
        Determine whether the agent should leave the model and, if so,
        remove them. Otherwise do nothing.
        """
        if sum(abs(self.location - self.loc_desire)) / sqrt2 < model.exit_space:
            self.active = 2
            model.pop_active -= 1
            model.pop_finished += 1
            if model.do_save:
                time_delta = model.time_id - self.time_start
                model.time_taken.append(time_delta)
                time_delta -= self.time_expected
                model.time_delay.append(time_delta)

    def save(self, model):
        """
        Save agent location.
        """
        if model.do_save:
            self.history_loc.append(self.location)


class Model:
    """
    A class to represent the StationSim model.
    """
    def __init__(self, params):
        """
        Create a new model, reading parameters from a dictionary.
        XXXX Need to document the required parameters.
        """
        self.params = params
        # There are a lot of required attributes here that we hope are in params
        # Perhaps we should have a way to ensure we get what we require?
        # Also, consider using **kwargs
        [setattr(self, key, value) for key, value in params.items()]
        # Average number of speeds to check
        self.speed_step = (self.speed_desire_mean - self.speed_min) / 3
        # Batch Details
        self.time_id = 0
        self.step_id = 0
        if self.do_save:
            self.time_taken = []
            self.time_delay = []
        # Model Parameters
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.initialise_gates()
        self.agents = [Agent(self, unique_id) for unique_id in range(self.pop_total)]

    def step(self):
        """
        Iterate model forward one step.
        """
        if self.pop_finished < self.pop_total and self.step:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        self.time_id += 1
        self.step_id += 1

    def initialise_gates(self):
        """
        Initialise the locations of the entrances and exits.
        """
        self.loc_entrances = self.initialise_gates_generic(self.entrances, 0)
        self.loc_exits = self.initialise_gates_generic(self.exits, self.width)

    def initialise_gates_generic(self, n_gates, x):
        """
        General method for initialising gates.
        Note: This method relies on a lot of class attributes, many of which are
        not explicitly required in the init method - perhaps we should be
        careful of this?  Answer: see note cm at top
        """
        gates = np.zeros((n_gates, 2))
        gates[:, 0] = x
        if n_gates == 1:
            gates[0, 1] = self.height / 2
        else:
            gates[:, 1] = np.linspace(self.height / 4, 3 * self.height / 4,
                                      n_gates)
        return gates

    def kdtree_build(self):
        """
        Build kdtree for the model.
        """
        state = self.agents2state(do_ravel=False)
        self.tree = cKDTree(state)

    def agents2state(self, do_ravel=True):
        """
        Convert list of agents in model to state vector.
        """
        state = [agent.location for agent in self.agents]
        state = np.ravel(state) if do_ravel else np.array(state)
        return state

    def state2agents(self, state):
        """
        Use state vector to set agent locations.
        """
        for i in range(len(self.agents)):
            self.agents[i].location = state[2 * i:2 * i + 2]

    def batch(self):
        """
        Run the model.
        """
        print("Starting batch mode with following parameters:")
        print('\tParameter\tValue')
        for k, v in self.params.items():
            print('\t{0}:\t{1}'.format(k, v))
        print('\n')
        for i in range(self.batch_iterations):
            self.step()
            if i % 100 == 0:
                print("\tIterations: ", i)
            if self.do_ani:
                self.ani()
            if self.pop_finished == self.pop_total:
                print('Everyone made it!')
                break
        print("Finished at iteration", i)
        if self.do_save:
            self.save_stats()
            if self.do_plot: self.save_plot()

    def ani(self, agents=None, colour='k', alpha=1, show_separation=True):
        # Design for use in PF
        wid = 8  # image size
        hei = wid * self.height / self.width
        if show_separation:
            # the magic formular for marksize scaling
            magic = 1.8  # dependant on the amount of figure space used
            markersizescale = magic*72*hei/self.height
        plt.figure(1, figsize=(wid, hei))
        plt.clf()
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.axes().set_aspect('equal')
        for agent in self.agents[:agents]:
            if agent.active == 1:
                if show_separation:
                    plt.plot(*agent.location, marker='s', markersize=markersizescale*self.separation, color=colour, alpha=.05)
                plt.plot(*agent.location, marker='.', markersize=2, color=colour, alpha=alpha)
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.pause(1 / 30)
        return

    def save_ani(self):
        return

    def save_plot(self):
        """
        Produce plots for model.
        """
        self.plot_trails()
        self.plot_agent_times()

    def plot_trails(self):
        """
        Produce a plot of the trails of each agent in the 2-d corridor.
        """
        # Trails
        plt.figure()
        for agent in self.agents:
            if agent.active == 0:
                colour = 'r'
            elif agent.active == 1:
                colour = 'b'
            else:
                colour = 'm'
            locs = np.array(agent.history_loc).T
            plt.plot(locs[0], locs[1], color=colour, linewidth=.5)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.legend(['Agent trails'])
        plt.show()

    def plot_agent_times(self):
        """
        Produce a plot of the time taken by each agent, and the delay of each
        agent.
        """
        # Time Taken, Delay Amount
        plt.figure()
        plt.hist(self.time_taken, alpha=.5, label='Time taken')
        plt.hist(self.time_delay, alpha=.5, label='Time delay')
        plt.xlabel('Time')
        plt.ylabel('Number of Agents')
        plt.legend()
        plt.show()

    def save_stats(self):
        """
        Print model run stats to console.
        """
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time_id))
        print('Active / Finished / Total agents: ' +
              str(self.pop_active) + '/' + str(self.pop_finished) +
              '/' + str(self.pop_total))
        print('Average time taken: {:.2f} steps'.format(np.mean(self.time_taken)))
        print('Average time delay: {:.2f} steps'.format(np.mean(self.time_delay)))

    def __repr__(self):
        """Print this model's ID and its memory location"""
        return "StationSim [{}]".format(hex(id(self)))

    @classmethod
    def run_defaultmodel(cls):
        """
        Run a model with some common parameters. Mostly used for testing.
        """
        np.random.seed(42)
        model_params = {
            'width': 200,
            'height': 100,
            'pop_total': 300,
            'entrances': 3,
            'entrance_space': 2,
            'entrance_speed': 1,
            'exits': 2,
            'exit_space': 1,
            'speed_min': .1,
            'speed_desire_mean': 1,
            'speed_desire_std': 1,
            'separation': 4,
            'wiggle': 1,
            'batch_iterations': 10_000,
            'do_save': True,
            'do_plot': True,
            'do_ani': False
        }
        # Run the model
        Model(model_params).batch()


# If this is called from the command line then run a default model.
if __name__ == '__main__':
    
    
    Model.run_defaultmodel()