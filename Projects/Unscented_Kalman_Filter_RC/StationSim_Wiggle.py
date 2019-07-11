# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def error(text='Self created error.'):
    from sys import exit
    print()
    exit(text)
    return


    
class Agent:

    def __init__(self, model, unique_id):
        """
        Initialise a new agent.
        
        PARAMETERS
          - model: a pointer to the station sim model that is creating this agent
        
        DESCRIPTION
        Creates a new agent and gives it a randomly chosen entrance, exit, and 
        desired speed. All agents start with active state 0 ('not started').
        Their initial location (** HOW IS LOCATION REPRESENTED?? **) is set
        to the location of the entrance that they are assigned to.
        """
        # Required
        self.unique_id = unique_id
        self.active = 0  # 0 Not Started, 1 Active, 2 Finished
        model.pop_active += 1
        # Location
        self.location = model.loc_entrances[np.random.randint(model.entrances)]
        self.location[1] += model.entrance_space * (np.random.uniform() - .5) # XXXX WHAT DOES THIS DO?
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]
        # Parameters
        self.time_activate = np.random.exponential(model.entrance_speed) # XXXX WHAT DOES THIS DO?
        self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)
        self.speeds = np.arange(self.speed_desire, model.speed_min, -model.speed_step)# XXXX WHAT DOES THIS DO?
        self.wiggle = 0
        if model.do_save:
            self.history_loc = []
            self.history_loc.append(self.location)
        return

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
        return

    def activate(self, model):
        """
        Test whether an agent should become active. This happens when the model
        time is greater than the agent's activate time.
        """
        if not self.active and model.time_id > self.time_activate:
            self.active = 1
            self.time_start = model.time_id
            self.time_expected = np.linalg.norm(self.location - self.loc_desire) / self.speed_desire
        return

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
            new_location = self.lerp(self.loc_desire, self.location, speed)
            if not self.collision(model, new_location):
                self.wiggle=0 #not wiggling
                break
            elif speed == self.speeds[-1]:
                self.wiggle=1 # Wiggle
                new_location = self.location + np.random.randint(-1, 1+1, 2)
        # Rebound
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
        # Move
        self.location = new_location
        return

    def collision(self, model, new_location):
        """
        Detects whether a move to the new_location will cause a colision 
        (either with the model boundary or another agent).
        """
        within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
        if not within_bounds:
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location, do_kd_tree=True):
        """
        XXXX WHAT DOES THIS DO??
        
        PARMETERS
         - model:        the model that this agent is part of
         - new_location: the proposed new location that the agent will move to
                         (a XXXX - what kind of object/data is the location?)
         - do_kd_tree    whther to use a spatial index (kd_tree) (default true)
        """
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
        """
        Determine whether the agent should leave the model and, if so, 
        remove them. Otherwise do nothing.
        """
        if np.linalg.norm(self.location - self.loc_desire) < model.exit_space:
            self.active = 2
            model.pop_active -= 1
            model.pop_finished += 1
            self.wiggle=0
            if model.do_save:
                time_delta = model.time_id - self.time_start
                model.time_taken.append(time_delta)
                time_delta -= self.time_expected
                model.time_delayed.append(time_delta)
        return

    def save(self, model):
        if model.do_save:
            self.history_loc.append(self.location)
        return


class Model:

    def __init__(self, params):
        """
        Create a new model, reading parameters from a dictionary. 
        
        XXXX Need to document the required parameters.
        """
        self.params = (params,)
        [setattr(self, key, value) for key, value in params.items()]
        self.speed_step = (self.speed_desire_mean - self.speed_min) / 3  # Average number of speeds to check
        # Batch Details
        self.time_id = 0
        self.step_id = 0
        if self.do_save:
            self.time_taken = []
            self.time_delayed = []
        # Model Parameters
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.initialise_gates()
        self.agents = list([Agent(self, unique_id) for unique_id in range(self.pop_total)])
        return

    def step(self):
        if self.pop_finished < self.pop_total and self.step:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        self.time_id += 1
        self.step_id += 1
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
        """
        Run the model.
        """
        for i in range(self.batch_iterations):
            self.step()
            if self.do_ani:
                self.ani()
            if self.pop_finished == self.pop_total:
                print('Everyone made it!')
                break
        if self.do_save:
            self.save_stats()
            self.save_plot()
        return

    def ani(self):
        plt.figure(1)
        plt.clf()
        for agent in self.agents:
            if agent.active == 1:
                plt.plot(*agent.location, '.k')#, markersize=4)
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.xlabel('Corridor Width')
        plt.ylabel('Corridor Height')
        plt.pause(1 / 30)
        return

    def save_ani(self):
        return

    def save_plot(self):
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
        # Time Taken, Delay Amount
        plt.figure()
        plt.hist(self.time_taken, alpha=.5, label='Time taken')
        plt.hist(self.time_delayed, alpha=.5, label='Time delay')
        plt.xlabel('Time')
        plt.ylabel('Number of Agents')
        plt.legend()

        plt.show()
        return

    def save_stats(self):
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time_id))
        print('Active / Finished / Total agents: ' + str(self.pop_active) + '/' + str(self.pop_finished) + '/' + str(self.pop_total))
        print('Average time taken: ' + str(np.mean(self.time_taken)) + 's')
        return
    
    
    @classmethod
    def run_defaultmodel(cls):
        """
        Run a model with some common parameters. Mostly used for testing.
        """
        model_params = {
            'width': 200,
            'height': 100,
            'pop_total': 700,
            'entrances': 3,
            'entrance_space': 2,
            'entrance_speed': .1,
            'exits': 2,
            'exit_space': 1,
            'speed_min': .1,
            'speed_desire_mean': 1,
            'speed_desire_std': 1,
            'separation': 2,
            'batch_iterations': 900,
            'do_save': True,
        'do_ani': False,
        }
        # Run the model
        Model(model_params).batch()
