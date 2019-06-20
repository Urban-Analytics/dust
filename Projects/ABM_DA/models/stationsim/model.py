"""
Model.py
@author: ksuchak1990
date_created: 19/04/10
A file for a class to represent the StationSim model.
"""

# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from agent import Agent

# Class
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
        self.agents = [Agent(self, u_id) for u_id in range(self.pop_total)]
        self.state_history = list()
        self.state = None

    def step(self):
        """
        Iterate model forward one step.
        """
        if self.pop_finished < self.pop_total and self.step:
            self.kdtree_build()
            [agent.step(self) for agent in self.agents]
        self.state_history.append(self.get_state())
        self.state = self.get_state()
        self.time_id += 1
        self.step_id += 1

    def initialise_gates(self):
        """
        Initialise the locations of the entrances and exits.
        """
        self.loc_entrances = self.initialise_gates_generic(self.height,
                                                           self.entrances, 0)
        self.loc_exits = self.initialise_gates_generic(self.height,
                                                       self.exits, self.width)

    @staticmethod
    def initialise_gates_generic(height, n_gates, x):
        """
        General method for initialising gates.
        Note: This method relies on a lot of class attributes, many of which are
        not explicitly required in the init method - perhaps we should be
        careful of this?  Answer: see note cm at top
        """
        gates = np.zeros((n_gates, 2))
        gates[:, 0] = x
        if n_gates == 1:
            gates[0, 1] = height / 2
        else:
            gates[:, 1] = np.linspace(height / 4, 3 * height / 4, n_gates)
        return gates

    def is_within_bounds(self, new_location):
        """
        Utility function to check whether new_location is within model bounds.
        """
        within0 = all(self.boundaries[0] <= new_location)
        within1 = all(new_location <= self.boundaries[1])
        return within0 and within1

    def kdtree_build(self):
        """
        Build kdtree for the model.
        """
        state = self.get_state(do_ravel=False)
        self.tree = cKDTree(state)

    def get_state(self, do_ravel=True):
        """
        Convert list of agents in model to state vector.
        """
        state = [agent.location for agent in self.agents]
        state = np.ravel(state) if do_ravel else np.array(state)
        return state

    def set_state(self, state):
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
            if self.do_plot:
                self.save_plot()

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
                    plt.plot(*agent.location, marker='s',
                             markersize=markersizescale*self.separation,
                             color=colour, alpha=.05)
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
            'wiggle': 1,
            'batch_iterations': 900,
            'do_save': True,
            'do_plot': False,
            'do_ani': False
        }
        # Run the model
        Model(model_params).batch()


# If this is called from the command line then run a default model.
if __name__ == '__main__':
    Model.run_defaultmodel()
