'''
StationSim
    author: aw-west
    created: 19/06/18
    version: 0.8.1
Changelog:
    .0 merged models
    .0 batch() moved to experiments
    .0 separated figures
    .1 checked figures and created experiment notebooks
    .1 fixed _heightmap
Todo:
    new entering
    speeds[-1] separation drawn
    set_pickle?
    get_pickle?
    Look at model get_state and set_state - should comparisons be '==' or 'is'?
'''

import warnings
import numpy as np
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# Dont automatically load seaborn as it isn't needed on the HPC
try: 
    from seaborn import kdeplot as sns_kdeplot
except ImportError as e:
    warnings.warn("The seaborn module is not available. If you try to create kde plots for this model (i.e. a wiggle map or density map) then it will fail.")


class Agent:
    '''
    A class representing a generic agent for the StationSim ABM.
    '''
    def __init__(self, model, unique_id):
        '''
        Initialise a new agent.

        Desctiption:
            Creates a new agent and gives it a randomly chosen entrance, exit,
            and desired speed.
            All agents start with active state 0 ('not started').
            Their initial location (** (x,y) tuple-floats **) is set to the location
            of the entrance that they are assigned to.

        Parameters:
            model - a pointer to the StationSim model that is creating this agent
        '''
        self.unique_id = unique_id
        # Required
        self.status = 0  # 0 Not Started, 1 Active, 2 Finished
        # Location
        perturb = model.gates_space * np.random.uniform(-1, +1)
        gate_in = np.random.randint(model.gates_in)
        self.loc_start = model.gates_locations[gate_in] + [0, perturb]
        gate_out = np.random.randint(model.gates_out) + model.gates_in
        self.loc_desire = model.gates_locations[gate_out]
        self.location = self.loc_start
        # Speed
        speed_max = 0
        while speed_max <= model.speed_min:
            speed_max = np.random.normal(model.speed_mean, model.speed_std)
        self.speeds = np.arange(speed_max, model.speed_min, -model.speed_step)
        self.speed = None
        # Others
        self.steps_activate = np.random.exponential(model.gates_speed)
        self.wiggle = min(model.max_wiggle, speed_max)
        # History
        if model.do_history:
            self.history_locations = []
            self.history_speeds = []
            self.history_wiggles = 0
            self.history_collisions = 0
            self.step_start = None

    def step(self, model):
        '''
        Iterate the agent.

        Description:
            If they are inactive then it checks to see if they should become active.
            If they are active then they move or leave the model.
        '''
        if self.status == 0:
            self.activate(model)
        elif self.status == 1:
            self.move(model)
            self.deactivate(model)
        self.history(model)

    def activate(self, model):
        '''
        Test whether an agent should become active.
        This happens when the model time is greater than the agent's activate time.
        '''
        if model.step_id > self.steps_activate:
            self.status = 1
            model.pop_active += 1
            self.step_start = model.step_id

    @staticmethod
    def distance(loc1, loc2):
        '''
        A helpful function to calculate the distance between two points.
        This simply takes the square root of the sum of the square of the elements.
        This appears to be faster than using np.linalg.norm.
        No doubt the numpy implementation would be faster for large arrays.
        Fortunately, all of our norms are of two-element arrays.
        :param arr:     A numpy array (or array-like DS) with length two.
        :return norm:   The norm of the array.
        '''
        x = loc1[0] - loc2[0]
        y = loc1[1] - loc2[1]
        norm = (x*x + y*y)**.5
        return norm

    def move(self, model):
        '''
        Move the agent towards their destination. If the way is clear then the
        agent moves the maximum distance they can given their maximum possible
        speed (self.speed_desire). If not, then they iteratively test smaller
        and smaller distances until they find one that they can travel to
        without causing a colision with another agent.
        '''
        direction = (self.loc_desire - self.location) / self.distance(self.loc_desire, self.location)
        for speed in self.speeds:
            # Direct. Try to move forwards by gradually smaller and smaller amounts
            new_location = self.location + speed * direction
            if self.collision(model, new_location):
                if model.do_history:
                    self.history_collisions += 1
                    model.history_collision_locs.append(new_location)
                    model.history_collision_times.append(model.step_id)

            else:
                break
            # If even the slowest speed results in a colision, then wiggle.
            if speed == self.speeds[-1]:
                new_location = self.location + [0, self.wiggle*np.random.randint(-1, 1+1)]
                if model.do_history:
                    self.history_wiggles += 1
                    model.history_wiggle_locs.append(new_location)
        # Rebound
        if not model.is_within_bounds(new_location):
            new_location = model.re_bound(new_location)
        # Move
        self.location = new_location
        self.speed = speed

    def collision(self, model, new_location):
        '''
        Detects whether a move to the new_location will cause a collision
        (either with the model boundary or another agent).
        '''
        if not model.is_within_bounds(new_location):
            collide = True
        elif self.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, model, new_location):
        '''
        This method finds whether or not nearby neighbours are a collision.

        :param model:        the model that this agent is part of
        :param new_location: the proposed new location that the agent will move to
                             (a standard (x,y) floats-tuple)
        '''
        neighbours = False
        neighbouring_agents = model.tree.query_ball_point(new_location, model.separation)
        for neighbouring_agent in neighbouring_agents:
            agent = model.agents[neighbouring_agent]
            if agent.status == 1 and self.unique_id != agent.unique_id and new_location[0] <= agent.location[0]:
                neighbours = True
                break
        return neighbours

    def deactivate(self, model):
        '''
        Determine whether the agent should leave the model and, if so,
        remove them. Otherwise do nothing.
        '''
        if self.distance(self.location, self.loc_desire) < model.gates_space:
            self.status = 2
            model.pop_active -= 1
            model.pop_finished += 1
            if model.do_history:
                steps_exped = (self.distance(self.loc_start, self.loc_desire) - model.gates_space) / self.speeds[0]
                model.steps_exped.append(steps_exped)
                steps_taken = model.step_id - self.step_start
                model.steps_taken.append(steps_taken)
                steps_delay = steps_taken - steps_exped
                model.steps_delay.append(steps_delay)

    def history(self, model):
        '''
        Save agent location.
        '''
        if model.do_history:
            if self.status == 1:
                self.history_locations.append(self.location)
            else:
                self.history_locations.append((None, None))


class Model:
    '''
    StationSim Model

    Description:
        An Agent-Based Model (ABM) that synchronously `steps`
        step()

    Params:
        unique_id
        **kwargs    # check `params`, and `params_changed`
        do_history  # save memory
        do_print    # mute printing

    Returns:
        step_id
        params
        params_changed

        get_state()
        set_state()

        get_analytics()
        get_trails()
        get_timehist()
        get_location_map()
        get_wiggle_map()
        get_ani()
    '''
    def __init__(self, unique_id=None, **kwargs):
        '''
        Create a new model, reading parameters from a keyword arguement dictionary.
        '''
        self.unique_id = unique_id
        self.status = 1
        # Default Parameters (usually overridden by the caller)
        params = {
            'pop_total': 100,

            'width': 400,
            'height': 200,

            'gates_in': 3,
            'gates_out': 2,
            'gates_space': 1, # Distance around gate that will cause the agent to leave the simulation
            'gates_speed': 1,

            # Agent maximum speed is chosen when the agent is created and drawn from a normal distribution
            'speed_min': .2, # No speeds can be below this
            'speed_mean': 1,
            'speed_std': 1,
            'speed_steps': 3, #

            'separation': 5,
            'max_wiggle': 1,

            'step_limit': 3600,

            'do_history': True,
            'do_print': True,

            'random_seed': int.from_bytes(os.urandom(4), byteorder='little')

        }
        if len(kwargs) == 0:
            warnings.warn(
                "No parameters have been passed to the model; using the default parameters: {}".format(params),
                RuntimeWarning
            )
        self.params, self.params_changed = Model._init_kwargs(params, kwargs)
        [setattr(self, key, value) for key, value in self.params.items()]
        # Set the random seed
        np.random.seed(self.random_seed)
        # Constants
        self.speed_step = (self.speed_mean - self.speed_min) / self.speed_steps
        self.boundaries = np.array([[0, 0], [self.width, self.height]])
        # Following replaced with a normal function
        #gates_init = lambda x, y, n: np.array([np.full(n, x), np.linspace(0, y, n + 2)[1:-1]]).T
        self.gates_locations = np.concatenate([Model._gates_init(0, self.height, self.gates_in), Model._gates_init(self.width, self.height, self.gates_out)])
        # Variables
        self.step_id = 0
        self.pop_active = 0
        self.pop_finished = 0
        # Initialise
        self.agents = [Agent(self, unique_id) for unique_id in range(self.pop_total)]
        # Following replaced with a normal function
        #self.is_within_bounds = lambda loc: all(self.boundaries[0] <= loc) and all(loc <= self.boundaries[1])
        #self.re_bound = lambda loc: np.clip(loc, self.boundaries[0], self.boundaries[1])
        if self.do_history:
            self.history_state = []
            self.history_wiggle_locs = []
            self.history_collision_locs = []
            self.history_collision_times = []
            self.steps_taken = []
            self.steps_exped = []
            self.steps_delay = []
            # Figure Shape Stuff
            self._wid = 8
            self._rel = self._wid / self.width
            self._hei = self._rel * self.height
            self._figsize = (self._wid, self._hei)
            self._dpi = 160

    @staticmethod
    def _gates_init(x, y, n):
        return np.array([np.full(n, x), np.linspace(0, y, n+2)[1:-1]]).T

    def is_within_bounds(self, loc):
        return all(self.boundaries[0] <= loc) and all(loc <= self.boundaries[1])

    def re_bound(self, loc):
        return np.clip(loc, self.boundaries[0], self.boundaries[1])

    @staticmethod
    def _init_kwargs(dict0, dict1):
        '''
        Internal dictionary update tool

        dict0 is updated by dict1 adding no new keys.
        dict2 is the changes excluding 'do_' keys.
        '''
        dict2 = dict()
        for key in dict1.keys():
            if key in dict0:
                if dict0[key] is not dict1[key]:
                    dict0[key] = dict1[key]
                    if 'do_' not in key:
                        dict2[key] = dict1[key]
            else:
                print(f'BadKeyWarning: {key} is not a model parameter.')
        return dict0, dict2

    def step(self):
        '''
        Iterate model forward one step.
        '''
        if self.pop_finished < self.pop_total and self.step_id < self.step_limit and self.status==1:
            if self.do_print and self.step_id%100==0:
                print(f'\tIteration: {self.step_id}/{self.step_limit}')
            state = self.get_state('location2D')
            self.tree = cKDTree(state)
            [agent.step(self) for agent in self.agents]
            if self.do_history:
                self.history_state.append(state)
            self.step_id += 1
        else:
            if self.do_print and self.status==1:
                print(f'StationSim {self.unique_id} - Everyone made it!')
                self.status = 0

    # State
    def get_state(self, sensor=None):
        '''
        Convert list of agents in model to state vector.
        '''
        if sensor is None:
            state = [(agent.status, *agent.location, agent.speed) for agent in self.agents]
            state = np.append(self.step_id, np.ravel(state))
        elif sensor is 'location':
            state = [agent.location for agent in self.agents]
            state = np.ravel(state)
        elif sensor is 'location2D':
            state = [agent.location for agent in self.agents]
        return state

    def set_state(self, state, sensor=None):
        '''
        Use state vector to set agent locations.
        '''
        if sensor is None:
            self.step_id = int(state[0])
            state = np.reshape(state[1:], (self.pop_total, 3))
            for i, agent in enumerate(self.agents):
                agent.status = int(state[i, 0])
                agent.location = state[i, 1:]
        elif sensor is 'location':
            state = np.reshape(state, (self.pop_total, 2))
            for i, agent in enumerate(self.agents):
                agent.location = state[i, :]
        elif sensor is 'location2D':
            for i, agent in enumerate(self.agents):
                agent.location = state[i, :]

    # TODO: Deprecated, update PF
    def agents2state(self, do_ravel=True):
        warnings.warn("Replace 'state = agents2state()' with 'state = get_state(sensor='location')'", DeprecationWarning)
        return self.get_state(sensor='location')
    def state2agents(self, state):
        warnings.warn("Replace 'state2agents(state)' with 'set_state(state, sensor='location')'", DeprecationWarning)
        return self.set_state(state, sensor='location')

    # Analytics
    def get_analytics(self, sig_fig=None):
        '''
        A collection of analytics.
        '''
        analytics = {
            'Finish Time': self.step_id,
            'Total': self.pop_total,
            'Active': self.pop_active,
            'Finished': self.pop_finished,
            'Mean Time Taken': np.mean(self.steps_taken),
            'Mean Time Expected': np.mean(self.steps_exped),
            'Mean Time Delay': np.mean(self.steps_delay),
            'Mean Collisions': np.mean([agent.history_collisions for agent in self.agents]),
            'Mean Wiggles': np.mean([agent.history_wiggles for agent in self.agents]),
            # 'GateWiggles': sum(wig[0]<self.gates_space for wig in self.history_wiggle_locs)/self.pop_total
            }
        return analytics

    def get_trails(self, plot_axis=False, plot_legend=True, colours=('b','g','r'), xlim=None, ylim=None):
        """
        Make a figure showing the trails of the agents.

        :param plot_axis: Whether to show the axis (default False)
        :param plot_legend: Whether to show the legend (default False)
        :param colours: Optional tuple with three values representing the colours of agents in states
        1 (no started), 2 (active), 3 (finished). Default: ('b','g','r')
        :param xlim Optional x axis limits (usually a tuple of (xmin,xmax)).
        :param ylim Optional y axis limits (usually a tuple of (ymin,ymax)).
        :return: The matplotlib Figure object.
        """
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        plt.axis(np.ravel(self.boundaries, 'f'))
        if not plot_axis:
            plt.axis('off')
        else:
            plt.ylabel("Y position")
            plt.xlabel("X position")
        plt.plot([], 'b')
        plt.plot([], 'g')
        plt.title('Agent Trails')
        if plot_legend:
            plt.legend(['Active', 'Finished'])
        plt.tight_layout(pad=0)
        for agent in self.agents:
            if agent.status == 1:
                alpha = 1
                colour = colours[0]
            elif agent.status == 2:
                alpha = .5
                colour = colours[1]
            else:
                alpha = 1
                colour = colours[2]
            locs = np.array(agent.history_locations).T
            plt.plot(*locs, color=colour, alpha=alpha, linewidth=.5)
        if xlim != None: # Optionally set the x limits
            plt.xlim(xlim)
        if ylim != None:  # Optionally set the x limits
            plt.xlim(ylim)
        return fig

    def get_histogram(self):
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        fmax = max(np.amax(self.steps_exped), np.amax(self.steps_taken), np.amax(self.steps_delay))
        sround = lambda x, p: float(f'%.{p-1}e'%x)
        bins = np.linspace(0, sround(fmax, 2), 20)
        plt.hist(self.steps_exped, bins=bins+4, alpha=.5, label='Expected')
        plt.hist(self.steps_taken, bins=bins+2, alpha=.5, label='Taken')
        plt.hist(self.steps_delay, bins=bins+0, alpha=.5, label='Delayed')
        plt.xlabel('Time')
        plt.ylabel('Number of Agents')
        plt.grid(False)
        plt.legend()
        plt.tight_layout(pad=0)
        return fig

    @staticmethod
    def _heightmap(data, ax=None, kdeplot=True, cmap=None, alpha=.7, cbar=False):
        if kdeplot:
            from seaborn import kdeplot as sns_kdeplot
            sns_kdeplot(*data, ax=ax, cmap=cmap, alpha=alpha, shade=True, shade_lowest=False, cbar=cbar)
        else:
            hdata, binx, biny = np.histogram2d(*data, (20, 10))
            ax.contourf(hdata.T, cmap=cmap, alpha=alpha, extend='min', extent=(binx[0],binx[-1],biny[0],biny[-1]))
        return ax

    def get_wiggle_map(self, do_kdeplot=True, title="Collision Map"):
        """ Show where wiggles and collisions took place

        :param do_kdeplot:
        :param title: (optional) title for the graph
        :return: The figure object
        """
        fig, ax = plt.subplots(1, figsize=self._figsize, dpi=self._dpi)
        fig.tight_layout(pad=0)
        self._heightmap(np.array(self.history_collision_locs).T, ax=ax, kdeplot=do_kdeplot)
        self._heightmap(np.array(self.history_wiggle_locs).T, ax=ax)
        ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:,0], xticks=[],
               ylim=self.boundaries[:,1], yticks=[], title=title)
        return fig

    def get_collision_map(self, *args, **kwargs):
        """For making a map of collisions and wiggles. Just calls get_wiggle_map()"""
        self.get_wiggle_map(*args, **kwargs)

    def get_location_map(self, do_kdeplot=True, title="Location Map", color_bar = False, plot_axis=False):
        """
        Create a density plot of the agents' locations

        :param do_kdeplot:
        :param title: (optional) title for the plot
        :return:
        """
        history_locs = []
        for agent in self.agents:
            for loc in agent.history_locations:
                if None not in loc:
                    history_locs.append(loc)
        history_locs = np.array(history_locs).T
        fig, ax = plt.subplots(1, figsize=self._figsize, dpi=self._dpi)
        fig.tight_layout(pad=0)
        self._heightmap(data=history_locs, ax=ax, kdeplot=do_kdeplot, cmap='gray_r', cbar=color_bar)
        ax.set(frame_on=plot_axis, aspect='equal', xlim=self.boundaries[:,0], xticks=[],
               ylim=self.boundaries[:,1], yticks=[], title=title)
        if plot_axis:
            ax.set_ylabel("Y position")
            ax.set_xlabel("X position")
        return fig

    def get_ani(self, agents=None, colour='k', alpha=.5, show_separation=False, wiggle_map=False):
        # Load Data
        locs = np.array([agent.history_locations for agent in self.agents[:agents]]).transpose((1,2,0))
        markersize = self.separation * 216*self._rel  # 3*72px/in=216
        #
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        if wiggle_map:
            sns.kdeplot(*np.array(self.collision_map).T, ax=ax, cmap='gray_r', alpha=.3, shade=True, shade_lowest=False)
        ln0, = plt.plot([], [], '.', alpha=.05, color=colour, markersize=markersize)
        ln1, = plt.plot([], [], '.', alpha=alpha, color=colour)
        def init():
            fig.tight_layout(pad=0)
            ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:,0], xticks=[], ylim=self.boundaries[:,1], yticks=[])
            return ln0, ln1,
        def func(frame):
            if show_separation:
                ln0.set_data(*locs[frame])
            ln1.set_data(*locs[frame])
            return ln0, ln1,
        frames = self.step_id
        ani = FuncAnimation(fig, func, frames, init, interval=100, blit=True)
        return ani

    @classmethod
    def set_random_seed(cls, seed=None):
        """Set a new numpy random seed
        :param seed: the optional seed value (if None then get one from os.urandom)
        """
        new_seed = int.from_bytes(os.urandom(4), byteorder='little') if seed == None else seed
        np.random.seed(new_seed)

if __name__ == '__main__':
    warnings.warn("The stationsim_model.py code should not be run directly. Create a separate script and use that "
                  "to run experimets (e.g. see ABM_DA/experiments/StationSim basic experiment.ipynb )")
    print("Nothing to do")
