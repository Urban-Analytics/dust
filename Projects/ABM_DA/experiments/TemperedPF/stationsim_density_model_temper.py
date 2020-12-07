
'''
StationSim - Density version
    author: patricia-ternes
    created: 20/08/2020
    
    edited by: Vijay Kumar
    edited 22/09/2020
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
    warnings.warn("The seaborn module is not available. If you try to create "
                  "kde plots for this model (i.e. a wiggle map or density "
                  "map) then it will fail.")


class Agent:
    '''
    A class representing a generic agent for the StationSim ABM.
    '''
    def __init__(self, model, unique_id):
        '''
        Initialise a new agent.

        Desctiption:
            Creates a new agent and gives it a randomly chosen exit,
            and desired speed.
            All agents start with active state 0 ('not started').
            Their initial location (** (x,y) tuple-floats **) is set to
            (0,0) and changed when the agent is activated.

        Parameters:
            model - a pointer to the StationSim model that is creating
            this agent
        '''
        # Required
        self.model = model
        self.unique_id = unique_id
        self.status = 0  # 0 Not Started, 1 Active, 2 Finished
        self.location = np.array([0, 0])  # replaced when the agent is activated
        self.size = model.agent_size
        self.local_density_radius = self.size * 10.0
        self.gate_in = np.random.randint(model.gates_in)
        self.set_gate_out()
        self.loc_desire = self.set_agent_location(self.gate_out)
        
        # Speed
        speed_max = 0
        while speed_max <= model.speed_min:
            speed_max = np.random.normal(model.speed_mean, model.speed_std)
        speeds = np.arange(speed_max, model.speed_min, - model.speed_step)
        self.speed = np.random.choice((speeds))

        # Others
        '''
          This function was chosen to agree with the 
          Grand Central Terminal data.
          model.birth_rate 
        '''
        self.steps_activate = self.unique_id * 25.0 / model.birth_rate

        # History
        if model.do_history:
            self.history_locations = []
            self.history_speeds = []
            self.history_wiggles = 0  # it is not used.
            self.history_collisions = 0  # it is not used.
            self.step_start = None
            self.history_density = []
            self.history_angle = []
            self.history_locations = []  # necessary in Particle Filter
            self.history_locations_var = []  # necessary in Particle Filter
        else:
            self.history_locations = []  # necessary in Particle Filter
            self.history_locations_var = []  # necessary in Particle Filter
            self.step_start = None

    def set_gate_out(self):
        '''
        Set a exit gate for the agent.
        - The exit gate ca be any gate that is on a different side of
        the entrance gate.
        '''
        if (self.model.station == 'Grand_Central'):
            if (self.gate_in == 0):
                self.gate_out = np.random.random_integers(1, 10)

            elif (self.gate_in == 1 or self.gate_in == 2):
                self.gate_out = np.random.choice( (0, 3, 4, 5, 6, 7, 8, 9, 10))
            elif (self.gate_in == 3 or self.gate_in == 4 or self.gate_in == 5 or self.gate_in == 6):
                self.gate_out = np.random.choice( (0, 1, 2, 7, 8, 9, 10))
            else:
                self.gate_out = np.random.random_integers(0, 6)
        else:
            self.gate_out = np.random.randint(self.model.gates_out) + self.model.gates_in

    def step(self):
        '''
        Iterate the agent.

        Description:
            If they are active then they move and maybe leave the model.
        '''
        if self.status == 1:
            self.move()
            self.deactivate()
            
            
    def monte_step(self):
        '''
        Step for tempering 


        '''
        if self.status == 1:
            self.monte_move()
            self.deactivate()


    def activate(self):
        '''
        Test whether an agent should become active.
        This happens when the model time is greater than the agent's
        activate time.
        '''

        if self.status == 0:
            if self.model.step_id >= self.steps_activate:
                new_location = self.set_agent_location(self.gate_in)
                self.location = new_location
                self.status = 1
                self.model.pop_active += 1
                self.step_start = self.model.step_id
                self.loc_start = self.location

    def set_agent_location(self, gate):
        '''
            Define one final or initial position for the agent.

            It is necessary to ensure that the agent has a distance from
            the station wall compatible with its own size.
        '''

        wd = self.model.gates_width[gate] / 2.0
        perturb = np.random.uniform(-wd, +wd)
        if(self.model.gates_locations[gate][0] == 0):
            new_location = self.model.gates_locations[gate] + [1.05*self.size, perturb]
        elif(self.model.gates_locations[gate][0] == self.model.width):
            new_location = self.model.gates_locations[gate] + [-1.05*self.size, perturb]
        elif(self.model.gates_locations[gate][1] == 0):
            new_location = self.model.gates_locations[gate] + [perturb, 1.05*self.size]
        else:
            new_location = self.model.gates_locations[gate] + [perturb, -1.05*self.size]
        
        '''
            As there are gates near the corners it is possible to create 
            a position outside the station. To fix this, rebound:
        '''
        if not self.model.is_within_bounds(self, new_location):
            new_location = self.model.re_bound(self, new_location)

        return new_location

    @staticmethod
    def distance(loc1, loc2):
        '''
        A helpful function to calculate the distance between two points.
        This simply takes the square root of the sum of the square of the
        elements. This appears to be faster than using np.linalg.norm.
        No doubt the numpy implementation would be faster for large
        arrays. Fortunately, all of our norms are of two-element arrays.
        :param arr:     A numpy array (or array-like DS) with length two.
        :return norm:   The norm of the array.
        '''
        x = loc1[0] - loc2[0]
        y = loc1[1] - loc2[1]
        norm = (x*x + y*y)**.5
        return norm

    def get_direction(self, loc_desire, location):
        '''
         A helpful function to determine a unitary vector that gives
         the direction between two locations. 
        '''
        if(self.distance(loc_desire, location)==0.0):
            dir_vector = np.array([0, 0])
        else:
            dir_vector =  (loc_desire - location) / self.distance(loc_desire, location)
        
        return dir_vector

    #@staticmethod
    def set_direction(self, vector, std):
        '''
         A helpful function to rotate a vector randomly, based
         in a standard deviation value (std).
        '''
        dir_vector = np.array([0.0,0.0])
        angle = np.random.normal(0, std*180.0)
        if self.model.do_history:
            self.history_angle.append(angle)
        angle = np.radians(angle)
        dir_vector[0] = vector[0]*np.cos(angle) - vector[1]*np.sin(angle)
        dir_vector[1] = vector[0]*np.sin(angle) + vector[1]*np.cos(angle)

        return dir_vector


    def move(self):
        '''
         Move the agent towards their destination.
         The speed and direction of movement can
         change depends on the local_density value.

         - self.local_density: 0.0 to 1.0 value.
        '''

        self.get_local_density()
        if self.model.do_history:
            self.history_density.append(self.local_density)
        direction = self.get_direction(self.loc_desire, self.location)
        new_direction = self.set_direction(direction, self.local_density)

        velocity = self.speed * (1.0 - self.local_density)
        
        self.location = self.location + velocity * new_direction # velocity * new_direction * time_step
        self.location = self.model.re_bound(self, self.location)

    def monte_move(self):
        '''
         Random Monte Carlo step for tempering. Steps 5 meters in a random direction.

         - self.local_density: 0.0 to 1.0 value.
         
        '''

        
        #for set distance of 2.5m (35 pixels)
        rand_dir= np.array((float(np.random.uniform(-1,1,1)),float(np.random.uniform(-1,1,1))))
        f=rand_dir[0]/rand_dir[1]
        z=35/((f**2)+1)**0.5
        dist_vect=np.array([np.random.choice([1,-1])*z,np.random.choice([-1,1])*(z*np.abs(f))])

        self.location = self.location + dist_vect


        self.location = self.model.re_bound(self, self.location)



    def get_local_density(self):
        '''
         A function to determine the normalized
         local geographic density around each agent. 
        '''

        state = self.model.get_state('location2D')
        self.model.tree = cKDTree(state)
        neighbouring_agents = self.model.tree.query_ball_point(self.location, self.local_density_radius)
        self.local_density = len(neighbouring_agents) * self.size**2 / (self.local_density_radius**2 - self.size**2) # density between 0-1


        #can determine the density in other ways?
        #self.local_density = 0.5
        #self.local_density = np.random.uniform()
        #self.local_density = np.random.exponential(0.034755918) # (1.0 / 28.772078751013954)

    def deactivate(self):
        '''
        Determine whether the agent should leave the model and, if so,
        remove them. Otherwise do nothing.
        '''

        if self.distance(self.location, self.loc_desire) < self.model.gates_space:
            self.status = 2
            self.model.pop_active -= 1
            self.model.pop_finished += 1
            if self.model.do_history:
                steps_exped = (self.distance(self.loc_start, self.loc_desire) -
                               self.model.gates_space) / self.speed
                self.model.steps_exped.append(steps_exped)
                steps_taken = self.model.step_id - self.step_start
                self.model.steps_taken.append(steps_taken)
                steps_delay = steps_taken - steps_exped
                self.model.steps_delay.append(steps_delay)

    def history(self):
        '''
        Save agent location.
        '''
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
        Create a new model, reading parameters from a keyword arguement
        dictionary.
        '''
        self.unique_id = unique_id
        self.status = 1
        # Default Parameters (usually overridden by the caller)
        params = {
            'pop_total': 100,
            'agent_size': 1.0,  # new parameter
            'birth_rate': 1.0,  # new parameter

            'width': 400,
            'height': 200,
            'gates_in': 3,
            'gates_out': 2,
            'gates_space': 1.0,

            'speed_min': .2,
            'speed_mean': 1,
            'speed_std': 1,
            'speed_steps': .2,

            'separation': 5,  # just used in animation

            'step_limit': 10000,

            'do_history': True,
            'do_print': True,

            'random_seed': int.from_bytes(os.urandom(4), byteorder='little'),

            'tolerance': 0.1,  # new parameter
            'station': None  # None or Grand_Central  # new parameter

        }

        if len(kwargs) == 0:
            warnings.warn(
                "No parameters have been passed to the model; using the "
                "default parameters: {}".format(params),
                RuntimeWarning
            )
        self.params, self.params_changed = Model._init_kwargs(params, kwargs)
        [setattr(self, key, value) for key, value in self.params.items()]
        # Set the random seed
        np.random.seed(self.random_seed)
        self.speed_step = (self.speed_mean - self.speed_min) / self.speed_steps

        # Variables
        self.step_id = 0
        self.pop_active = 0
        self.pop_finished = 0

        # Initialise station
        self.set_station()

        # Initialise agents
        self.agents = [Agent(self, unique_id) for unique_id in
                       range(self.pop_total)]

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

    def set_station(self):
        '''
        Allows to manually set a station (e.g. 'Grand_Central') rather
        than automatically generating a station from parameters like
        number of gates, gate size, etc.
        '''
        if(self.station == 'Grand_Central'):
            self.width = 740  # 53 m
            self.height = 700  # 50 m
            self.boundaries = np.array([[0, 0], [self.width, self.height]])
            self.gates_locations =\
                np.array([[0, 275],  # gate 0
                          [125, 700],   # gate 1
                          [577.5 , 700],  # gate 2
                          [740, 655],  # gate 3
                          [740, 475],  # gate 4
                          [740, 265],   # gate 5
                          [740, 65],   # gate 6
                          [647.5, 0],   # gate 7
                          [462.5, 0],   # gate 8
                          [277.5, 0],   # gate 9
                          [92.5, 0]])   # gate 10

            self.gates_width = [250, 250, 245, 90, 150, 150, 120, 185, 185, 185, 185]

            self.gates_in = len(self.gates_locations)
            self.gates_out = len(self.gates_locations)
            self.agent_size = 7.0  # 0.5 m
            self.speed_mean = 0.839236  # pixel / frame
            self.speed_std = 0.349087  # pixel / frame
            self.speed_min = 0.2 # 0.1  # pixel / frame
            self.gates_space = 28.0  # 2 m
        else:
            self.gates_locations = np.concatenate([
                Model._gates_init(0, self.height, self.gates_in),
                Model._gates_init(self.width, self.height, self.gates_out)])
            self.gates_width = [20 for _ in range (len(self.gates_locations))]
            self.boundaries = np.array([[0, 0], [self.width, self.height]])

            if(self.station is not None):
                warnings.warn(
                    "The station parameter passed to the model is not valid; "
                    "Using the default station.",
                    RuntimeWarning
                )

    def is_within_bounds(self, agent, loc):
        return all((self.boundaries[0] + agent.size*2.0) < loc) and\
               all(loc < (self.boundaries[1] - agent.size*2.0))

    def re_bound(self, agent, loc):
        return np.clip(loc, self.boundaries[0] + agent.size*1.1,
                       self.boundaries[1] - agent.size*1.1)

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
        Iterate model forward one second.
        '''
        if self.step_id == 0:
            state = self.get_state('location2D')


        if self.pop_finished < self.pop_total and\
                self.step_id < self.step_limit and self.status == 1:
            if self.do_print and self.step_id % 100 == 0:
                print(f'\tIteration: {self.step_id}/{self.step_limit}')

            [agent.activate() for agent in self.agents]

            [agent.step() for agent in self.agents]

            if self.do_history:
                state = self.get_state('location2D')
                self.history_state.append(state)
                [agent.history() for agent in self.agents]

            self.step_id += 1
        else:
            if self.do_print and self.status == 1:
                print(f'StationSim {self.unique_id} - Everyone made it!')
                self.status = 0
                
                
    def step_mc(self):
        '''
        Begin Monte Carlo step.
        '''


        [agent.activate() for agent in self.agents]

        [agent.monte_step() for agent in self.agents]




    # State
    def get_state(self, sensor=None):
        '''
        Convert list of agents in model to state vector.
        '''
        if sensor is None:
            state = [(agent.status, *agent.location, agent.speed) for agent in
                     self.agents]
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
        warnings.warn("Replace 'state = agents2state()' with 'state = "
                      "get_state(sensor='location')'", DeprecationWarning)
        return self.get_state(sensor='location')

    def state2agents(self, state):
        warnings.warn("Replace 'state2agents(state)' with 'set_state(state, "
                      "sensor='location')'", DeprecationWarning)
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
            'Mean Collisions': np.mean([agent.history_collisions for agent in
                                        self.agents]),
            'Mean Wiggles': np.mean([agent.history_wiggles for agent in
                                     self.agents])
            }
        return analytics

    def get_trails(self, plot_axis=False, plot_legend=True, colours=('b', 'g',
                   'r'), xlim=None, ylim=None):
        '''
        Make a figure showing the trails of the agents.

        :param plot_axis: Whether to show the axis (default False)
        :param plot_legend: Whether to show the legend (default False)
        :param colours: Optional tuple with three values representing
        the colours of agents in states
        1 (no started), 2 (active), 3 (finished). Default: ('b','g','r')
        :param xlim Optional x axis limits (usually a tuple of (xmin,xmax)).
        :param ylim Optional y axis limits (usually a tuple of (ymin,ymax)).
        :return: The matplotlib Figure object.
        '''
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
        if xlim is not None:  # Optionally set the x limits
            plt.xlim(xlim)
        if ylim is not None:  # Optionally set the x limits
            plt.xlim(ylim)
        return fig

    def get_histogram(self):
        fig = plt.figure(figsize=self._figsize, dpi=self._dpi)
        fmax = max(np.amax(self.steps_exped), np.amax(self.steps_taken),
                   np.amax(self.steps_delay))
        sround = lambda x, p: float(f'%.{p-1}e' % x)
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
    def _heightmap(data, ax=None, kdeplot=True, cmap=None, alpha=.7,
                   cbar=False):
        if kdeplot:
            from seaborn import kdeplot as sns_kdeplot
            sns_kdeplot(*data, ax=ax, cmap=cmap, alpha=alpha, shade=True,
                        shade_lowest=False, cbar=cbar)
        else:
            hdata, binx, biny = np.histogram2d(*data, (20, 10))
            ax.contourf(hdata.T, cmap=cmap, alpha=alpha, extend='min',
                        extent=(binx[0], binx[-1], biny[0], biny[-1]))
        return ax

    def get_wiggle_map(self, do_kdeplot=True, title="Collision Map"):
        """ Show where wiggles and collisions took place

        :param do_kdeplot:
        :param title: (optional) title for the graph
        :return: The figure object
        """
        fig, ax = plt.subplots(1, figsize=self._figsize, dpi=self._dpi)
        fig.tight_layout(pad=0)
        self._heightmap(np.array(self.history_collision_locs).T, ax=ax,
                        kdeplot=do_kdeplot)
        self._heightmap(np.array(self.history_wiggle_locs).T, ax=ax)
        ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:, 0],
               xticks=[], ylim=self.boundaries[:, 1], yticks=[], title=title)
        return fig

    def get_collision_map(self, *args, **kwargs):
        '''For making a map of collisions and wiggles.
        Just calls get_wiggle_map()'''
        self.get_wiggle_map(*args, **kwargs)

    def get_location_map(self, do_kdeplot=True, title="Location Map",
                         color_bar=False, plot_axis=False):
        '''
        Create a density plot of the agents' locations

        :param do_kdeplot:
        :param title: (optional) title for the plot
        :return:
        '''
        history_locs = []
        for agent in self.agents:
            for loc in agent.history_locations:
                if None not in loc:
                    history_locs.append(loc)
        history_locs = np.array(history_locs).T
        fig, ax = plt.subplots(1, figsize=self._figsize, dpi=self._dpi)
        fig.tight_layout(pad=0)
        self._heightmap(data=history_locs, ax=ax, kdeplot=do_kdeplot,
                        cmap='gray_r', cbar=color_bar)
        ax.set(frame_on=plot_axis, aspect='equal', xlim=self.boundaries[:, 0],
               xticks=[], ylim=self.boundaries[:, 1], yticks=[], title=title)
        if plot_axis:
            ax.set_ylabel("Y position")
            ax.set_xlabel("X position")
        return fig

    def get_ani(self, agents=None, colour='k', alpha=.5, show_separation=False,
                wiggle_map=False):
        # Load Data
        locs = np.array([agent.history_locations for agent in
                        self.agents[:agents]]).transpose((1, 2, 0))
        markersize1 = self.separation * 216*self._rel  # 3*72px/in=216
        markersize2 = 216*self._rel
        #
        fig, ax = plt.subplots(figsize=self._figsize, dpi=self._dpi)
        if wiggle_map:
            sns.kdeplot(*np.array(self.collision_map).T, ax=ax, cmap='gray_r',
                        alpha=.3, shade=True, shade_lowest=False)
        ln0, = plt.plot([], [], '.', alpha=.05, color=colour,
                        markersize=markersize1)
        ln1, = plt.plot([], [], '.', alpha=alpha, color=colour,
                        markersize=markersize2)

        def init():
            fig.tight_layout(pad=0)
            ax.set(frame_on=False, aspect='equal', xlim=self.boundaries[:, 0],
                   xticks=[], ylim=self.boundaries[:, 1], yticks=[])
            return ln0, ln1,

        def func(frame):
            if show_separation:
                ln0.set_data(*locs[frame])
            ln1.set_data(*locs[frame])
            return ln0, ln1,
        frames = self.step_id
        ani = FuncAnimation(fig, func, frames, init, interval=100, blit=True)
        return ani

    def get_distace_plot(self, real_data_dir, frame_i, frame_f, dt):
        self.graphX1 = []; self.graphY1 = []; self.graphERR1 = [] # x, y, dy
        data = []
        for frame in range(frame_i, frame_f, dt):
            ID, x, y = np.loadtxt(real_data_dir + str(frame) + '.0.dat', unpack=True)
            dist = []
            for i in range(len(ID)):
                agent_ID = int(ID[i])
                r1 = self.agents[agent_ID].history_locations[int(frame/dt)]
                r2 = (x[i], y[i])
                if np.all(r1 != (None, None)):
                    distance = self.agents[agent_ID].distance(r1, r2)
                    dist.append(distance)
                    time = int(frame - self.agents[agent_ID].step_start)
                    data.append([time, distance])
            dist = np.asarray(dist)
            self.graphX1.append(frame); self.graphY1.append(dist.mean()); self.graphERR1.append(dist.std())

        from operator import itemgetter
        #sort by frame
        data1 = sorted(data, key=itemgetter(0))

        frame = data1[0][0]
        self.graphX2 = []; self.graphY2 = []; self.graphERR2 = [] # x, y, dy
        dist = []
        for line in data1:
            if (line[0]==frame):
                dist.append(line[1])
            else:
                dist = np.asarray(dist)
                self.graphX2.append(frame); self.graphY2.append(dist.mean()); self.graphERR2.append(dist.std())
                frame = line[0]
                dist = []
                dist.append(line[1])
        dist = np.asarray(dist)
        self.graphX2.append(frame); self.graphY2.append(dist.mean()); self.graphERR2.append(dist.std())

    @classmethod
    def set_random_seed(cls, seed=None):
        '''Set a new numpy random seed
        :param seed: the optional seed value (if None then
        get one from os.urandom)
        '''
        new_seed = int.from_bytes(os.urandom(4), byteorder='little')\
            if seed is None else seed
        np.random.seed(new_seed)

if __name__ == '__main__':
    warnings.warn("The stationsim_gcs_model.py code should not be run directly"
                  ". Create a separate script and use that to run experimets "
                  "(e.g. see ABM_DA/experiments/StationSim basic experiment."
                  "ipynb )")
    print("Nothing to do")