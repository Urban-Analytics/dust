'''
StationSim - GrandCentral version
    author: patricia-ternes
    created: 30/04/2020
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
        self.location = [0, 0]  # replaced when the agent is activated
        self.size = model.agent_size

        self.gate_in = np.random.randint(model.gates_in)
        self.set_gate_out()
        self.loc_desire = self.set_agent_location(self.gate_out)
        
        # Speed
        speed_max = 0
        while speed_max <= model.speed_min:
            speed_max = np.random.normal(model.speed_mean, model.speed_std)
        self.speeds = np.arange(speed_max, model.speed_min, - model.speed_step)
        self.speed = np.random.choice((self.speeds))
        # Others
        self.steps_activate = np.random.exponential(model.gates_speed)

        # History
        if model.do_history:
            self.history_locations = []
            self.history_speeds = []
            self.history_wiggles = 0
            self.history_collisions = 0
            self.step_start = None
    
    def set_gate_out(self):
        '''
        Set a exit gate for the agent.
        - The exit gate ca be any gate that is on a different side of
        the entrance gate.
        '''
        if (self.model.station == 'Grand_Central'):
            if (self.gate_in == 0):
                self.gate_out = np.random.choice( (1, 2, 3, 4, 5, 6, 7, 8, 9))
            elif (self.gate_in == 1 or self.gate_in == 2):
                self.gate_out = np.random.choice( (0, 3, 4, 5, 6, 7, 8, 9))
            elif (self.gate_in == 3 or self.gate_in == 4):
                self.gate_out = np.random.choice( (0, 1, 2, 5, 6, 7, 8, 9))
            else:
                self.gate_out = np.random.choice( (0, 1, 2, 3, 4))

            '''
            Use this while statement if exit gates at same side of 
            entrance gate are allowed. With the above if statement, this
            while statement never will be access, so if needed, comment
            the above if statement.
            '''
            '''
            while (self.gate_out == self.gate_in or
                   self.gate_out >= len(self.model.gates_locations)):
                self.gate_out = np.random.randint(self.model.gates_out)
            '''
        else:
        	self.gate_out = np.random.randint(self.model.gates_out) + self.model.gates_in

    def step(self, time):
        '''
        Iterate the agent.

        Description:
            If they are active then they move and maybe leave the model.
        '''
        if self.status == 1:
            self.move(time)
            self.deactivate()

    def activate(self):
        '''
        Test whether an agent should become active.
        This happens when the model time is greater than the agent's
        activate time.

        It is necessary to ensure that the agent has an initial position
        different from the position of all active agents. If it was not
        possible, activate the agent on next time step.
        '''
        if self.status == 0:
            if self.model.total_time > self.steps_activate:
                state = self.model.get_state('location2D')
                self.model.tree = cKDTree(state)
                for _ in range(10):
                    new_location = self.set_agent_location(self.gate_in)
                    neighbouring_agents = self.model.tree.query_ball_point(
                        new_location, self.size*1.1)
                    if (neighbouring_agents == [] or
                            neighbouring_agents == [self.unique_id]):
                        self.location = new_location
                        self.status = 1
                        self.model.pop_active += 1
                        self.step_start = self.model.total_time  # self.model.step_id
                        self.loc_start = self.location
                        break

    def set_agent_location(self, gate):
        '''
            Define one final or initial position for the agent.

            It is necessary to ensure that the agent has a distance from
            the station wall compatible with its own size.
        '''
        perturb = self.model.gates_space * np.random.uniform(-10, +10)
        if(self.model.gates_locations[gate][0] == 0):
            return self.model.gates_locations[gate] + [1.05*self.size, perturb]
        elif(self.model.gates_locations[gate][0] == self.model.width):
            return self.model.gates_locations[gate] + [-1.05*self.size, perturb]
        elif(self.model.gates_locations[gate][1] == 0):
            return self.model.gates_locations[gate] + [perturb, 1.05*self.size]
        else:
            return self.model.gates_locations[gate] + [perturb, -1.05*self.size]

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
        return (loc_desire - location) / self.distance(loc_desire, location)

    @staticmethod
    def get_normal_direction(direction):
        '''
        Rotate a two-dimensional array by 90 degrees in clockwise or
        counter clockwise direction (np.random.choice((-1, 1))).
        '''
        return np.array([direction[1], direction[0] *
                         np.random.choice((-1, 1))])

    def move(self, time_step):
        '''
        Move the agent towards their destination. The agent moves the
        maximum distance they can given their maximum possible speed
        (self.speed_desire) and the time_step.
        '''
        direction = self.get_direction(self.loc_desire, self.location)
        self.location = self.location + self.speed * direction * time_step

    def set_wiggle(self):
        '''
        Determine a new position for an agent that collided with another
        agent, or with some element of the station.
        The new position simulates a lateral step. The side on which the
        agent will take the step is chosen at random, as well as the
        amplitude of the step.

        Description:
        - Determine a new position and check if it is a unique position.
        - If it is unique, then the agent receives this position.
        - Otherwise, a new position will be determined.
        - This process has a limit of 10 attempts. If it is not possible
        to determine a new unique position, the agent just stay stopped.
        '''
        direction = self.get_direction(self.loc_desire, self.location)

        state = self.model.get_state('location2D')
        self.model.tree = cKDTree(state)
        for _ in range(10):
            normal_direction = self.get_normal_direction(direction)
            new_location = self.location +\
                normal_direction *\
                np.random.normal(self.size, self.size/2.0)

            # Rebound
            if not self.model.is_within_bounds(self, new_location):
                new_location = self.model.re_bound(self, new_location)

            # collision_map
            if self.model.do_history:
                self.history_collisions += 1
                self.model.history_collision_locs.append(new_location)
                self.model.history_collision_times.append(self.model.total_time)

            # Check if the new location is possible
            neighbouring_agents = self.model.tree.query_ball_point(new_location,
                                                              self.size*1.1)
            dist = self.distance(new_location, self.model.clock.location)
            if ((neighbouring_agents == [] or
                    neighbouring_agents == [self.unique_id]) and
                    (dist > (self.size + self.model.clock.size))):
                self.location = new_location

                # wiggle_map
                if self.model.do_history:
                    self.history_wiggles += 1
                    self.model.history_wiggle_locs.append(new_location)
                break

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
                               self.model.gates_space) / self.speeds[0]
                self.model.steps_exped.append(steps_exped)
                steps_taken = self.model.total_time - self.step_start
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

    def get_collisionTime2Agents(self, agentB):
        '''
        Returns the collision time between two agents.
        '''
        tmin = 1.0e300

        rAB = self.location - agentB.location
        directionA = self.get_direction(self.loc_desire, self.location)
        directionB = agentB.get_direction(agentB.loc_desire, agentB.location)
        sizeAB = self.size + agentB.size

        vA = self.speed
        vB = agentB.speed
        vAB = vA*directionA - vB*directionB
        bAB = np.dot(vAB, rAB)
        if bAB < 0.0:
            delta = bAB**2 - (np.dot(vAB, vAB)*(np.dot(rAB, rAB) - sizeAB**2))
            if (delta > 0.0):
                collisionTime = abs((-bAB - np.sqrt(delta)) / np.dot(vAB, vAB))
                tmin = collisionTime

        return tmin

    def get_collisionTimeWall(self):
        '''
        Returns the shortest collision time between an agent and a wall.
        '''
        tmin = 1.0e300
        collisionTime = 1.0e300

        direction = self.get_direction(self.loc_desire, self.location)
        vx = self.speed*direction[0]  # horizontal velocity
        vy = self.speed*direction[1]  # vertical velocity

        if(vy > 0):  # collision in botton wall
            collisionTime = (self.model.height - self.size - self.location[1]) / vy
        elif (vy < 0):  # collision in top wall
            collisionTime = (self.size - self.location[1]) / vy
        if (collisionTime < tmin):
            tmin = collisionTime
        if(vx > 0):  # collision in right wall
            collisionTime = (self.model.width - self.size - self.location[0]) / vx
        elif (vx < 0):  # collision in left wall
            collisionTime = (self.size - self.location[0]) / vx
        if (collisionTime < tmin):
            tmin = collisionTime

        return tmin


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

            'width': 400,
            'height': 200,

            'gates_in': 3,
            'gates_out': 2,
            'gates_space': 1,
            'gates_speed': 100,  # new default value

            'speed_min': .2,
            'speed_mean': 1,
            'speed_std': 1,
            'speed_steps': 3,

            'separation': 5,  # just used in animation

            'step_limit': 3600,

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
        self.total_time = 0.0

        # Initialise station
        self.set_station()
        self.boundaries = np.array([[0, 0], [self.width, self.height]])

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
            self.width = 200
            self.height = 400
            self.gates_locations =\
                np.array([[0, self.height/2],  # south side
                          [20, self.height], [170, self.height],  # west side
                          [20, 0], [170, 0],  # east side
                          [self.width, 60], [self.width, 125],  # north side
                          [self.width, 200], [self.width, 275],  # north side
                          [self.width, 340]])  # north side
            self.gates_in = len(self.gates_locations)
            self.gates_out = len(self.gates_locations)
            self.clock = Agent(self, self.pop_total)
            self.clock.size = 10.0
            self.clock.location = [self.width/2.0, self.height/2.0]
            self.clock.speed = 0.0
        else:
            self.gates_locations = np.concatenate([
                Model._gates_init(0, self.height, self.gates_in),
                Model._gates_init(self.width, self.height, self.gates_out)])
            # create a clock outside the station.
            self.clock = Agent(self, self.pop_total)
            self.clock.speed = 0.0

            if(self.station is not None):
                warnings.warn(
                    "The station parameter passed to the model is not valid; "
                    "Using the default station.",
                    RuntimeWarning
                )

    def is_within_bounds(self, agent, loc):
        return all((self.boundaries[0] + agent.size) < loc) and\
               all(loc < (self.boundaries[1] - agent.size))

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

            t = 1.0
            while (t>=0):
                collisionTable, tmin = self.get_collisionTable()
                if (tmin > t):
                    [agent.step(t) for agent in self.agents]
                    self.total_time += t
                    t -= tmin
                else:
                    tmin *= 0.98  # stop just before the collision
                    t -= tmin
                    [agent.step(tmin) for agent in self.agents]
                    wiggleTable = self.get_wiggleTable(collisionTable, tmin)
                    [self.agents[i].set_wiggle() for i in wiggleTable]
                    self.total_time += tmin

                if self.do_history:
                    state = self.get_state('location2D')
                    self.history_state.append(state)
                    [agent.history() for agent in self.agents]

            self.step_id += 1
        else:
            if self.do_print and self.status == 1:
                print(f'StationSim {self.unique_id} - Everyone made it!')
                self.status = 0

    # information about next collision
    def get_collisionTable(self):
        '''
        Returns the time of next colision (tmin) and a table with 
        information about every possible colision:
        - collisionTable[0]: collision time
        - collisionTable[1]: agent agent.unique_id
        '''
        collisionTable = []
        for i in range(self.pop_total):
            if (self.agents[i].status == 1):
                collisionTime = self.agents[i].get_collisionTimeWall()
                collision = (collisionTime, i)
                collisionTable.append(collision)

                collisionTime =\
                    self.agents[i].get_collisionTime2Agents(self.clock)
                collision = (collisionTime, i)
                collisionTable.append(collision)

                for j in range(i+1, self.pop_total):
                    if (self.agents[j].status == 1):
                        collisionTime = self.agents[i].\
                            get_collisionTime2Agents(self.agents[j])
                        collision = (collisionTime, i)
                        collisionTable.append(collision)
                        collision = (collisionTime, j)
                        collisionTable.append(collision)

        try:
            tmin = min(collisionTable)
            tmin = tmin[0]
        except:
            tmin = 1.0e300

        return collisionTable, tmin

    def get_wiggleTable(self, collisionTable, time):
        '''
        Returns a table with the agent.unique_id of all agents that
        collide in the specified time. A tolerance time is used to
        capture almost simultaneous collisions.

        Each line in the collisionTable has 2 columns:
        - Column 0: collision time
        - Column 1: agent.unique_id
        '''
        return set([line[1] for line in collisionTable
                    if (abs(line[0] - time) < self.tolerance)])

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

    def get_data(self, time_id, agents=None, sensor='frame'):
        '''
        Save all locations of all agents. there are many ways to
        organize this information. In principle, each frame will be
        stored in a different file.
        '''
        directory = sensor + '_' + time_id
        if not(os.path.exists(directory)):
            os.mkdir(directory)
        locs = np.array([agent.history_locations for agent in
                         self.agents[:agents]]).transpose((1, 2, 0))        
        if(sensor == 'frame'):
            for frame in range (self.step_id):
                save_file = open(directory+'/frame_'+ str(frame+1) +'.dat', 'w')
                print('#agentID', 'x', 'y', file=save_file)
                x = locs[frame-1][0]
                y = locs[frame-1][1]
                for agent in range(self.pop_total):
                    if(x[agent]!=None):
                        print(agent, x[agent], y[agent], file=save_file)
                save_file.close()
        elif(sensor == 'activation'):
            save_file = open(directory+'/activation.dat', 'w')
            print('#agentID', 'time_activation', 'gate_in', 'gate_out', 'speed', 'loc_desireX', 'loc_desireY', file=save_file)
            for agent in self.agents:
                print(agent.unique_id, agent.step_start, agent.gate_in, agent.gate_out, agent.speed, agent.loc_desire[0], agent.loc_desire[1], file=save_file)
                #print(agent.unique_id, agent.step_start, agent.loc_start[0], agent.loc_start[1], agent.gate_out, file=save_file)
            save_file.close()
        elif(sensor == 'trails'):
            for agent in self.agents:
                save_file = open(directory+'/agent_{}.dat'.format(agent.unique_id), 'w')
                loc = agent.history_locations
                for xy in loc:
                    if(xy[0]!=None):
                        print(xy[0], xy[1], file=save_file)
                save_file.close()

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
                                     self.agents]),
            # 'GateWiggles': sum(wig[0]<self.gates_space for wig in
            # self.history_wiggle_locs)/self.pop_total
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
