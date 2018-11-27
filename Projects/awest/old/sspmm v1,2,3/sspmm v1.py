# StationSim.py
'''
StationSim (aka Mike's model) converted into python.

Todos:
Allow agents to start at different times
Write a read me
'''
import numpy as np
import matplotlib.pyplot as plt


class NextRandom:
    '''
    This defines the random number systems to be used.

    The random numbers will all come from np.random.random(), which
    produces uniform random doubles in the range [0,1).

    Also a seed is set so all numbers will come from the same stream.
    '''
    def __init__(self):
        np.random.seed(303)
        self.random_number_usage = 0

    def uniform(self, high=1, low=0, shape=1):
        r = np.random.random(shape)
        r = (high - low) * r - low
        self.random_number_usage += np.size(r)
        return r

    def gaussian(self, mu=0, sigma=1, shape=1):
        '''
        Converting distributions is not trivial. Here the Box-Muller
        Transform is used due to its efficiency and ease of implimentation.
        https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        for improvement see
        https://en.wikipedia.org/wiki/Ziggurat_algorithm
        '''
        # For shapes with an odd number of elements
        numel = np.prod(shape)
        oddness = numel % 2
        # Two sets of uniform random numbers
        r1 = np.random.random(numel // 2 + oddness)
        r2 = np.random.random(numel // 2 + oddness)
        # Two sets of gaussian random numbers
        a = np.sqrt(-2 * np.log(r1))
        b = 2 * np.pi * r2
        r1 = a * np.cos(b)
        r2 = a * np.sin(b)
        # Reshape and adjust
        r = np.ravel([r1, r2])    # Flattens pair
        r = r[:numel]             # Adjusts for oddness
        r = np.reshape(r, shape)  # Reshape to the requested shape
        r = r * sigma**2 + mu     # Adjust mean and variance
        self.random_number_usage += np.size(r1) + np.size(r2)
        return r

    def integer(self, high=1, low=0, shape=1):
        r = np.random.random(shape)
        r = np.round((high - low + 1) * r + low + .5) - 1
        r = np.int_(r)
        self.random_number_usage += np.size(r)
        return r

    def shuffle(self, x):
        # x - is a vector
        N = np.size(x)
        L = np.arange(N, dtype=int)  # Old Order
        I = np.zeros(N, dtype=int)   # New Order
        for i in range(N):
            l = self.integer(np.size(L) - 1)
            I[i] = L[l]
            L = np.delete(L, l)
        x = x[I]
        self.random_number_usage += N * (N - 1) // 2
        return x

    def sign(self, sigma=1, shape=1):
        r = self.integer(shape=shape)
        r = sigma * (2 * r - 1)
        return r

    '''
    These are the equivalent functions using numpy.random methods
    https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    '''

    def np_uniform(self, high=1, low=0, shape=1):
        r = np.random.random(shape)
        r = r * (high - low)
        self.random_number_usage += np.size(r)
        return r

    def np_gaussian(self, mu=0, sigma=1, shape=1):
        r = np.random.randn(*shape)
        r = r * sigma**2 + mu
        self.random_number_usage += np.size(r)
        return r

    def np_integer(self, high=1, low=0, shape=1):
        r = np.random.integers(low, high, shape)
        self.random_number_usage += np.size(r)
        return r

    def np_shuffle(self, x):
        np.random.shuffle(x)
        self.random_number_usage += np.size(x)
        return x


class Agent:

    def __init__(self, model):
        # Unique id
        self.unique_id = model.agent_count
        model.agent_count += 1
        # Model
        self.model = model
        # Gates
        entrance = random.integer(model.entrances - 1)
        exit = random.integer(model.exits - 1)
        # Location
        self.location = self.model.loc_entrances[entrance][0]
        self.location[1] += model.span * random.uniform(-.5, .5)
        self.loc_desire = self.model.loc_exits[exit][0]
        # Velocity
        self.speed_min = model.speed_min
        self.speed_desire = max(random.gaussian(model.speed_max), model.speed_min)
        # Parameters
        self.separation = model.separation
        # Save
        self.start_time = model.time
        self.history_loc = []
        return

    def step(self):
        self.move()
        self.save()
        self.exit_query()
        return

    def exit_query(self):
        if np.linalg.norm(self.location - self.loc_desire) < self.model.span:
            self.model.remove(self)
            self.model.time_taken.append(self.model.time - self.start_time)
        return

    def save(self):
        self.history_loc.append(self.location)
        return

    def move(self):
        '''
        Description:
            This mechanism moves the agent. It checks certain conditions for collisions at decreasing speeds.
            First check for direct new_location in the desired direction.
            Second check for a new_location in a varied desired direction.
            Third check for a new_location with a varied current direction.

        Dependencies:
            collision - any agents in radius
                neighbourhood - find neighbours in radius
            lerp      - linear interpolation
            loc_perturb - perturb location in y-axis

        Arguments:
            self

        Returns:
            new_location
        '''
        # For decreasing speed
        speeds = np.linspace(self.speed_desire - .001, self.speed_min, 15)  # the .001 is to avoid floating errors
        for speed in speeds:
            # direct movement
            new_location = self.lerp(self.loc_desire, self.location, speed)
            if not self.collision(new_location):
                break
            # wiggle movement
            vel = np.sqrt(self.speed_desire**2 - speed**2)  # approx norm is speed desire
            loc = np.copy(self.location)
            loc[1] += random.sign(vel/2)
            new_location = self.lerp(self.loc_desire, loc, speed)
            if not self.collision(new_location):
                break
        # minimal movement
        if speed == self.speed_min:
            new_location = self.location  # lerp(self.loc_desire, self.location, speed)
        self.location = new_location
        return

    def collision(self, new_location):
        '''
        Description:
            Determine whether or not there is another object at this location.
            Requires get neighbour from mesa?

        Dependencies:
            neighbourhood - find neighbours in radius

        Arguments:
            self.model.boundaries
                ((f, f), (f, f))
                A pair of tuples defining the lower limits and upper limits to the rectangular world.
            new_location
                (f, f)
                The potential location of an agent.

        Returns:
            collide
                b
                The answer to whether this position is blocked
        '''
        within_bounds = np.all(self.model.boundaries[0] < new_location) and np.all(new_location < self.model.boundaries[1])
        if not within_bounds:
            collide = True
        elif self.neighbourhood(new_location):
            collide = True
        else:
            collide = False
        return collide

    def neighbourhood(self, new_location, just_one=True, forward_vision=True):
        '''
        Description:
            Get agents within the defined separation.

        Arguments:
            self.unique_id
                i
                The current agent's unique identifier
            self.separation
                f
                The radius in which to search
            self.model.agents
                <agent object>s
                The set of all agents
            new_location
                (f, f)
                A location tuple
            just_one
                b
                Defines if more than one neighbour is needed.
            forward_vision
                b
                Restricts separation radius to a semi-circle only infront.

        Returns:
            neighbours
                (agent object)s
                A set of agents in a region
        '''
        neighbours = []
        for agent in self.model.agents:
            if agent.location[0] - new_location[0] < 0 and forward_vision:
                distance = self.separation + 1
            else:
                distance = np.linalg.norm(new_location - agent.location)
            if distance < self.separation and agent.unique_id != self.unique_id:
                neighbours.append(agent)
                if just_one:
                    break
        return neighbours

    def lerp(self, loc1, loc2, speed):
        '''
        Description:
            Linear interpolation at a constant rate
            https://en.wikipedia.org/wiki/Linear_interpolation

        Arguments:
            loc1
                (f, f)
                Point One defining the destination position
            loc2
                (f, f)
                Point Two defining the agent position
            speed
                f
                The suggested speed of the agent

        Returns:
            loc
                (f, f)
                The location if travelled at this speed
        '''
        distance = np.linalg.norm(loc1 - loc2)
        loc = loc2 + speed * (loc1 - loc2) / distance
        return loc


class Model:

    def __init__(self, width, height, population, entrances, exits, speed_min, speed_max, separation):
        self.time = 0
        self.agents = []
        self.time_taken = []
        self.agent_count = 0
        # Model Parameters
        self.boundaries = np.array([[0, 0], [width, height]])
        self.pop_total = population
        self.pop_current = 0
        self.entrances = entrances
        self.exits = exits
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.separation = separation
        # Initialise
        self.initialise_gates()
        self.initialise_agents()
        return

    def initialise_gates(self):
        width = self.boundaries[1, 0]
        height = self.boundaries[1, 1]
        self.span = height / max(self.entrances, self.exits) / 16  # 1/8th of the wall is gate
        # Entrances
        self.loc_entrances = np.zeros((self.entrances, 2))
        self.loc_entrances[:, 0] = 0
        if self.entrances == 1:
            self.loc_entrances[0, 1] = height / 2
        else:
            self.loc_entrances[:, 1] = np.linspace(height / 4, 3 * height / 4, self.entrances)
        # Exits
        self.loc_exits = np.zeros((self.exits, 2))
        self.loc_exits[:, 0] = width
        if self.exits == 1:
            self.loc_exits[0, 1] = height / 2
        else:
            self.loc_exits[:, 1] = np.linspace(height / 4, 3 * height / 4, self.exits)
        return

    def initialise_agents(self):  # different start times
        for unique_id in range(self.pop_total):
            agent = Agent(self)
            self.add(agent)
        return

    def step(self, random_order=False):
        self.time += 1
        # self.initialise_agents()
        if random_order:
            random.shuffle(self.agents)
        for agent in self.agents:
            agent.step()
        return

    def add(self, agent):
        self.pop_total += 1
        self.pop_current += 1
        self.agents.append(agent)
        return

    def remove(self, agent):
        self.pop_current -= 1
        self.agents.remove(agent)
        return

    def batch(self, I=100):
        self.initialise_agents()
        for i in range(I):
            self.step()
            # self.ani_agents()
            if self.pop_current == 0:
                print('Everyone made it!')
                break
        self.stats()
        # self.hist_time()
        self.plot_agents()
        return

    def ani_agents(self):
        plt.clf()
        for agent in self.agents:
            plt.plot(*agent.location, '.k')
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.pause(.1)
        return

    def plot_agents(self):
        for agent in self.agents:
            locs = np.array(agent.history_loc).T
            plt.plot(locs[0], locs[1])
        plt.axis(np.ravel(self.boundaries, 'F'))
        plt.show()
        return

    def hist_time(self):
        plt.clf()
        plt.hist(self.time_taken)
        plt.show()
        return

    def stats(self):
        print()
        print('Stats:')
        print('Finish Time: ' + str(self.time))
        print('Random number usage: ' + str(random.random_number_usage))
        print('Unfinished agents: ' + str(self.pop_current) + '/' + str(self.pop_total))
        print('Average time taken: ' + str(np.mean(self.time_taken)) + 's')
        return


random = NextRandom()
model_params = {
    "width": 100,
    "height": 100,
    "population": 700,
    "entrances": 3,
    "exits": 2,
    "speed_min": .01,
    "speed_max": 1,
    "separation": 2
}
model = Model(**model_params)
model.batch(100)
print()