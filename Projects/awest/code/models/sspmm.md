# sspmm
StationSim or Mike's model, is an agent-based-model simulating a corridor in which *agents* *move* from their *entrance* to their *exit*.  The time steps are discrete and movement is done sequentially.  The movement ensures agents avoid walking into others in their *separation* radius, and if they cannot move towards their exit they move a small random amount.

Text within the square brackets is for those who want extra detail and work in progress.
\[*Agents* are sometimes called *passengers* or *pedestrians* when viewed in the context of a train. In a similar light *move*ment is considered as *walk*ing, and *entrances* and *exits* may be *gates*.\]

### Objectives
We would like to understand the circumstances which lead to congestion in pedestrian movement. For this a set of parameters is provided with the model which can be change. An animation of the agents moving is projected, and a pair of final graphs showing the trails of finished agents, and time taken. By changing the variables the mean time to exit the corridor may decrease.

\[Another objective was to use as few `imports` as possible so that there is no mysterious operations. This is not always optimal, but is explainatory.\]

\[For more simplicity elements have be omitted from this code, short things like saving the agent location, and timing processes.\]

### Input
```Python
if __name__ == '__main__':
	random = NextRandom()
	model_params = {
		'width': 200,
		'height': 100,
		'population': 300,
		'entrances': 3,
		'exits': 2,
		'gate_space': 4,
		'speed_min': .1,
		'speed_desire_min': 1,
		'speed_desire_max': 5,
		'initial_separation': 1,
		'separation': 5,
		'batch_iterations': 200,
		'do_save': True,
		'do_time': True,
		'do_ani': True
	}
	Model(**model_params).batch()
```


# Model
Initialising everything is a mess. There is so many things stored here.
```Python
	def __init__(self, width, height, population, entrances, exits, gate_space, speed_min, speed_desire_min, speed_desire_max, initial_separation, separation, batch_iterations=10, do_save=False, do_time=False, do_ani=False):
		# Batch Details
		self.time = 0
		self.batch_iterations = batch_iterations
		self.do_save = do_save
		if self.do_save:
			self.time_taken = []
		self.do_time = do_time
		self.do_ani= do_ani
		# Model Parameters
		self.boundaries = np.array([[0, 0], [width, height]])
		self.pop_total = population
		self.pop_active = 0
		self.pop_finished = 0
		self.entrances = entrances
		self.exits = exits
		self.gate_space = gate_space
		self.speed_min = speed_min
		self.speed_desire_min = speed_desire_min
		self.speed_desire_max = speed_desire_max
		self.initial_separation = initial_separation
		self.separation = separation
		# Initialise
		self.initialise_gates()
		self.initialise_agents()
		return
```
The step is so simple.
```Python
def step(self):
	self.time += 1
	for agent in self.agents:
		agent.step(self)
	return
```
\[For future reference, I use `method(agent, model)` instead of defining `agent.model` as this required less RAM in my testing.\]


### Initialising Entances and Exits
Entances are exits are created on either wall. Let's draw a picture to see how.
```Python
def initialise_gates(self):
	width = self.boundaries[1, 0]
	height = self.boundaries[1, 1]
	self.span = height / max(self.entrances, self.exits) / self.gate_space
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
```
\[This code is not pretty and creating gates could be done in a more eligant way. We could even not define between entrances and exits.\]

### Initialising Agents
All agents are created at the start (with minimal information), in the set `agents`.
```Python
def initialise_agents(self):
	self.agent_count = 0
	self.agents = set([Agent(self) for _ in range(self.pop_total)])
	return
```

# Agents
Agents are a class and initialised with as little as possible, their parameters and variables are created in their activation. The agent id is simple given as a number in the range \[0, population).  A less mistakable uuid method could be used, however a problems hasn't arisen with this.
```Python
def __init__(self, model):
	self.unique_id = model.agent_count
	model.agent_count += 1
	self.active = 0
	self.location = np.zeros(2)
	return
```
At each step agents can `activate` if not *active* (0), if they are *active* (1) they `move` and if they're near the exit they `exit_query` and *finish* (2).
```Python
def step(self, model):
	if self.active == 0:
		self.activate(model)
	elif self.active == 1:
		self.move(model)
		self.exit_query(model)
	return
```
\[The step process is made as simple as possible such that adding data assimilation may be a seamless as possible.\]


### `activate`
At each step if an agent is not active they get a new attempt to activate (or disembark from the train).  A new attempt means they're given a new entrance and initial location. Since all agents are created at the beginning an `inital_separation` is used to determine the speed at which agents 'disembark'.
```Python
def activate(self, model):
	# Location
	entrance = random.np_integer(model.entrances - 1)
	self.location = model.loc_entrances[entrance][0]
	self.location[1] += model.span * random.np_uniform(-.5, +.5)
	# Empty space to step off of 'train'
	self.separation = model.initial_separation
	if not self.collision(model, self.location):
		self.active = 1
		model.pop_active += 1
		# Parameters
		self.loc_desire = model.loc_exits[random.np_integer(model.exits - 1)][0]
		self.speed_min = model.speed_min
		self.speed_desire = max(random.np_gaussian(model.speed_desire_max), model.speed_desire_min)
		self.separation = model.separation
	return
```

### `move`
This method determines the new location of an agent.
First at decreasing speeds and linear interpolation (`lerp`) is used.
At each speed collision detection (`collision`) is used if there is no collision the loop is broken and this is the new location for this agent.
If they can't move towards their destination, they *wiggle*, this is a random unit pertubation to location in any direction.
Bound the new location within the corridor.
```Python
def move(self, model):
	# Decreasing Speeds
	speeds = np.linspace(self.speed_desire, self.speed_min, 15)
	for speed in speeds:
		new_location = self.lerp(self.loc_desire, self.location, speed)
		if not self.collision(model, new_location):
			break
	# Wiggle
	if speed == self.speed_min:
		new_location = self.location + random.np_integer(low=-1, high=+1, shape=2)
	# Boundary check
	within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
	if not within_bounds:
		new_location = np.clip(new_location, model.boundaries[0], model.boundaries[1])
	# Move
	self.location = new_location
	return
```
The final line changes the agent's location to the new location. For simultanious movement `new_location` would be saved and updated after all agents have moved. \[For parallisation this would be nessasary for repeatability.\]

#### `collision`
Is a method for determining if there is anything in a location.
  First checks for out of bounds using `within_bounds` a boolean question.
  Then calls `neighbourhood` for agents in the way and checks if the list is empty.
```Python
def collision(self, model, new_location):
	within_bounds = all(model.boundaries[0] <= new_location) and all(new_location <= model.boundaries[1])
	if not within_bounds:
		collide = True
	elif self.neighbourhood(model, new_location):
		collide = True
	else:
		collide = False
	return collide
```

#### `neighbourhood`
Uses a `for loop` to determine if each agent's location is in the separation radius of the initial agent's new location.
```Python
def neighbourhood(self, model, new_location, just_one=True, forward_vision=True):
	neighbours = []
	for agent in model.agents:
		if agent.active == 1:
			if forward_vision and agent.location[0] < new_location[0]:
				distance = self.separation + 1
			else:
				distance = np.linalg.norm(new_location - agent.location)
			if distance < self.separation and agent.unique_id != self.unique_id:
				neighbours.append(agent)
				if just_one:
					break
	return neighbours
```
\[Neighbourhood searching is called a lot, several times per agent per iteration, and has a high comutational cost (cc) of O(NlogN). Better methods could be implemented, such as a *kd-tree*, which is updated each step (this would give a semi-simultaneous stepping on its own) with cc of O(N) for search plus O(logN) for creation. Or related trees for reducing cc.\]

#### `lerp`
Is a standard method for determining a position given a inital location speed, and destination. In essence creates a velocity from a speed.
```Python
def lerp(self, loc1, loc2, speed):
	distance = np.linalg.norm(loc1 - loc2)
	loc = loc2 + speed * (loc1 - loc2) / distance
	return loc
```
\[`lerp` too is called a lot, and the euclidean distance (p=2-norm) is computationally expensive, a cheaper alternative might be a p=1-norm/sqrt(2) (sometimes taxicab norm or manhattan norm), but this is an approximation, where difference can be up to 30\%.\]

### `exit_query`
If the agent is within radius (`model.span`) of the exit, they exit.
```Python
def exit_query(self, model):
	if np.linalg.norm(self.location - self.loc_desire) < model.span:
		self.active = 2
		model.pop_active -= 1
		model.pop_finished += 1
	return
```
\[Again this uses `np.linalg.norm` (p=2-norm), this is not called as often. But it is porbably more acceptable to exit less precisely, then move with error.\]


## Table of Variables
|     . |                . |                 . |   |
|-------|------------------|-------------------|---|
| model | time             | variable          |   |
|     . | boundaries       | parameter         |   |
|     . | pop_total        | parameter         |   |
|     . | pop_active       | variable          |   |
|     . | pop_finished     | variable          |   |
|     . | extrances        | parameter         |   |
|     . | exits            | parameter         |   |
|     . | gate_space       | parameter         |   |
|     . | speed_min        | parameter         |   |
|     . | speed_desire_max | parameter         |   |
|     . | speed_desire_min | parameter         |   |
|     . | separation_init  | parameter         |   |
|     . | separation       | parameter         |   |
|     . | gate_span        | parameter         |   |
|     . | loc_entances     | parameter         |   |
|     . | loc_exits        | parameter         |   |
|     . | agent_count      | variable          |   |
|     . | agents           | set of objects    |   |
| agent | unique_id        | parameter         |   |
|     . | active           | variable          |   |
|     . | location         | variable          |   |
|     . | separation       | parameter         |   |
|     . | loc_desire       | parameter         |   |
|     . | speed_min        | parameter         |   |
|     . | speed_desire     | parameter         |   |
|     . | start_time       | parameter         |   |
|     . | history_loc      | list of locations |   |
