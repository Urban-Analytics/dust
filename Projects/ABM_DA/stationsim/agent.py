"""
Agent.py
@author: ksuchak1990
date_created: 19/04/10
A separate file for a class representing a generic agent for StationSim.
"""

# Imports
import numpy as np

# Agent Class
class Agent:
    """
    A class representing a generic agent for the StationSim ABM.
    """
    def __init__(self, model, unique_id):
        """
        Initialise a new agent.

        Creates a new agent and gives it a randomly chosen entrance, exit, and
        desired speed. All agents start with active state 0 ('not started').
        Their initial location (** (x,y) tuple-floats **) is set
        to the location of the entrance that they are assigned to.

        :param model: a pointer to the station sim model that is creating this agent
        """
        # Required
        self.unique_id = unique_id
        self.active = 0  # 0 Not Started, 1 Active, 2 Finished
        model.pop_active += 1

        # Choose at random at which of the entrances the agent starts
        self.location = model.loc_entrances[np.random.randint(model.entrances)]
        self.location[1] += model.entrance_space * (np.random.uniform() - .5)
        self.loc_desire = model.loc_exits[np.random.randint(model.exits)]

        # Parameters
        # model.entrance_speed -> the rate at which agents enter
        # self.time_activate -> the time at which the agent should become active
        # time_activate is exponentially distributed based on entrance_speed
        self.time_activate = np.random.exponential(model.entrance_speed)
        # The maximum speed that this agent can travel at:
        # self.speed_desire = max(np.random.normal(model.speed_desire_mean, model.speed_desire_std), 2*model.speed_min)  # this is not a truncated normal distribution
        self.speed_desire = model.speed_min - 1
        while self.speed_desire <= model.speed_min:
            self.speed_desire = np.random.normal(model.speed_desire_mean, model.speed_desire_std)
        # if they can wiggle faster than they can move they may beat the expected time
        self.wiggle = min(self.speed_desire, model.wiggle)
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
            norm_diff = self.distance(self.location, self.loc_desire)
            self.time_expected = (norm_diff - model.exit_space) / self.speed_desire

    @staticmethod
    def is_within_bounds(boundaries, new_location):
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
        diff = self.loc_desire - self.location
        lerp_vector = diff / self.distance(self.loc_desire, self.location)
        for speed in self.speeds:
            # Direct
            new_location = self.location + speed * lerp_vector
            if not Agent.collision(model, new_location):
                break
            elif speed == self.speeds[-1]:
                # Wiggle
                # randint is upper-bound exclusive, so 2 instead of 1
                new_location = self.location + self.wiggle*np.random.randint(-1, 1+1, 2)
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
        if not model.is_within_bounds(new_location):
            collide = True
        elif Agent.neighbourhood(model, new_location):
            collide = True
        else:
            collide = False
        return collide

    @classmethod
    def neighbourhood(cls, model, new_location):
        """
        XXXX WHAT DOES THIS DO??  Answer:
        This method finds whether or not nearby neighbours are a collision.

         :param model:        the model that this agent is part of
         :param new_location: the proposed new location that the agent will move to
         (a standard (x,y) floats-tuple)
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
        lerp is a intensively used method hence profiling and adjustments have been made
        see 'github dust/Projects/awest/code/experiments/lerp.py' for more understanding.
        """
        distance = Agent.distance(loc1, loc2)
        loc = loc2 + speed * (loc1 - loc2) / distance
        return loc

    def exit_query(self, model):
        """
        Determine whether the agent should leave the model and, if so,
        remove them. Otherwise do nothing.
        """
        if self.distance(self.location, self.loc_desire) < model.exit_space:
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

    @staticmethod
    def distance(loc1, loc2):
        """
        A helpful function to calculate the distance between two points.
        This simply takes the square root of the sum of the square of the elements.
        This appears to be faster than using np.linalg.norm.
        No doubt the numpy implementation would be faster for large arrays.
        Fortunately, all of our norms are of two-element arrays.
        :param arr:     A numpy array (or array-like DS) with length two.
        :return norm:   The norm of the array.
        """
        x = loc1[0] - loc2[0]
        y = loc1[1] - loc2[1]
        return (x * x + y * y) ** 0.5
