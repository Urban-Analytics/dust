# MesaFree
from random import shuffle, random
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, model, unique_id, location):
        self.model = model
        self.unique_id = unique_id
        # Parameters
        self.location = location
        return

    def step(self):
        self.move()
        return

    def move(self):
        x, y = self.location
        x += 1
        x %= self.model.width
        y += 1
        y %= self.model.height
        self.location = (x, y)
        return

class Model:

    def __init__(self, width=500, height=500, population=100):
        self.step_id = 0
        self.agents = []
        # Parameters
        self.width = width
        self.height = height
        self.population = population
        # Initial Condition
        self.initial_condition()
        return

    def step(self, random_order=False):
        self.step_id += 1
        if random_order:
            shuffle(self.agents)
        for agent in self.agents:
            agent.step()
        return

    def initial_condition(self):
        for unique_id in range(self.population):
            location = (random() * self.width, random() * self.height)
            agent = Agent(self, unique_id, location)
            self.add(agent)
        return

    def add(self, agent):
        self.agents.append(agent)
        return

    def remove(self, agent):
        self.agents.remove(agent)
        return


if True:  # Run
    model_params = {
        "width": 500,
        "height": 500,
        "population": 10
    }
    model = Model(**model_params)
    for _ in range(100):  # Be careful with your time
        model.step()
        for agent in model.agents:
            plt.plot(*agent.location, '.k')  # for python2.7 use plt.plot(agent.location[0], agent.location[1], '.k')
        plt.axis([0, model.width, 0, model.height])
        plt.pause(.05)
        plt.clf()
