# basicModel
'''
An agent based model so basic, a `child` made it.
'''
import numpy as np
import matplotlib.pyplot as plt


class Agent:

    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.active = np.random.randint(2)
        self.location = np.random.uniform(size=2)  # Noncompulsory
        return

    def step(self):
        self.move()  # Noncompulsory
        return

    def move(self):  # Noncompulsory
        self.location += .01
        self.location %= 1
        return


class Model:

    def __init__(self, population):
        self.params = (population,)
        self.agents = list([Agent(unique_id) for unique_id in range(population)])
        self.boundaries = np.array([[0, 0], [1, 1]])
        return

    def step(self):  # Desgined
        [agent.step() for agent in self.agents]
        return

    def agents2state(self, do_ravel=True):  # Desgined
        state = [agent.location for agent in self.agents]
        if do_ravel:
            state = np.ravel(state)
        else:  # Noncompulsory
            state = np.array(state)
        return state

    def state2agents(self, state):  # Desgined
        for i in range(len(self.agents)):
            self.agents[i].location = state[2*i:2*i+2]
        return

    def ani(model):  # Noncompulsory
        state = model.agents2state(do_ravel=False).T
        plt.clf()
        plt.plot(state[0], state[1], '.k', alpha=.5)
        plt.axis((0, 1, 0, 1))
        plt.pause(.1)
        return
