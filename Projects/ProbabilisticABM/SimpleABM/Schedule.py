from copy import deepcopy
import sys


class Schedule:

    def __init__(self, **kwargs):
        self.n_steps = kwargs['n_steps']
        self.steps = []

    def run(self, agents, step):
        i = 0
        while i < self.n_steps:
            self.steps.append(deepcopy(agents))
            agents = list(map(step, agents))
            i = i + 1
        return self.steps
