from copy import deepcopy

class Schedule:

    def __init__(self):
        self.n_steps = 0
        self.steps = []

    def step(self, environment, step):
        i = 0
        while i < self.n_steps:
            self.steps.append(deepcopy(environment.agents))
            environment.agents = list(map(step, environment.agents))
            i = i + 1
