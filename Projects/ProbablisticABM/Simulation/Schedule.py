from copy import deepcopy
from multiprocessing import Pool
from multiprocessing import cpu_count
import time

class Schedule:

    def __init__(self):
        self.n_steps = 0
        self.steps = []

    def run(self, agents, step):
        i = 0
        start = time.time()
        while i < self.n_steps:
            self.steps.append(deepcopy(agents))
            agents = list(map(step, agents))
            i = i + 1
        end = time.time()
        print(end - start)
        return agents

    def run_parallel(self, agents, step):
        i = 0
        p = Pool(cpu_count())
        start = time.time()
        while i < self.n_steps:
            self.steps.append(deepcopy(agents))
            agents = list(p.map(step, agents))
            i = i + 1
        end = time.time()
        print(end - start)
        return agents

