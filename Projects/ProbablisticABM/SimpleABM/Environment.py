from copy import deepcopy


class Environment:
    def __init__(self, width=480, height=480, **kwargs):
        self.width = width
        self.height = height
        self.n_agents = 0
        self.agents = []
        self.environmentalObjects = []

        self.attributes = kwargs

    def populate(self, agent):
        i = 0
        while i < self.n_agents:
            agent.attributes['id'] = i
            self.agents.append(deepcopy(agent))
            i = i + 1
        # May move alerts to separate module later on.
        print('Populated with {} agents'.format(self.n_agents))
