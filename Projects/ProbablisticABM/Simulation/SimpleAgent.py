from Simulation.Steppable import Steppable


class SimpleAgent(Steppable):

    def __init__(self, **kwargs):
        self.attributes = kwargs
