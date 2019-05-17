from Simulation.Steppable import Steppable


class SimpleAgent(Steppable):

    def __init__(self, **kwargs):
        self.attributes = kwargs
        # self.attributes['deactivate'] = self.deactivate

    def deactivate(self):
        self.attributes['isActive'] = False
        self.attributes['completed'] = True
        print('Agent {} has left the station at exit {}'.format(self.attributes['id'],
                                                                self.attributes['d']))
