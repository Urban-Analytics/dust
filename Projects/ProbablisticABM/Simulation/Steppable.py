class Steppable:

    def __init__(self, **kwargs):
        self.attributes = kwargs

    def step(self):
        pass

    def get_object_attributes(self):
        return self.attributes
