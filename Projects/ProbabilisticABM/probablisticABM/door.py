from matplotlib.patches import Rectangle


class Door(Rectangle):
    """
    Door object extended from the matplotlib rectangle patch object. Simple identifier is also added.
    """

    def __init__(self, key=None, xy=None, width=None, height=None, fill=False):
        Rectangle.__init__(self, xy, width, height, fill)
        self.id = key
