from matplotlib.patches import Rectangle


class Door(Rectangle):
	def __init__(self, id=None, xy=None, width=None, height=None, fill=False):
		Rectangle.__init__(self, xy, width, height, fill)
		self.id = id