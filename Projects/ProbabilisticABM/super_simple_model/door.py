from matplotlib.patches import Rectangle

class Door(Rectangle):
	def __init__(self, xy=None, width=None, height=None, fill=False):
		Rectangle.__init__(self, xy, width, height, fill)