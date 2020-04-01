class Environment:
	def __init__(self, width=1000, height=1000):
		self.width = width
		self.height = height
		self.agent = None
		self.doors = []

	def add_door(self, door):
		self.doors.append(door)

	def get_env_size(self):
		size = (self.height, self.width)
		return size


