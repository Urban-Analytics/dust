# Python Methods
'''
Abst: An analysis of methods for classes in Python.

There are three types of methods within a class,
Instance Methods
	def method0(self, arg1):
		pass
Class Methods
	@classmethod
	def method1(cls, agr1):
		pass
Static Methods
	@staticmethod
	def method2(arg1):
		pass
Ref: https://realpython.com/instance-class-and-static-methods-demystified/
The following code is an analysis of how methods are stored, as to avoid unnessary reproduction in our creation of 'Agents' and 'Models'.


TLDR:  Thankfully we don't have to worry about this.  Python uses pointers for identical methods.  Even when deepcopied.
'''


# Python
import copy
import numpy as np
import sys
hid = lambda x: hex(id(x))		  # memory location
syz = lambda x: sys.getsizeof(x)  # memory size

def local():

	def foo():
		pass

	def bar():
		pass

	print(hid(foo), hid(bar))
	print(syz(foo), syz(bar))


class Agent():

	def __init__(self):
		self.x = np.random.rand()

	def method0(self):
		return self.x+1

	@classmethod
	def method1(cls):
		return cls.x

	@staticmethod
	def method2(x):
		return x+2

	def method_id(self):
		print()
		print(hid(self), hid(self.method0), hid(self.method1), hid(self.method2))
		print(syz(self), syz(self.method0), syz(self.method1), syz(self.method2))


class Model():

	def __init__(self):
		self.agents = [Agent() for _ in range(5)]
		for agent in self.agents:
			agent.method_id()


class Filter():

	def __init__(self):
		Agent_Base = Agent()
		self.agents = [copy.deepcopy(Agent_Base) for _ in range(2)]
		for i, agent in enumerate(self.agents):
			if i == 1:
				agent.x += 1
			agent.method_id()
			print(agent.x, hid(agent.x), hid(agent))


if __name__=='__main__':
	# local()
	# Agent().method_id()
	# Model()
	# Filter()
	pass
